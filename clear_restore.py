from sympy import im, true
import torch
from wandb import restore
from benchmarl.algorithms import MappoConfig
from benchmarl.algorithms import IppoConfig
from benchmarl.environments import LayupTask # 替换为您重构后的新环境
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.gru import GruConfig
from benchmarl.models.gtrxl import GTrXLConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.experiment.callback import Callback
from tensordict import TensorDict, TensorDictBase
from typing import List, Set
import glob
import os
from datetime import datetime
from benchmarl.models import EnsembleModelConfig
from benchmarl.algorithms import EnsembleAlgorithmConfig
from torch.profiler import profile, ProfilerActivity
from collections import OrderedDict
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from benchmarl.models.common import SequenceModelConfig
# from torch.cuda.amp import GradScaler, autocast

def print_dict_paths(d, path=""):
    for key, value in d.items():
        current_path = f"{path}->{key}" if path else key
        # print(current_path)
        if isinstance(value, dict) or isinstance(value, TensorDict):
            print_dict_paths(value, current_path)
        else:
            print(current_path+" "+f"{type(value)}")

def find_latest_file(path, pattern='*'):
    """
    查找指定路径下最新的文件
    
    参数:
        path (str): 要搜索的目录路径
        pattern (str): 文件匹配模式，默认为所有文件
    
    返回:
        str: 最新文件的完整路径，如果没有文件则返回None
    """
    # 获取所有匹配的文件列表
    files = glob.glob(os.path.join(path, pattern))
    
    # 过滤掉目录，只保留文件
    files = [f for f in files if os.path.isfile(f)]
    
    if not files:
        return None
    
    # 按修改时间排序文件（最新的排在最前面）
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return files[0]

def find_latest_checkpoint(search_pattern: str):
    """
    查找匹配指定模式的所有文件，并返回最新的那个。

    Args:
        search_pattern (str): 用于搜索文件的 glob 模式。

    Returns:
        str: 最新文件的路径。如果找不到任何文件，则返回 None。
    """
    # 1. 使用 glob 找到所有匹配模式的文件
    file_list = glob.glob(search_pattern, recursive=True)

    # 2. 检查是否找到了文件
    if not file_list:
        print(f"警告：在模式 '{search_pattern}' 下没有找到任何文件。")
        return None

    # 3. 使用 max() 和 os.path.getmtime 找出最新的文件
    # os.path.getmtime 会返回文件的最后修改时间（一个时间戳）
    # max() 函数会根据这个时间戳来比较并找出“最大”的那个，也就是最新的文件
    latest_file = max(file_list, key=os.path.getmtime)

    return latest_file

REASON_CODE_LEGEND = {
    # 0: "回合未结束",
    1: "胜利: 投篮命中",
    2: "胜利: 对手犯规",
    3: "胜利: 对手失误-撞墙",
    4: "胜利: 对手失误-越线",
    5: "胜利: 对手友军误伤",
    11: "失败: 投篮被盖",
    12: "失败: 进攻超时",
    13: "失败: 己方犯规",
    14: "失败: 己方失误-撞墙",
    15: "失败: 己方失误-友军误伤",
}

def log_and_calculate_win_rate(terminated_codes: torch.Tensor, win_codes: Set[int]):
    """
    接收一个包含已终止回合原因码的一维张量，统计详细信息，并根据指定的“胜利码”计算胜率。

    Args:
        terminated_codes (torch.Tensor): 只包含已终止回合原因码的一维张量。
        win_codes (Set[int]): 一个包含所有被视作“胜利”的原因码的集合。
                               使用集合(set)可以高效地进行查找。

    Returns:
        float: 计算出的胜率 (0.0 到 1.0之间)。如果没有任何回合终止，则返回 0.0。
    """
    total_terminated = terminated_codes.numel()
    if total_terminated == 0:
        # 如果本批次没有回合结束，直接返回0
        return 0.0

    # --- 统计和打印详细信息 ---
    max_code = max(REASON_CODE_LEGEND.keys()) if REASON_CODE_LEGEND else 0
    # 确保张量是整数类型以用于 bincount
    counts = torch.bincount(terminated_codes.to(torch.int64), minlength=max_code + 1)
    
    print("\n" + "="*20 + " 回合结束原因统计 " + "="*20)
    print(f"本批次数据中有 {total_terminated} 个回合结束，详情如下:")
    
    for code, description in REASON_CODE_LEGEND.items():
        count = counts[code].item()
        if count > 0:
            percentage = 100 * count / total_terminated
            # 如果当前原因码是胜利条件之一，则在打印时进行标记
            is_win_str = " (胜利条件)" if code in win_codes else ""
            print(f"  - {description} (码 {code}): {count} 次 ({percentage:.2f}%) {is_win_str}")
    
    # --- 根据传入的 win_codes 计算胜率 ---
    total_wins = 0
    for code in win_codes:
        if code < len(counts):  # 确保原因码有效
            total_wins += counts[code].item()
    
    win_rate = total_wins / total_terminated
    
    print("-" * 58)
    print(f"指定的胜利条件码: {win_codes}")
    print(f"总胜利次数: {total_wins} / 总结束次数: {total_terminated}")
    print(f"胜率: {win_rate:.2%}")
    print("="*58 + "\n")
    
    # 返回计算出的胜率，以便在其他地方使用
    return win_rate


class WinRateCurriculum(Callback):
    """
    一个自定义回调，根据胜率动态调整训练的智能体组。

    Args:
        win_rate_threshold (float): 胜率的阈值。如果实际比率低于此值，
                                    将只训练进攻方。
    """
    def __init__(self, win_rate_threshold: float = 0.3):
        self.win_rate_threshold = win_rate_threshold
        # 这个变量将保存实验原始的训练组，以便我们恢复
        self.original_group_map = None
        print(f"[WinRateCurriculum] Callback initialized with threshold {self.win_rate_threshold}.")

    def on_setup(self):
        """
        在实验设置之初被调用一次。
        这是初始化和保存原始状态的最佳位置。
        """
        # 通过 self.experiment 可以访问到 Experiment 实例本身
        # 我们复制一份原始的 train_group_map，这是控制训练哪些组的关键
        self.original_group_map = self.experiment.train_group_map.copy()
        print(f"[WinRateCurriculum] Setup complete. Original training groups: {list(self.original_group_map.keys())}")

    def on_batch_collected(self, batch: TensorDictBase):
        """
        在每个数据批次收集完成之后，训练开始之前被调用。
        这是实现我们核心逻辑的地方。
        """
        try:
            # 1. 从 batch 中计算胜率
            # 'shots_in_step' 来自您 layup.py 的 info() 函数
            # win_info = batch.get(("attacker", "info", "win_in_step"))[...,0,:]
            done_info = batch.get(("next", "done"))
            reason_codes_tensor = batch.get(("next", "agents", "info", "termination_reason"))[...,0,:]
            reason_codes = reason_codes_tensor.squeeze(-1)
            dones_mask = done_info.squeeze(-1).bool()
            terminated_codes_in_batch = reason_codes[dones_mask] # 得到一个一维张量，长度不定
            win_rate = log_and_calculate_win_rate(terminated_codes_in_batch,{1,2,3,4,5})

            print(f"Win rate: {win_rate:.2f}")
        
        except (KeyError, AttributeError) as e:
            pass


# checkpoint_path = "outputs/2025-07-06_19-39-05/mappo_layup_gru__c217740f_25_07_06-19_39_05/checkpoints"
checkpoint_pattern="outputs/**/checkpoints/*.pt"

if __name__ == '__main__':
    # 1. 定义预训练模型的路径
    # restore_file_path = find_latest_file(checkpoint_path,"*.pt")
    # restore_file_path = find_latest_checkpoint(checkpoint_pattern)
    # print(f"found checkpoint: {restore_file_path}")
    # if restore_file_path is None:
    #     exit(1)

    # # # # 2. 加载检查点文件并只提取模型权重
    # print(f"Loading checkpoint from {restore_file_path}...")
    # checkpoint = torch.load(restore_file_path)
    # # print_dict_paths(checkpoint)
    # print("Successfully extracted model weights.")
    # 3. 配置并创建新环境的实验
    experiment_config = ExperimentConfig.get_from_yaml()
    # experiment_config.restore_file = restore_file_path

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").replace(":", "-")
    folder_name= f"outputs/{current_time}"
    os.makedirs(folder_name)
    experiment_config.save_folder = folder_name

    # 使用您重构后的新环境
    new_task = LayupTask.LAYUP.get_from_yaml() 

    algorithm_config = IppoConfig.get_from_yaml()
    # algorithm_config = MappoConfig.get_from_yaml()
    attention_model_config = AttentionConfig.get_from_yaml()
    attention_model_config.num_attention_layers = 2
    attention_model_config.final_mlp_hidden_layers = [64,64]
    model_config = SequenceModelConfig(
        model_configs=[
            attention_model_config,
            GruConfig.get_from_yaml(),
        ],
        intermediate_sizes=[
            64
        ],  # Nuber of intermediate outputs. List of size n_layers - 1
    )
    # model_config = EnsembleModelConfig({"attacker":attacker_model_config, "defender":defender_model_config})
    critic_model_config = AttentionConfig.get_from_yaml("benchmarl/conf/model/layers/attention_critic.yaml")

        

    # 创建一个全新的实验对象，所有状态都是初始化的
    experiment = Experiment(
        task=new_task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=114514,
        config=experiment_config,
        callbacks=[WinRateCurriculum()]
    )
    print("New experiment created with fresh training states (optimizers, buffers, etc.).")

    # 手动将预训练权重加载到新实验的模型中,actor和critic都恢复
    # 遍历新实验中的每一个智能体组
    # for group in experiment.group_map.keys():
    #     if group == "attacker" or True:
    #         loss_key = f"loss_{group}"
    #         if loss_key in checkpoint:
    #             print(f"Loading weights for group '{group}' from '{loss_key}'...")
    #             # experiment.losses[group] 是一个 LossModule，它包含了actor和critic网络
    #             # 加载它的状态字典，就会恢复网络的权重
    #             experiment.losses[group].load_state_dict(checkpoint[loss_key])
    #             print(f"Successfully loaded weights for group '{group}'.")
    #         else:
    #             print(f"Warning: No weights found for group '{group}' in the checkpoint. Using freshly initialized weights.")
    
    
    # 只恢复 actor 网络不恢复 critic
    # ACTOR_PREFIX = "actor_network_params."
    # for group in experiment.group_map.keys():
    #     if group == "defender" or True:
    #         loss_key = f"loss_{group}"
    #         if loss_key in checkpoint:
    #             print(f"Partially loading ONLY ACTOR weights for group '{group}'...")

    #             # 1. 从 checkpoint 中获取该 group 的完整状态字典
    #             full_group_state_dict = checkpoint[loss_key]

    #             # 2. 筛选出 actor 的所有参数（不再检查类型），并移除key的前缀
    #             #    这样就会把 __batch_size (torch.Size) 和 __device (NoneType) 也包含进来
    #             actor_state_dict = {
    #                 key.removeprefix(ACTOR_PREFIX): value
    #                 for key, value in full_group_state_dict.items()
    #                 if key.startswith(ACTOR_PREFIX)
    #             }

    #             # 3. 将筛选后的 state_dict 加载到 actor 网络中
    #             try:
    #                 actor_network = experiment.losses[group].actor_network_params
                    
    #                 # 使用 strict=False 增加加载的灵活性，这是一个好习惯
    #                 # 它会加载所有匹配的键，并忽略不匹配的键，从而避免因细微差异导致的错误
    #                 actor_network.load_state_dict(actor_state_dict, strict=False)
                    
    #                 print(f"Successfully loaded actor weights for group '{group}'.")
    #             except AttributeError:
    #                 print(f"[Error] Could not find attribute 'actor_network_params' in LossModule for group '{group}'. Please check the model definition.")
    #             except Exception as e:
    #                 print(f"[Error] Failed to load weights for group '{group}': {e}")

    print("The optimizer, replay buffer, and collector states remain freshly initialized as intended.")

    # 5. 开始在新环境上训练
    print("\nStarting training on the new environment...")

    experiment.run()