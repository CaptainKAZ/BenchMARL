from sympy import true
import torch
from wandb import restore
from benchmarl.algorithms import MappoConfig
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
        
        # 默认情况下，我们计划训练所有原始组
        new_train_map = self.original_group_map.copy()

        try:
            # 1. 从 batch 中计算胜率
            # 'shots_in_step' 来自您 layup.py 的 info() 函数
            # win_info = batch.get(("attacker", "info", "win_in_step"))[...,0,:]
            done_info = batch.get(("next", "done"))
            reason_codes_tensor = batch.get(("next", "attacker", "info", "termination_reason"))[...,0,:]
            reason_codes = reason_codes_tensor.squeeze(-1)
            dones_mask = done_info.squeeze(-1).bool()
            terminated_codes_in_batch = reason_codes[dones_mask] # 得到一个一维张量，长度不定
            win_rate = log_and_calculate_win_rate(terminated_codes_in_batch,{1,2,3,4,5})
            # total_shots_in_batch = win_info.sum().item()
            # # 'done' 标志着一个回合的结束
            total_dones_in_batch = done_info.sum().item()

            # # 课程学习，先学会进攻
            # if "defender" in new_train_map:
            #     del new_train_map["defender"]
            # return

            # # 计算在所有结束的回合中，由投篮导致的比率
            # win_rate = total_shots_in_batch / total_dones_in_batch if total_dones_in_batch > 0 else 0.0
            print(f"Win rate: {win_rate:.2f}")

            if self.experiment.n_iters_performed < 20: #or self.experiment.n_iters_performed % 50 < 3:
                self.experiment.train_group_map = new_train_map
                return

            # 2. 根据比率决定本次迭代要训练哪些组
            if win_rate < self.win_rate_threshold and total_dones_in_batch > 0:
                # 胜率低，只训练进攻方。我们从训练地图中移除防守方。
                if "defender" in new_train_map:
                    del new_train_map["defender"]
                print(f"\n[WinRateCurriculum] Win rate ({win_rate:.2f}) is LOW. Training groups: {list(new_train_map.keys())}")
            elif win_rate > 1 - self.win_rate_threshold and total_dones_in_batch > 0:
                # 胜率高，只训练防守方。我们从训练地图中移除进攻方。
                if "attacker" in new_train_map:
                    del new_train_map["attacker"]
                print(f"\n[WinRateCurriculum] Win rate ({win_rate:.2f}) is HIGH. Training groups: {list(new_train_map.keys())}")
            else:
                # 胜率达标，训练所有组 (new_train_map 已经是所有组了)
                if total_dones_in_batch > 0:
                    print(f"\n[WinRateCurriculum] Win rate ({win_rate:.2f}) is GOOD. Training all groups: {list(new_train_map.keys())}")
        
        except (KeyError, AttributeError) as e:
            # 如果在 batch 中找不到所需信息 (例如，在第一次迭代时)，则默认训练所有组
            print(f"\n[WinRateCurriculum] Could not compute win rate ({e}). Defaulting to train all groups.")
            pass

        # 3. 【核心】更新 Experiment 的 train_group_map
        # 这是我们与 Experiment 交互的“暴露接口”。
        # 下一个训练循环将只会遍历我们在这里设置的组。
        self.experiment.train_group_map = new_train_map

# checkpoint_path = "outputs/2025-07-06_19-39-05/mappo_layup_gru__c217740f_25_07_06-19_39_05/checkpoints"
checkpoint_pattern="outputs/**/checkpoints/*.pt"

if __name__ == '__main__':
    # 1. 定义预训练模型的路径
    restore_file_path = find_latest_checkpoint(checkpoint_pattern)
    print(f"found checkpoint: {restore_file_path}")
    if restore_file_path is None:
        exit(1)

    exp = Experiment.reload_from_file(restore_file_path)
    exp.seed = 0
    torch.manual_seed(1)
    exp.evaluate()