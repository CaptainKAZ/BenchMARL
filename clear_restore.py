import torch
from benchmarl.algorithms import MappoConfig
from benchmarl.algorithms import IppoConfig
from benchmarl.environments import LayupTask  # 替换为您重构后的新环境
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.gru import GruConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.models.attention import AttentionConfig
from benchmarl.models.mamba import MambaConfig
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
import argparse

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from benchmarl.models.common import SequenceModelConfig
from benchmarl.models.debug_utils import setup_model_logging

# from torch.cuda.amp import GradScaler, autocast


def print_dict_paths(d, path=""):
    for key, value in d.items():
        current_path = f"{path}->{key}" if path else key
        # print(current_path)
        if isinstance(value, dict) or isinstance(value, TensorDict):
            print_dict_paths(value, current_path)
        else:
            print(current_path + " " + f"{type(value)}")


def find_latest_file(path, pattern="*"):
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

    print("\n" + "=" * 20 + " 回合结束原因统计 " + "=" * 20)
    print(f"本批次数据中有 {total_terminated} 个回合结束，详情如下:")

    for code, description in REASON_CODE_LEGEND.items():
        count = counts[code].item()
        if count > 0:
            percentage = 100 * count / total_terminated
            # 如果当前原因码是胜利条件之一，则在打印时进行标记
            is_win_str = " (胜利条件)" if code in win_codes else ""
            print(
                f"  - {description} (码 {code}): {count} 次 ({percentage:.2f}%) {is_win_str}"
            )

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
    print("=" * 58 + "\n")

    # 返回计算出的胜率，以便在其他地方使用
    return win_rate


class WinRateReport(Callback):
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
        print(
            f"[WinRateCurriculum] Callback initialized with threshold {self.win_rate_threshold}."
        )

    def on_setup(self):
        """
        在实验设置之初被调用一次。
        这是初始化和保存原始状态的最佳位置。
        """
        # 通过 self.experiment 可以访问到 Experiment 实例本身
        # 我们复制一份原始的 train_group_map，这是控制训练哪些组的关键
        self.original_group_map = self.experiment.train_group_map.copy()
        print(
            f"[WinRateCurriculum] Setup complete. Original training groups: {list(self.original_group_map.keys())}"
        )

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
            reason_codes_tensor = batch.get(
                ("next", "attacker", "info", "termination_reason")
            )[..., 0, :]
            reason_codes = reason_codes_tensor.squeeze(-1)
            dones_mask = done_info.squeeze(-1).bool()
            terminated_codes_in_batch = reason_codes[
                dones_mask
            ]  # 得到一个一维张量，长度不定
            win_rate = log_and_calculate_win_rate(
                terminated_codes_in_batch, {1, 2, 3, 4, 5}
            )
            # total_shots_in_batch = win_info.sum().item()
            # # 'done' 标志着一个回合的结束
            total_dones_in_batch = done_info.sum().item()

            print(f"Win rate: {win_rate:.2f}")

            if (
                self.experiment.n_iters_performed < 20
            ):  # or self.experiment.n_iters_performed % 50 < 3:
                self.experiment.train_group_map = new_train_map
                return

            # 2. 根据比率决定本次迭代要训练哪些组
            if win_rate < self.win_rate_threshold and total_dones_in_batch > 0:
                # 胜率低，只训练进攻方。我们从训练地图中移除防守方。
                if "defender" in new_train_map:
                    del new_train_map["defender"]
                print(
                    f"\n[WinRateCurriculum] Win rate ({win_rate:.2f}) is LOW. Training groups: {list(new_train_map.keys())}"
                )
            elif win_rate > 1 - self.win_rate_threshold and total_dones_in_batch > 0:
                # 胜率高，只训练防守方。我们从训练地图中移除进攻方。
                if "attacker" in new_train_map:
                    del new_train_map["attacker"]
                print(
                    f"\n[WinRateCurriculum] Win rate ({win_rate:.2f}) is HIGH. Training groups: {list(new_train_map.keys())}"
                )
            else:
                # 胜率达标，训练所有组 (new_train_map 已经是所有组了)
                if total_dones_in_batch > 0:
                    print(
                        f"\n[WinRateCurriculum] Win rate ({win_rate:.2f}) is GOOD. Training all groups: {list(new_train_map.keys())}"
                    )

        except (KeyError, AttributeError) as e:
            # 如果在 batch 中找不到所需信息 (例如，在第一次迭代时)，则默认训练所有组
            print(
                f"\n[WinRateCurriculum] Could not compute win rate ({e}). Defaulting to train all groups."
            )
            pass

        # 3. 【核心】更新 Experiment 的 train_group_map
        # 这是我们与 Experiment 交互的“暴露接口”。
        # 下一个训练循环将只会遍历我们在这里设置的组。
        self.experiment.train_group_map = new_train_map



class WinRateReportSimple(Callback):
    """
    [P0 单组化] 只统计并打印进攻方胜率与终局分布，不再切换训练组（课程逻辑已移除）。

    单组 "agents" (A1/A2/D1/D2) 全部一起训练，胜率仅作为监控指标。
    """

    def __init__(self):
        print("[WinRateReport] Callback initialized (single-group, no curriculum).")

    def on_setup(self):
        print(
            f"[WinRateReport] Setup complete. Training groups: {list(self.experiment.train_group_map.keys())}"
        )

    def on_batch_collected(self, batch: TensorDictBase):
        try:
            done_info = batch.get(("next", "done"))
            # 单组下 termination_reason 形状 [T, B, n_agents, 1]，index 0 = A1（进攻方）
            reason_codes_tensor = batch.get(
                ("next", "agents", "info", "termination_reason")
            )[..., 0, :]
            reason_codes = reason_codes_tensor.squeeze(-1)
            dones_mask = done_info.squeeze(-1).bool()
            terminated_codes_in_batch = reason_codes[dones_mask]

            win_rate = log_and_calculate_win_rate(
                terminated_codes_in_batch, {1, 2, 3, 4, 5}
            )
            print(
                f"Win rate: {win_rate:.2f} | total dones: {int(dones_mask.sum().item())}"
            )
        except (KeyError, AttributeError) as e:
            print(f"\n[WinRateReport] Could not compute win rate ({e}).")

from health_check import HealthCheckCallback
from reward_histogram_callback import (
    RewardHistogramCallback,
    RewardDistributionCallback,
)


class GradConflictCallback(Callback):
    """[P0-D] 共享主干梯度冲突监控（每 N 轮打印一次，绝不打断训练）。

    对 buffer 中同一个 minibatch 分别只用攻方(A1/A2)/守方(D1/D2)的 advantage
    反传 PPO 目标（loss_objective + loss_entropy），得到共享主干上的两组梯度，
    打印其余弦相似度与 per-role 头权重范数，作为"共享主干是否正在被攻守撕扯"的判据。
    """

    def __init__(self, log_interval: int = 10, probe_batch: int = 1024, group: str = "agents"):
        self.log_interval = int(log_interval)
        self.probe_batch = int(probe_batch)
        self.group = group
        self._calls = 0

    @staticmethod
    def _flatten_grads(params):
        grads = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
        if not grads:
            return None
        return torch.cat(grads)

    def on_batch_collected(self, batch: TensorDictBase):
        self._calls += 1
        if self.log_interval <= 0 or self._calls % self.log_interval != 0:
            return
        try:
            exp = self.experiment
            loss = exp.losses[self.group]
            buf = exp.replay_buffers[self.group]
            actor = loss.actor_network
            device = exp.config.train_device

            td = buf.sample()
            if td.shape[0] > self.probe_batch:
                td = td[: self.probe_batch]
            td = td.to(device)

            # 共享主干参数（排除角色专属模块：角色嵌入 / FiLM / 各角色输出头）
            role_specific = ("role_embedding", "role_proj", "role_film", "heads")
            trunk = [
                p
                for name, p in actor.named_parameters()
                if not any(k in name for k in role_specific)
            ]

            adv_key = loss.tensor_keys.advantage
            masks = {"A": [1.0, 1.0, 0.0, 0.0], "D": [0.0, 0.0, 1.0, 1.0]}
            grads = {}
            for tag, m in masks.items():
                td_masked = td.clone()
                adv = td_masked.get(adv_key)
                shape = [1] * adv.dim()
                shape[-2] = len(m)
                mask = torch.as_tensor(m, device=adv.device, dtype=adv.dtype).view(shape)
                td_masked.set(adv_key, adv * mask)
                for p in trunk:
                    p.grad = None
                loss_vals = loss(td_masked)
                total = loss_vals.get("loss_objective")
                ent = loss_vals.get("loss_entropy", None)
                if ent is not None:
                    total = total + ent
                total.backward()
                grads[tag] = self._flatten_grads(trunk)
                del td_masked, loss_vals

            # 清理残留梯度，避免污染正式训练
            for p in actor.parameters():
                p.grad = None

            ga, gd = grads.get("A"), grads.get("D")
            if ga is not None and gd is not None:
                cos = torch.nn.functional.cosine_similarity(
                    ga.unsqueeze(0), gd.unsqueeze(0)
                ).item()
                print(
                    f"[GradConflict] calls={self._calls} | cos(攻,守)={cos:+.3f} "
                    f"| ||g_A||={ga.norm():.2f} ||g_D||={gd.norm():.2f}"
                )
            else:
                print(f"[GradConflict] calls={self._calls} | 无有效梯度")

            head_norms = []
            for mod in actor.modules():
                if mod.__class__.__name__ == "RoleConditionedMLP":
                    for rid, head in enumerate(mod.heads):
                        n = sum(float(p.detach().pow(2).sum()) for p in head.parameters()) ** 0.5
                        head_norms.append((rid, n))
            if head_norms:
                txt = " ".join(f"r{rid}={n:.2f}" for rid, n in head_norms)
                print(f"[GradConflict] per-role 头范数: {txt}")
        except Exception as e:  # 诊断逻辑绝不打断训练
            print(f"[GradConflict] skipped ({type(e).__name__}: {e})")


# checkpoint_path = "outputs/2025-07-06_19-39-05/mappo_layup_gru__c217740f_25_07_06-19_39_05/checkpoints"
checkpoint_pattern = "outputs/**/checkpoints/*.pt"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RoboCon 2025 MARL Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
训练模式:
  cold        从零开始 (默认)
  cont        继续训练所有组 (actor + critic + optimizer + buffer)
  atk-a       只加载attacker actor
  def-a       只加载defender actor
  both-a      加载双方actor
  atk-c       保留attacker critic (旧命名，建议使用atk-ac)
  def-c       保留defender critic (旧命名，建议使用def-ac)
  atk-ac      加载attacker的actor + critic
  def-ac      加载defender的actor + critic
  both-ac     加载双方的actor + critic

示例:
  python clear_restore.py -m cold
  python clear_restore.py -m cont
  python clear_restore.py -m atk-a -c outputs/xxx/checkpoint.pt
  python clear_restore.py -m both-ac
        """,
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="cold",
        choices=[
            "cold",
            "cont",
            "atk-a",
            "def-a",
            "both-a",
            "atk-c",
            "def-c",
            "atk-ac",
            "def-ac",
            "both-ac",
        ],
        help="训练模式",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint路径 (不指定则自动找最新)",
    )

    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="outputs/**/checkpoints/*.pt",
        help="checkpoint搜索模式",
    )

    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="覆盖实验的 max_n_iters (用于短程冒烟/监控, 不指定则使用 yaml 中的 4000)",
    )

    # 模型调试日志参数
    parser.add_argument("--debug-log", action="store_true", help="启用模型调试日志")

    parser.add_argument(
        "--debug-console", action="store_true", help="输出调试日志到控制台 (默认: True)"
    )

    parser.add_argument(
        "--no-debug-console",
        dest="debug_console",
        action="store_false",
        default=True,
        help="禁用控制台调试日志输出",
    )

    parser.add_argument("--debug-file", action="store_true", help="输出调试日志到文件")

    parser.add_argument(
        "--debug-file-path",
        type=str,
        default=None,
        help="调试日志文件路径 (默认: outputs/{timestamp}/model_debug.log)",
    )

    parser.add_argument(
        "--debug-console-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="控制台日志级别 (默认: DEBUG)",
    )

    parser.add_argument(
        "--debug-file-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="文件日志级别 (默认: DEBUG)",
    )

    return parser.parse_args()


def load_checkpoint_for_mode(experiment_config, mode, checkpoint_path, pattern):
    """根据模式设置checkpoint配置，返回checkpoint对象（用于部分加载）"""
    if mode == "cold":
        print("\n[COLD START] Starting fresh training.\n")
        # Cold start: 设置宽松的初始 shot threshold
        os.environ["VMAS_INITIAL_SHOT_THRESHOLD"] = "1.2"
        print("[ENV] Set VMAS_INITIAL_SHOT_THRESHOLD = 1.2 (lenient for cold start)")
        return None

    # 确定checkpoint路径
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(pattern)
        if checkpoint_path is None:
            print(f"[WARNING] No checkpoint found. Starting from scratch.\n")
            return None

    print(f"\n[LOADING] Checkpoint: {checkpoint_path}")

    if mode == "cont":
        # [P0 单组化] 兼容性校验：旧 checkpoint 属 attacker/defender 分组架构，无法续训新模型
        try:
            _ck_keys = list(
                torch.load(
                    checkpoint_path, map_location="cpu", mmap=True, weights_only=False
                ).keys()
            )
        except Exception as _e:
            _ck_keys = []
            print(f"[P0] checkpoint 预检失败（忽略）: {_e}")
        if _ck_keys and "loss_agents" not in _ck_keys:
            raise SystemExit(
                "[P0 单组化] 该 checkpoint 属于旧的 attacker/defender 分组架构"
                "（无 'loss_agents'），与单组共享主干模型不兼容。\n"
                "           请用 cold 模式从零开始训练。"
            )

        # 使用 experiment_config.restore_file 恢复完整状态
        print("[CONTINUE] Setting restore_file for full state recovery...")
        experiment_config.restore_file = checkpoint_path
        # Continue training: 使用目标难度
        os.environ["VMAS_INITIAL_SHOT_THRESHOLD"] = "0.2"
        print(
            "[ENV] Set VMAS_INITIAL_SHOT_THRESHOLD = 0.2 (target difficulty for continue)"
        )
        print("  ✓ Will restore: actor + critic + optimizer + buffer")
        print("[DONE] Checkpoint configured!\n")
        return None
    else:
        # 其他模式需要手动加载部分权重（部分加载也算继续训练，使用目标难度）
        os.environ["VMAS_INITIAL_SHOT_THRESHOLD"] = "0.2"
        print(
            "[ENV] Set VMAS_INITIAL_SHOT_THRESHOLD = 0.2 (target difficulty for partial load)"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print_dict_paths(checkpoint)
        return checkpoint


def load_actor_only(experiment, checkpoint, group):
    """只加载指定group的actor权重"""
    ACTOR_PREFIX = "actor_network_params."
    loss_key = f"loss_{group}"

    if loss_key not in checkpoint:
        print(f"  ✗ Warning: No weights found for group '{group}'")
        return

    full_group_state_dict = checkpoint[loss_key]
    actor_state_dict = {
        key.removeprefix(ACTOR_PREFIX): value
        for key, value in full_group_state_dict.items()
        if key.startswith(ACTOR_PREFIX)
    }

    try:
        actor_network = experiment.losses[group].actor_network_params
        actor_network.load_state_dict(actor_state_dict, strict=False)
        print(f"  ✓ Loaded {group} actor (critic remains fresh)")
    except Exception as e:
        print(f"  ✗ Failed to load {group} actor: {e}")


def apply_partial_checkpoint(experiment, checkpoint, mode):
    """在实验创建后应用部分checkpoint加载"""
    if checkpoint is None:
        return

    # [P0 单组化] 旧分组架构的 checkpoint 与新单组网络不兼容，直接跳过并提示
    if "loss_agents" not in checkpoint:
        print(
            "[P0 单组化] checkpoint 不含 'loss_agents'（旧 attacker/defender 分组架构），"
            "与单组共享主干模型不兼容 -> 跳过部分加载。请用 cold 模式从零训练。"
        )
        return

    if mode == "atk-c" or mode == "atk-ac":
        print(f"[{mode.upper()}] Attacker full + Defender fresh...")
        if "loss_attacker" in checkpoint:
            experiment.losses["attacker"].load_state_dict(checkpoint["loss_attacker"])
            print("  ✓ Attacker (actor + critic)")
        print("  ✓ Defender fresh")

    elif mode == "def-c" or mode == "def-ac":
        print(f"[{mode.upper()}] Defender full + Attacker fresh...")
        if "loss_defender" in checkpoint:
            experiment.losses["defender"].load_state_dict(checkpoint["loss_defender"])
            print("  ✓ Defender (actor + critic)")
        print("  ✓ Attacker fresh")

    elif mode == "atk-a":
        print("[ATK-A] Loading attacker actor only...")
        load_actor_only(experiment, checkpoint, "attacker")

    elif mode == "def-a":
        print("[DEF-A] Loading defender actor only...")
        load_actor_only(experiment, checkpoint, "defender")

    elif mode == "both-a":
        print("[BOTH-A] Loading both actors...")
        load_actor_only(experiment, checkpoint, "attacker")
        load_actor_only(experiment, checkpoint, "defender")

    elif mode == "both-ac":
        print("[BOTH-AC] Loading both groups (actor + critic)...")
        if "loss_attacker" in checkpoint:
            experiment.losses["attacker"].load_state_dict(checkpoint["loss_attacker"])
            print("  ✓ Attacker (actor + critic)")
        if "loss_defender" in checkpoint:
            experiment.losses["defender"].load_state_dict(checkpoint["loss_defender"])
            print("  ✓ Defender (actor + critic)")

    print("[DONE] Checkpoint loaded!\n")


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    print("=" * 60)
    print(f"Training Mode: {args.mode}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print("=" * 60 + "\n")

    # 配置实验
    experiment_config = ExperimentConfig.get_from_yaml()

    # [性能 A/B] 采集设备可由环境变量覆盖（cpu/cuda），用于对比策略推断开销
    _sampling_device_env = os.environ.get("SAMPLING_DEVICE", None)
    if _sampling_device_env:
        print(
            f"[OVERWRITE] sampling_device: {experiment_config.sampling_device} -> {_sampling_device_env}"
        )
        experiment_config.sampling_device = _sampling_device_env

    # [性能] CPU 采集 bf16 加速：SAMPLING_AUTOCAST_BF16=1 时，策略前向在 CPU 上用 bf16
    # autocast 计算（模型内部 bf16、输出仍转回 fp32），评测进程保持 fp32。
    if os.environ.get("SAMPLING_AUTOCAST_BF16", "0") == "1":
        print(
            "[PERF] SAMPLING_AUTOCAST_BF16=1 -> CPU 采集策略前向使用 bf16 autocast（评测保持 fp32）"
        )

    # 可选: 覆盖最大迭代次数 (短程冒烟)
    if args.max_iters is not None:
        print(
            f"[OVERWRITE] max_n_iters: {experiment_config.max_n_iters} -> {args.max_iters}"
        )
        experiment_config.max_n_iters = args.max_iters

    # 根据模式配置checkpoint（cont模式会设置restore_file）
    checkpoint = load_checkpoint_for_mode(
        experiment_config, args.mode, args.checkpoint, args.pattern
    )

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S").replace(":", "-")
    folder_name = f"outputs/{current_time}"
    os.makedirs(folder_name)
    experiment_config.save_folder = folder_name

    # ========== 配置模型调试日志 ==========
    if args.debug_log:
        # 如果没有指定文件路径，使用默认路径（与实验输出同目录）
        debug_file_path = args.debug_file_path
        if args.debug_file and debug_file_path is None:
            debug_file_path = f"{folder_name}/model_debug.log"

        setup_model_logging(
            log_to_console=args.debug_console,
            log_to_file=args.debug_file,
            log_file_path=debug_file_path,
            console_level=args.debug_console_level,
            file_level=args.debug_file_level,
        )
    else:
        # 禁用所有调试日志
        setup_model_logging(log_to_console=False, log_to_file=False)
    # ======================================

    # 使用您重构后的新环境
    new_task = LayupTask.LAYUP.get_from_yaml()

    # [P0 单组化] 4 个 agent (A1/A2/D1/D2) 一个组 "agents"，共享一套主干网络：
    #   - actor: 共享 Attention+GRU 主干 + 角色条件化输出头（role_ids=[0,1,2,2] → 3 个角色头）
    #   - critic: 单网络一次前向输出 4 个价值 (share_param_critic=False => 输出 (n_agents,1))
    agent_algorithm_config = MappoConfig.get_from_yaml()
    agent_algorithm_config.share_param_actor = True
    agent_algorithm_config.share_param_critic = False
    algorithm_config = agent_algorithm_config
    print(f"agents algo: {agent_algorithm_config}")

    agent_model_config = SequenceModelConfig(
        model_configs=[
            AttentionConfig.get_from_yaml(
                "benchmarl/conf/model/layers/attention_agents.yaml"
            ),
            GruConfig.get_from_yaml(),
        ],
        intermediate_sizes=[256],
    )
    model_config = agent_model_config
    critic_model_config = AttentionConfig.get_from_yaml(
        "benchmarl/conf/model/layers/attention_critic.yaml"
    )

    # --- [已停用] MLP + CatFrames 冒烟配置, 保留备查 ---
    # attacker_model_config = MlpConfig.get_from_yaml("benchmarl/conf/model/layers/mlp.yaml")
    # defender_model_config = MlpConfig.get_from_yaml("benchmarl/conf/model/layers/mlp.yaml")
    # model_config = EnsembleModelConfig(
    #     {"attacker": attacker_model_config, "defender": defender_model_config}
    # )
    # critic_model_config = MlpConfig.get_from_yaml("benchmarl/conf/model/layers/mlp.yaml")

    # mamba
    # model_config = MambaConfig.get_from_yaml()

    # basic: mlp
    # model_config = MlpConfig.get_from_yaml()
    # critic_model_config = model_config

    # [LiveView] 观察窗回调（LIVE_VIEW=1 时启用）：每迭代从 replay buffer 取一条环境序列写 JSON
    _callbacks = [
        WinRateReportSimple(),
        RewardHistogramCallback(log_interval=5),
        RewardDistributionCallback(log_interval=5),
        GradConflictCallback(log_interval=10),
    ]
    if os.environ.get("LIVE_VIEW", "0") == "1":
        try:
            import sys as _sys
            _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # 观察窗组件（live_view.py / live_view.html / 面板服务）在 BenchMARL/liveview/
            _sys.path.insert(0, os.path.join(_repo_root, "BenchMARL", "liveview"))
            from live_view import LiveViewCallback
            _live_out = os.environ.get("LIVE_OUT", "outputs/live")
            _callbacks.append(LiveViewCallback(out_dir=_live_out))
            print(f"[LiveView] 已挂载观察窗回调，输出 {_live_out}/live_env.json")
        except Exception as _e:
            print(f"[LiveView] 挂载失败，跳过: {type(_e).__name__}: {_e}")

    # 创建一个全新的实验对象，所有状态都是初始化的
    experiment = Experiment(
        task=new_task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=114514,
        config=experiment_config,
        callbacks=_callbacks,
    )
    print("New experiment created.\n")

    # [PERF] 打印实际生效的 PPO 系数（防"参数名写错被 torchrl 静默丢弃"再次发生）
    for _g, _loss in experiment.losses.items():
        _line = [f"[LossCoef] {_g}:"]
        for _k in ("entropy_coef", "critic_coef"):
            _v = getattr(_loss, _k, None)
            try:
                _show = f"{float(_v):.4f}"
            except (TypeError, ValueError):
                _show = str(_v)
            _want = getattr(agent_algorithm_config, _k, "?")
            _line.append(f"{_k}={_show} (配置 {_want})")
        print(" | ".join(_line))

    # 对于非cont模式，手动应用部分checkpoint加载
    apply_partial_checkpoint(experiment, checkpoint, args.mode)

    # 5. 开始在新环境上训练
    print("\nStarting training on the new environment...")

    experiment.run()
