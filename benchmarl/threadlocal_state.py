"""线程安全的全局状态补丁（修复"采集线程与训练线程竞争进程级全局态"的 bug）。

背景
----
tensordict / torchrl 把若干"上下文状态"存在**模块级单例**里（`tensordict.utils._ContextManager`，
内部只有一个 `self._mode` + 一把锁），它们是**进程全局**的：

  * `tensordict.nn.probabilistic._interaction_type`      —— 采样模式（RANDOM / DETERMINISTIC / MODE / MEAN）
  * `tensordict.nn.utils._composite_lp_aggregate`        —— 复合分布 log-prob 聚合开关
  * `tensordict.nn.utils._skip_existing`                 —— GAE/价值估计时的 skip_existing
  * `torchrl.modules.tensordict_module.rnn.recurrent_mode_state_manager` —— RNN recurrent mode

而**训练路径会写这些全局态**：
  * `torchrl/objectives/common.py:49-54` 的 `_forward_wrapper`（所有 LossModule 的 forward/xxx_loss 都会被包）
    在每次进入 loss 时执行 `set_exploration_type(self.deterministic_sampling_mode)`
    （默认 = `ExplorationType.DETERMINISTIC`，见 common.py:132）与 `set_recurrent_mode(True)`；
  * `torchrl/objectives/ppo.py:517/600/1340` 用 `set_composite_lp_aggregate(False)`；
  * `torchrl/objectives/value/advantages.py:77/92` 用 `set_skip_existing(...)` / `set_composite_lp_aggregate(True)`。

采集线程里 `tensordict/nn/probabilistic.py:604` 每次采样都会读 `interaction_type()`（全局），
所以当"采集线程在跑 rollout、训练线程在跑 loss"并发时（OVERLAP_COLLECTION=1），
**采集侧会读到训练侧设置的 DETERMINISTIC / 聚合开关**，导致采到的动作与 log_prob 被污染
（实测：数据里回合变短、A1 不推进、动作变小；串行模式无此问题）。

修法
----
把这些单例替换成**线程局部**版本（每个线程有自己的模式，未设置时回落到默认值）。
单线程（串行采集/评测）行为与原来完全一致；多线程（重叠采集）互不干扰。
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import torch


class ThreadLocalContextManager:
    """`tensordict.utils._ContextManager` 的线程局部等价物。

    语义：`get_mode()` 返回"当前线程"设置的模式；当前线程没设置过则返回构造时的 default。
    """

    def __init__(self, default=None):
        self._default = default
        self._local = threading.local()

    def get_mode(self):
        return getattr(self._local, "mode", self._default)

    def set_mode(self, mode) -> None:
        self._local.mode = mode


def _true_default(expr_value, fallback):
    return fallback if expr_value is None else expr_value


def install_thread_local_state_managers(verbose: bool = True) -> list[str]:
    """把 tensordict/torchrl 的进程级上下文管理器替换为线程局部版本。返回被替换的名字列表。"""
    patched: list[str] = []
    try:
        import tensordict.nn.probabilistic as _p
        import tensordict.nn.utils as _u
        import torchrl.modules.tensordict_module.rnn as _r
    except Exception as exc:  # pragma: no cover - 环境缺依赖时静默跳过
        if verbose:
            print(f"[TLState] 跳过线程局部补丁（导入失败：{exc}）")
        return patched

    # 注意：_composite_lp_aggregate 不替换！
    # 它有两个特点：① 由装饰器在 import 期就永久置位（不同模块互相覆盖）；②
    # ProbabilisticTensorDictModule 会把"实例化时的值"和"当前值"做一致性检查
    # （不一致时 log_prob_key 直接抛 RuntimeError）。改成线程局部会让采集线程读到
    # 与实例化时不同的值而报错；而它被 loss 改动的窗口只会把它置为 False，
    # 与当前环境值一致，渗漏无害，故保持原样。
    targets = [
        (_p, "_interaction_type", None),
        (_u, "_skip_existing", _u._skip_existing.get_mode()),
        (_r, "recurrent_mode_state_manager", _r.recurrent_mode_state_manager.get_mode()),
    ]
    for mod, name, default in targets:
        cur = getattr(mod, name, None)
        if isinstance(cur, ThreadLocalContextManager):
            continue
        if cur is None:
            continue
        setattr(mod, name, ThreadLocalContextManager(default))
        patched.append(f"{mod.__name__}.{name}")

    if verbose:
        print(f"[TLState] 线程局部全局态补丁已安装：{', '.join(patched) if patched else '(无)'}")
    return patched


def maybe_install_from_env(verbose: bool = True) -> list[str]:
    """按环境变量 TL_STATE（默认 "1"=开启）安装补丁。"""
    if os.environ.get("TL_STATE", "1") == "0":
        if verbose:
            print("[TLState] 已通过 TL_STATE=0 关闭线程局部补丁")
        return []
    return install_thread_local_state_managers(verbose=verbose)
