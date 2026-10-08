#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import copy
import gc
import importlib

import os
import tempfile
import pickle
import shutil
import time
import warnings
from collections import deque, OrderedDict
from dataclasses import dataclass, MISSING, field
from pathlib import Path

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector

from torchrl.envs import ParallelEnv, SerialEnv, TransformedEnv
from torchrl.envs.transforms import Compose
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.record.loggers import generate_exp_name
from tqdm import tqdm

# --- 线程安全的全局态补丁 ---------------------------------------------------
# 把 tensordict/torchrl 的进程级上下文管理器（_interaction_type / _composite_lp_aggregate /
# _skip_existing / recurrent_mode_state_manager）换成线程局部版本。
# 否则重叠采集（OVERLAP_COLLECTION=1）时，训练线程在 loss 里设置的 DETERMINISTIC
# 会被采集线程的 rollout 读到，污染采样的动作与 log-prob。
from benchmarl.threadlocal_state import maybe_install_from_env as _maybe_install_tl_state

_maybe_install_tl_state()
# ---------------------------------------------------------------------------

from benchmarl.algorithms.ensemble import EnsembleAlgorithm
from benchmarl.algorithms import IppoConfig, MappoConfig

from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Task, TaskClass
from benchmarl.experiment.callback import Callback, CallbackNotifier
from benchmarl.experiment.logger import Logger
from benchmarl.models import GnnConfig, SequenceModelConfig
from benchmarl.models.common import ModelConfig
from benchmarl.utils import (
    _add_rnn_transforms,
    _read_yaml_config,
    local_seed,
    seed_everything,
)
import multiprocessing
import queue
from torch.profiler import profile, record_function, ProfilerActivity
from torch.amp import autocast, GradScaler
from tensordict.nn import set_composite_lp_aggregate

_has_hydra = importlib.util.find_spec("hydra") is not None
if _has_hydra:
    from hydra.core.hydra_config import HydraConfig


class _CollectionPrefetcher:
    """[PERF] 后台线程预取采集批次，使 CPU 采集与 GPU 训练重叠执行。

    安全性依据（已在容器内实测）：
    - 采样设备 != 训练设备时，torchrl 会给 collector 建一个独立的目标设备策略副本
      （collector.policy 是 CPU 副本，与训练侧 GPU 模块不共享张量），并配置
      `VanillaWeightUpdater.weight_getter` 去取训练侧原始权重；因此
      `collector.update_policy_weights_()` 会把训练侧权重拷贝进该 CPU 副本。
    - 只要"同步权重"发生在后台采集结束之后、下一批采集开始之前，训练线程更新
      GPU 权重与采集线程读取本地 CPU 副本之间就没有竞争。
    语义：预取的下一批使用与当前批相同的权重（1 次迭代的滞后，PPO 可接受）。
    """

    def __init__(self, iterator, collector):
        import threading

        self._iterator = iterator
        self._collector = collector
        self._q = queue.Queue(maxsize=1)
        self._go = threading.Event()
        self._stop = threading.Event()
        self._primed = False
        self._thread = threading.Thread(
            target=self._worker, name="benchmarl-prefetch-collect", daemon=True
        )
        self._thread.start()

    def _worker(self):
        try:
            while not self._stop.is_set():
                if not self._go.wait(timeout=0.5):
                    continue
                self._go.clear()
                if self._stop.is_set():
                    break
                t0 = time.perf_counter()
                batch = next(self._iterator)
                if os.environ.get("OV_CLONE", "0") == "1":
                    # [DIAG] 深拷贝：避免主线程消费该批次时，采集线程已在写入下一批（存储复用/别名）
                    batch = batch.clone()
                self._q.put((batch, time.perf_counter() - t0))
        except BaseException as e:  # 采集异常转发给主线程重新抛出
            try:
                self._q.put(e)
            except Exception:
                pass

    def get(self):
        """阻塞获取预取批次；首次调用会先同步权重并启动后台采集。"""
        if not self._primed:
            self._collector.update_policy_weights_()
            self._go.set()
            self._primed = True
        item = self._q.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def trigger_next(self):
        """在"上一批已取回、下一批未开始"的窗口内同步权重并触发下一批预取。"""
        self._collector.update_policy_weights_()
        self._go.set()

    def close(self):
        self._stop.set()
        self._go.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)


@dataclass
class ExperimentConfig:
    """
    Configuration class for experiments.
    This class acts as a schema for loading and validating yaml configurations.

    Parameters in this class aim to be agnostic of the algorithm, task or model used.
    To know their meaning, please check out the descriptions in ``benchmarl/conf/experiment/base_experiment.yaml``
    """

    sampling_device: str = MISSING
    train_device: str = MISSING
    buffer_device: str = MISSING

    share_policy_params: bool = MISSING
    prefer_continuous_actions: bool = MISSING
    collect_with_grad: bool = MISSING
    parallel_collection: bool = MISSING

    gamma: float = MISSING
    lr: float = MISSING
    adam_eps: float = MISSING
    clip_grad_norm: bool = MISSING
    clip_grad_val: Optional[float] = MISSING

    soft_target_update: bool = MISSING
    polyak_tau: float = MISSING
    hard_target_update_frequency: int = MISSING

    exploration_eps_init: float = MISSING
    exploration_eps_end: float = MISSING
    exploration_anneal_frames: Optional[int] = MISSING

    max_n_iters: Optional[int] = MISSING
    max_n_frames: Optional[int] = MISSING

    n_workers: int = MISSING

    on_policy_collected_frames_per_batch: int = MISSING
    on_policy_n_envs_per_worker: int = MISSING
    on_policy_n_minibatch_iters: int = MISSING
    on_policy_minibatch_size: int = MISSING

    off_policy_collected_frames_per_batch: int = MISSING
    off_policy_n_envs_per_worker: int = MISSING
    off_policy_n_optimizer_steps: int = MISSING
    off_policy_train_batch_size: int = MISSING
    off_policy_memory_size: int = MISSING
    off_policy_init_random_frames: int = MISSING
    off_policy_use_prioritized_replay_buffer: bool = MISSING
    off_policy_prb_alpha: float = MISSING
    off_policy_prb_beta: float = MISSING

    evaluation: bool = MISSING
    render: bool = MISSING
    evaluation_interval: int = MISSING
    evaluation_episodes: int = MISSING
    evaluation_deterministic_actions: bool = MISSING
    evaluation_static: bool = MISSING

    loggers: List[str] = MISSING
    project_name: str = MISSING
    wandb_extra_kwargs: Dict[str, Any] = MISSING
    create_json: bool = MISSING

    save_folder: Optional[str] = MISSING
    restore_file: Optional[str] = MISSING
    restore_map_location: Optional[Any] = MISSING
    checkpoint_interval: int = MISSING
    checkpoint_at_end: bool = MISSING
    keep_checkpoints_num: Optional[int] = MISSING

    # Mixed precision training
    use_amp: bool = MISSING
    amp_dtype: str = MISSING  # "float16" or "bfloat16"

    evaluation_device: str = "cpu"

    # Learning rate scheduler
    lr_scheduler: str = "constant"  # Options: constant, linear, cosine, exponential
    lr_scheduler_min_lr: float = 0.0  # Minimum learning rate for linear/cosine
    lr_scheduler_gamma: float = 0.99  # Decay factor for exponential
    lr_scheduler_T_max: Optional[int] = (
        None  # Number of iterations for decay (None = use max_n_iters)
    )

    def train_batch_size(self, on_policy: bool) -> int:
        """
        The batch size of tensors used for training

        Args:
            on_policy (bool): is the algorithms on_policy

        """
        return (
            self.collected_frames_per_batch(on_policy)
            if on_policy
            else self.off_policy_train_batch_size
        )

    def train_minibatch_size(self, on_policy: bool) -> int:
        """
        The minibatch size of tensors used for training.
        On-policy algorithms are trained by splitting the train_batch_size (equal to the collected frames) into minibatches.
        Off-policy algorithms do not go through this process and thus have the ``train_minibatch_size==train_batch_size``

        Args:
            on_policy (bool): is the algorithms on_policy
        """
        return (
            self.on_policy_minibatch_size
            if on_policy
            else self.train_batch_size(on_policy)
        )

    def n_optimizer_steps(self, on_policy: bool) -> int:
        """
        Number of times to loop over the training step per collection iteration.

        Args:
            on_policy (bool): is the algorithms on_policy

        """
        return (
            self.on_policy_n_minibatch_iters
            if on_policy
            else self.off_policy_n_optimizer_steps
        )

    def replay_buffer_memory_size(self, on_policy: bool) -> int:
        """
        Size of the replay buffer memory in terms of frames

        Args:
            on_policy (bool): is the algorithms on_policy

        """
        return (
            self.collected_frames_per_batch(on_policy)
            if on_policy
            else self.off_policy_memory_size
        )

    def collected_frames_per_batch(self, on_policy: bool) -> int:
        """
        Number of collected frames per collection iteration.

         Args:
             on_policy (bool): is the algorithms on_policy

        """
        return (
            self.on_policy_collected_frames_per_batch
            if on_policy
            else self.off_policy_collected_frames_per_batch
        )

    def n_envs_per_worker(self, on_policy: bool) -> int:
        """
        Number of environments used for collection

        - In vectorized environments, this will be the vectorized batch_size.
        - In other environments, this will be emulated by running them sequentially.

        Args:
            on_policy (bool): is the algorithms on_policy


        """
        return (
            self.on_policy_n_envs_per_worker
            if on_policy
            else self.off_policy_n_envs_per_worker
        )

    def get_max_n_frames(self, on_policy: bool) -> int:
        """
        Get the maximum number of frames collected before the experiment ends.

        Args:
            on_policy (bool): is the algorithms on_policy
        """
        if self.max_n_frames is not None and self.max_n_iters is not None:
            return min(
                self.max_n_frames,
                self.max_n_iters * self.collected_frames_per_batch(on_policy),
            )
        elif self.max_n_frames is not None:
            return self.max_n_frames
        elif self.max_n_iters is not None:
            return self.max_n_iters * self.collected_frames_per_batch(on_policy)

    def get_max_n_iters(self, on_policy: bool) -> int:
        """
        Get the maximum number of experiment iterations before the experiment ends.

        Args:
            on_policy (bool): is the algorithms on_policy
        """
        return -(
            -self.get_max_n_frames(on_policy)
            // self.collected_frames_per_batch(on_policy)
        )

    def get_exploration_anneal_frames(self, on_policy: bool):
        """
        Get the number of frames for exploration annealing.
        If self.exploration_anneal_frames is None this will be a third of the total frames to collect.

        Args:
            on_policy (bool): is the algorithms on_policy
        """
        return (
            (self.get_max_n_frames(on_policy) // 3)
            if self.exploration_anneal_frames is None
            else self.exploration_anneal_frames
        )

    @staticmethod
    def get_from_yaml(path: Optional[str] = None):
        """
        Load the experiment configuration from yaml

        Args:
            path (str, optional): The full path of the yaml file to load from.
                If None, it will default to
                ``benchmarl/conf/experiment/base_experiment.yaml``

        Returns:
            the loaded :class:`~benchmarl.experiment.ExperimentConfig`
        """
        if path is None:
            yaml_path = (
                Path(__file__).parent.parent
                / "conf"
                / "experiment"
                / "base_experiment.yaml"
            )
            return ExperimentConfig(**_read_yaml_config(str(yaml_path.resolve())))
        else:
            return ExperimentConfig(**_read_yaml_config(path))

    def validate(self, on_policy: bool):
        """
        Validates config.

        Args:
            on_policy (bool): is the algorithms on_policy

        """
        if (
            self.evaluation
            and self.evaluation_interval % self.collected_frames_per_batch(on_policy)
            != 0
        ):
            raise ValueError(
                f"evaluation_interval ({self.evaluation_interval}) "
                f"is not a multiple of the collected_frames_per_batch ({self.collected_frames_per_batch(on_policy)})"
            )
        if (
            self.checkpoint_interval != 0
            and self.checkpoint_interval % self.collected_frames_per_batch(on_policy)
            != 0
        ):
            raise ValueError(
                f"checkpoint_interval ({self.checkpoint_interval}) "
                f"is not a multiple of the collected_frames_per_batch ({self.collected_frames_per_batch(on_policy)})"
            )
        if self.keep_checkpoints_num is not None and self.keep_checkpoints_num <= 0:
            raise ValueError("keep_checkpoints_num must be greater than zero or null")
        if self.max_n_frames is None and self.max_n_iters is None:
            raise ValueError("max_n_frames and max_n_iters are both not set")
        if self.max_n_frames is not None and self.max_n_iters is not None:
            warnings.warn(
                f"max_n_frames and max_n_iters have both been set. The experiment will terminate after "
                f"{self.get_max_n_iters(on_policy)} iterations ({self.get_max_n_frames(on_policy)} frames)."
            )


def _evaluation_worker(
    stop_event,
    weight_queue,
    experiment_config: ExperimentConfig,
    algorithm_config: AlgorithmConfig,
    model_config: ModelConfig,
    critic_model_config: ModelConfig,
    task: Task,
    group_map: Dict,
    seed: int,
    continuous_actions: bool,
    experiment_name: str,
    folder_name: Path,
):
    """
    持久化评估工作进程。
    仅初始化一次环境和算法，通过队列接收新权重进行评估。
    """
    # 强制使用 CPU
    eval_device = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # 评测保持 fp32：不启用 CPU bf16 采样加速，保证评测数值与历史记录可比
    os.environ["SAMPLING_AUTOCAST_BF16"] = "0"
    # [修复] 这个子进程不是从主进程的上下文里"继承"开关的（threading.local 在新进程里为空），
    # 而复合动作策略要求 log_prob_keys（复数）=> 必须与 _setup() 一样显式关闭聚合，
    # 否则 ProbabilisticActor 构造直接抛：
    #   RuntimeError: composite_lp_aggregate is set to True but log_prob_keys were passed.
    set_composite_lp_aggregate(False).set()
    seed_everything(seed)

    # --- 1. 初始化阶段 (仅执行一次) ---
    # 创建环境
    test_env = task.get_env_fun(
        num_envs=experiment_config.evaluation_episodes,
        continuous_actions=continuous_actions,
        seed=seed,
        device=eval_device,
    )()

    # [修复] 先应用环境 transforms, 再提取 Specs。
    # CatFrames 等 transform 会改变 observation/state 的维度; 若仍按旧顺序
    # (先取 spec 后加 transform), 评估模型会按旧维度构造, 与 env 实际输出不匹配。
    transforms_env = Compose(*task.get_env_transforms(test_env))
    test_env = TransformedEnv(test_env, transforms_env.clone()).to(eval_device)

    # 提取 Specs (此时已包含 env transforms 造成的维度变化)
    observation_spec = task.observation_spec(test_env)
    action_spec = task.action_spec(test_env)
    info_spec = task.info_spec(test_env)
    state_spec = task.state_spec(test_env)
    action_mask_spec = task.action_mask_spec(test_env)

    # 伪造一个实验壳用于初始化算法
    class ExperimentShell:
        def __init__(self):
            self.config = experiment_config
            self.algorithm_config = algorithm_config
            self.model_config = model_config
            self.critic_model_config = critic_model_config
            self.task = task
            self.group_map = group_map
            self.continuous_actions = continuous_actions
            self.seed = seed
            self.on_policy = self.algorithm_config.on_policy()
            self.observation_spec = observation_spec
            self.action_spec = action_spec
            self.info_spec = info_spec
            self.state_spec = state_spec
            self.action_mask_spec = action_mask_spec

    exp_shell = ExperimentShell()
    algorithm = algorithm_config.get_algorithm(exp_shell)
    if isinstance(algorithm, EnsembleAlgorithm):
        for aglo in algorithm.algorithms_map.values():
            aglo.device = torch.device("cpu")
    else:
        algorithm.device = torch.device("cpu")

    # RNN hidden state transforms (仅 RNN 模型需要; 不改变 obs/state 维度)
    if model_config.is_rnn:
        test_env = _add_rnn_transforms(lambda: test_env, group_map, model_config)()
    test_env = algorithm.process_env_fun(lambda: test_env)()

    # 初始化策略和 Logger
    eval_policy = algorithm.get_policy_for_collection().to(eval_device)
    logger = Logger(
        experiment_name=experiment_name,
        folder_name=str(folder_name),
        experiment_config=experiment_config,
        algorithm_name=algorithm_config.associated_class().__name__.lower(),
        model_name=model_config.associated_class().__name__.lower(),
        environment_name=task.env_name().lower(),
        task_name=task.name.lower(),
        group_map=group_map,
        seed=seed,
        project_name=experiment_config.project_name,
        wandb_extra_kwargs=experiment_config.wandb_extra_kwargs,
    )

    max_steps = task.max_steps(test_env)

    # --- 2. 循环监听阶段 ---
    # [修复] 原实现是 `while not stop_event.is_set()`: 主进程 close() 时设置 stop_event,
    # worker 会立即退出, 队列中"最后一次评估"的权重永远不会被处理
    # (表现为: 训练结束时最后一次 eval 被终止, 没有 finished 日志/视频/指标)。
    # 改为: 只有当队列为空且收到停止信号时才退出, 保证 pending 的权重先评估完。
    while True:
        try:
            # 尝试获取新权重，设置超时以便检查 stop_event
            data = weight_queue.get(timeout=1.0)
        except queue.Empty:
            if stop_event.is_set():
                break
            continue

        if data is None:
            break  # 收到结束信号

        try:
            policy_state_dict, total_frames, n_iters_performed = data

            # 更新权重
            eval_policy.load_state_dict(policy_state_dict)
            eval_policy.eval()

            # 运行评估 Rollouts
            evaluation_start = time.time()

            # 视频：流式落盘（裸帧先写临时文件，裁剪后逐帧编码 mp4，峰值内存 ≈ 1 帧）
            video_raw_path = None
            video_path = None
            video_frame_shape = None
            frame_count = 0
            raw_file = None
            if task.has_render(test_env) and experiment_config.render:
                import imageio

                # 视频目录与训练侧 logger 的 log_dir 保持一致：<folder_name>/<experiment_name>/videos
                # （torchrl 的 CSVLogger 不是 CSVExperiment 子类，isinstance 探测恒为 False）
                video_dir = Path(folder_name) / experiment_name / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                video_path = video_dir / f"eval_video_{n_iters_performed}.mp4"
                video_raw_path = (
                    Path(tempfile.gettempdir())
                    / f"eval_raw_{os.getpid()}_{n_iters_performed}.bin"
                )

            with torch.no_grad():
                with set_exploration_type(
                    ExplorationType.DETERMINISTIC
                    if experiment_config.evaluation_deterministic_actions
                    else ExplorationType.RANDOM
                ):
                    # 渲染回调处理（流式：裸帧直接写临时文件，避免整段视频驻留内存）
                    if video_raw_path is not None:
                        raw_file = open(video_raw_path, "wb")

                        def callback(env, td):
                            nonlocal frame_count, video_frame_shape
                            frame = task.__class__.render_callback(exp_shell, env, td)
                            arr = (
                                frame.detach().cpu().numpy()
                                if torch.is_tensor(frame)
                                else np.asarray(frame)
                            )
                            if arr.dtype != np.uint8:
                                arr = np.clip(arr, 0, 255).astype(np.uint8)
                            arr = np.ascontiguousarray(arr)
                            video_frame_shape = arr.shape
                            raw_file.write(arr.tobytes())
                            frame_count += 1
                            del frame, arr
                    else:
                        callback = None

                    # 执行 Rollout
                    if test_env.batch_size == ():
                        rollouts = []
                        for eval_episode in range(
                            experiment_config.evaluation_episodes
                        ):
                            rollouts.append(
                                test_env.rollout(
                                    max_steps=max_steps,
                                    policy=eval_policy,
                                    callback=callback if eval_episode == 0 else None,
                                    auto_cast_to_device=True,
                                    break_when_any_done=False,
                                )
                            )
                    else:
                        rollouts = test_env.rollout(
                            max_steps=max_steps,
                            policy=eval_policy,
                            callback=callback,
                            auto_cast_to_device=True,
                            break_when_any_done=False,
                        )
                        rollouts = list(rollouts.unbind(0))

            # 视频编码：关掉临时文件，逐帧读回写入 mp4（裁剪口径同 logger.log_evaluation）
            if video_raw_path is not None and frame_count > 0:
                if raw_file is not None:
                    raw_file.close()
                    raw_file = None
                n_keep = frame_count
                try:
                    max_length_rollout_0 = 0
                    for i, r in enumerate(rollouts):
                        next_done = logger._get_global_done(r).squeeze(-1)
                        done_index = next_done.nonzero(as_tuple=True)[0]
                        if done_index.numel() > 0:
                            r = r[: done_index[0] + 1]
                        if i == 0:
                            max_length_rollout_0 = max(
                                int(r.batch_size[0]), max_length_rollout_0
                            )
                    if max_length_rollout_0 > 1:
                        n_keep = min(n_keep, max_length_rollout_0 - 1)
                except Exception as e:  # 裁剪失败不影响视频产出
                    print(f"[Eval Worker] video trim skipped: {e}")
                frame_bytes = int(np.prod(video_frame_shape))
                written = 0
                with open(video_raw_path, "rb") as f, imageio.get_writer(
                    str(video_path), fps=30, macro_block_size=None
                ) as writer:
                    for _ in range(n_keep):
                        buf = f.read(frame_bytes)
                        if len(buf) < frame_bytes:
                            break
                        writer.append_data(
                            np.frombuffer(buf, dtype=np.uint8).reshape(video_frame_shape)
                        )
                        written += 1
                video_raw_path.unlink(missing_ok=True)
                del buf
                gc.collect()
                print(
                    f"[Eval Worker] video written (streaming): {video_path} "
                    f"({written} frames, per-frame {frame_bytes / 1048576:.2f}MB)"
                )
            elif raw_file is not None:
                raw_file.close()
                raw_file = None
            if video_raw_path is not None and video_raw_path.exists():
                video_raw_path.unlink(missing_ok=True)

            # 日志记录
            evaluation_time = time.time() - evaluation_start
            logger.log(
                {"timers/evaluation_time": evaluation_time}, step=n_iters_performed
            )
            logger.log_evaluation(
                rollouts,
                video_frames=None,
                step=n_iters_performed,
                total_frames=total_frames,
            )
            logger.commit()
            print(
                f"[Eval Worker]: Iteration {n_iters_performed} evaluation finished on CPU."
            )
        except Exception as e:
            # [诊断修复] 打印评估错误, 避免静默失败
            import traceback

            print(
                f"[Eval Worker] ERROR during evaluation: {type(e).__name__}: {e}",
                flush=True,
            )
            traceback.print_exc()
            continue

    test_env.close()
    logger.finish()


class Experiment(CallbackNotifier):
    """
    Main experiment class in BenchMARL.

    Args:
        task (TaskClass): the task
        algorithm_config (AlgorithmConfig): the algorithm configuration
        model_config (ModelConfig): the policy model configuration
        seed (int): the seed for the experiment
        config (ExperimentConfig): The experiment config. Note that some of the parameters
            of this config may go un-consumed based on the provided algorithm or model config.
            For example, all parameters off-policy algorithm would not be used when running
            an experiment with an on-policy algorithm.
        critic_model_config (ModelConfig, optional): the policy model configuration.
            If None, it defaults to model_config
        callbacks (list of Callback, optional): callbacks for this experiment
    """

    def __init__(
        self,
        task: Union[Task, TaskClass],
        algorithm_config: AlgorithmConfig,
        model_config: ModelConfig,
        seed: int,
        config: ExperimentConfig,
        critic_model_config: Optional[ModelConfig] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        multiprocessing.set_start_method("spawn", force=True)
        super().__init__(
            experiment=self, callbacks=callbacks if callbacks is not None else []
        )

        self.config = config

        if isinstance(task, Task):
            warnings.warn(
                "Call `.get_task()` or `.get_from_yaml()` on your task Enum before passing it to the experiment. "
                "If you do not do this, benchmarl will load the default task config from yaml."
            )
            task = task.get_task()
        self.task = task
        self.model_config = model_config
        self.critic_model_config = (
            critic_model_config
            if critic_model_config is not None
            else copy.deepcopy(model_config)
        )
        self.critic_model_config.is_critic = True

        self.algorithm_config = algorithm_config
        self.seed = seed

        self._setup()

        self.total_time = 0
        self.total_frames = 0
        self.n_iters_performed = 0
        self.mean_return = 0
        # [修复] 评测/存档用"累计帧增量"触发（原来的 total_frames % interval 在
        # 续训起点与新的 frames_per_batch 不对齐时会永远不触发）
        self._last_eval_frames = 0
        self._last_checkpoint_frames = 0

        self.eval_weight_queue = multiprocessing.Queue(maxsize=1)  # 保证只评估最新的
        self.eval_stop_event = multiprocessing.Event()
        self.evaluation_process = None

        if self.config.evaluation:
            self._start_evaluation_worker()

        if self.config.restore_file is not None:
            self._load_experiment()

    def _start_evaluation_worker(self):
        # 准备静态参数
        args = (
            self.eval_stop_event,
            self.eval_weight_queue,
            self.config,
            self.algorithm_config,
            self.model_config,
            self.critic_model_config,
            self.task,
            self.group_map,
            self.seed,
            self.continuous_actions,
            self.name,
            self.folder_name,
        )
        self.evaluation_process = multiprocessing.Process(
            target=_evaluation_worker,
            args=args,
            daemon=True,  # 随主进程退出
        )
        self.evaluation_process.start()

    @property
    def on_policy(self) -> bool:
        """Whether the algorithm has to be run on policy."""
        return self.algorithm_config.on_policy()

    def _setup(self):
        set_composite_lp_aggregate(False).set()
        self.config.validate(self.on_policy)
        seed_everything(self.seed)
        self._perform_checks()
        self._set_action_type()
        self._setup_name()
        self._setup_task()
        self._setup_algorithm()
        self._setup_collector()
        self._setup_logger()

        self._on_setup()

    def _perform_checks(self):
        for config in (self.model_config, self.critic_model_config):
            if isinstance(config, SequenceModelConfig):
                for layer_config in config.model_configs[1:]:
                    if isinstance(layer_config, GnnConfig) and (
                        layer_config.position_key is not None
                        or layer_config.velocity_key is not None
                    ):
                        raise ValueError(
                            "GNNs reading position or velocity keys are currently only usable in first"
                            " layer of sequence models"
                        )

        if self.algorithm_config in (MappoConfig, IppoConfig):
            critic_model_config = self.critic_model_config
            if isinstance(critic_model_config, SequenceModelConfig):
                critic_model_config = self.critic_model_config.model_configs[0]
            if (
                isinstance(critic_model_config, GnnConfig)
                and critic_model_config.topology == "from_pos"
            ):
                raise ValueError(
                    "GNNs in PPO critics with topology 'from_pos' are currently not available, "
                    "see https://github.com/pytorch/rl/issues/2537"
                )

    def _set_action_type(self):
        if (
            self.task.supports_continuous_actions()
            and self.algorithm_config.supports_continuous_actions()
            and self.config.prefer_continuous_actions
        ):
            self.continuous_actions = True
        elif (
            self.task.supports_discrete_actions()
            and self.algorithm_config.supports_discrete_actions()
        ):
            self.continuous_actions = False
        elif (
            self.task.supports_continuous_actions()
            and self.algorithm_config.supports_continuous_actions()
        ):
            self.continuous_actions = True
        else:
            raise ValueError(
                f"Algorithm {self.algorithm_config} is not compatible"
                f" with the action space of task {self.task} "
            )

    def _setup_task(self):
        test_env = self.task.get_env_fun(
            num_envs=self.config.evaluation_episodes,
            continuous_actions=self.continuous_actions,
            seed=self.seed,
            device=self.config.sampling_device,
        )()
        env_func = self.task.get_env_fun(
            num_envs=self.config.n_envs_per_worker(self.on_policy),
            continuous_actions=self.continuous_actions,
            seed=self.seed,
            device=self.config.sampling_device,
        )

        transforms_env = self.task.get_env_transforms(test_env)
        transforms_training = transforms_env + [
            self.task.get_reward_sum_transform(test_env)
        ]
        transforms_env = Compose(*transforms_env)
        transforms_training = Compose(*transforms_training)

        # Initialize test env
        self.test_env = TransformedEnv(test_env, transforms_env.clone()).to(
            self.config.sampling_device
        )

        self.observation_spec = self.task.observation_spec(self.test_env)
        self.info_spec = self.task.info_spec(self.test_env)
        self.state_spec = self.task.state_spec(self.test_env)
        self.action_mask_spec = self.task.action_mask_spec(self.test_env)
        self.action_spec = self.task.action_spec(self.test_env)
        self.group_map = self.task.group_map(self.test_env)
        self.train_group_map = copy.deepcopy(self.group_map)
        self.max_steps = self.task.max_steps(self.test_env)

        # Add rnn transforms here so they do not show in the benchmarl specs
        if self.model_config.is_rnn:
            self.test_env = _add_rnn_transforms(
                lambda: self.test_env, self.group_map, self.model_config
            )()
            env_func = _add_rnn_transforms(env_func, self.group_map, self.model_config)

        # Initialize train env
        if self.test_env.batch_size == ():
            # If the environment is not vectorized, we simulate vectorization using parallel or serial environments
            env_class = (
                SerialEnv if not self.config.parallel_collection else ParallelEnv
            )
            self.env_func = lambda: TransformedEnv(
                env_class(self.config.n_envs_per_worker(self.on_policy), env_func),
                transforms_training.clone(),
            )
        else:
            # Otherwise it is already vectorized
            self.env_func = lambda: TransformedEnv(
                env_func(), transforms_training.clone()
            )

    def _make_optimizer(self, params, group: str = "", loss_name: str = ""):
        """创建优化器。

        默认 AdamW；设置环境变量 USE_MUON=1 时改用 Muon(矩阵参数)+AdamW(其余)。
        可用环境变量调参：MUON_LR（默认 0.005）、MUON_MOMENTUM（0.95）、
        MUON_NS_STEPS（5）、MUON_WEIGHT_DECAY（0.0）、MUON_SCALE_MODE（shape|match_rms|none）。

        ACTOR_LR_MULT（默认 1.0）：只作用于 loss_objective（actor）优化器的学习率倍数。
        历史原因：去重前 actor 参数在同一组里出现两次、每步被更新两次（等效 ~2x lr），
        现在的"统一网络"是 1x —— 想复现旧的有效步长就设 ACTOR_LR_MULT=2。
        """
        params = list(params)
        # 去重：actor_network_params 持有器与网络本体可能引用同一批张量，重复入组会让同一参数
        # 每步被更新两次（PyTorch 也会告警 "optimizer contains a parameter group with duplicate parameters"）。
        seen_ids = set()
        deduped = []
        for p in params:
            if id(p) not in seen_ids:
                seen_ids.add(id(p))
                deduped.append(p)
        if len(deduped) != len(params):
            print(f"[Optimizer] {group}: 参数去重 {len(params)} -> {len(deduped)}")
        params = deduped

        lr = float(self.config.lr)
        lr_mult = 1.0
        if loss_name == "loss_objective":
            lr_mult = float(os.environ.get("ACTOR_LR_MULT", "1.0"))
            lr = lr * lr_mult
        if lr_mult != 1.0:
            print(f"[Optimizer] {group}/{loss_name}: lr {self.config.lr:.3e} x{lr_mult:g} -> {lr:.3e}")
        if os.environ.get("USE_MUON", "0") == "1":
            from benchmarl.muon import MuonWithAdamW

            muon_lr = float(os.environ.get("MUON_LR", "0.005")) * lr_mult
            opt = MuonWithAdamW(
                params,
                lr=lr,
                muon_lr=muon_lr,
                momentum=float(os.environ.get("MUON_MOMENTUM", "0.95")),
                ns_steps=int(os.environ.get("MUON_NS_STEPS", "5")),
                eps=self.config.adam_eps,
                weight_decay=1e-4,
                muon_weight_decay=float(os.environ.get("MUON_WEIGHT_DECAY", "0.0")),
                scale_mode=os.environ.get("MUON_SCALE_MODE", "shape"),
            )
            if group not in getattr(self, "_muon_banner_printed", set()):
                n_muon = sum(
                    p.numel() for g in opt.param_groups if g.get("use_muon") for p in g["params"]
                )
                n_aux = sum(
                    p.numel() for g in opt.param_groups if not g.get("use_muon") for p in g["params"]
                )
                print(
                    f"[PERF] USE_MUON=1 [{group}] -> Muon(矩阵 {n_muon/1e6:.3f}M 参数, lr={muon_lr}, "
                    f"mom={opt.param_groups[0]['momentum']}) + AdamW(其余 {n_aux/1e6:.3f}M, "
                    f"lr={self.config.lr}, eps={self.config.adam_eps}) | scale_mode="
                    f"{opt.param_groups[0].get('scale_mode', '-')}"
                )
                self._muon_banner_printed = getattr(self, "_muon_banner_printed", set()) | {group}
            return opt
        return torch.optim.AdamW(
            params,
            lr=lr,
            eps=self.config.adam_eps,
            weight_decay=1e-4,
        )

    def _setup_algorithm(self):
        self.algorithm = self.algorithm_config.get_algorithm(experiment=self)

        self.test_env = self.algorithm.process_env_fun(lambda: self.test_env)()
        self.env_func = self.algorithm.process_env_fun(self.env_func)

        self.replay_buffers = {
            group: self.algorithm.get_replay_buffer(
                group=group,
                transforms=self.task.get_replay_buffer_transforms(self.test_env, group),
            )
            for group in self.group_map.keys()
        }
        # [BufferGuard] 记录"按当前配置应有的采样 batch 大小"：从旧 checkpoint 恢复 buffer
        # 时会把 _batch_size 一起带回旧值（例如旧 run 是 minibatch 6000 -> 30 条序列），
        # 那样 minibatch 调大只会减少优化步数、每步采样量不变（等价于偷偷减 PPO 遍数）。
        self._configured_buffer_batch_size = {
            group: getattr(self.replay_buffers[group], "_batch_size", None)
            for group in self.group_map.keys()
        }
        self.losses = {
            group: self.algorithm.get_loss_and_updater(group)[0]
            for group in self.group_map.keys()
        }
        self.target_updaters = {
            group: self.algorithm.get_loss_and_updater(group)[1]
            for group in self.group_map.keys()
        }
        self.optimizers = {
            group: {
                loss_name: self._make_optimizer(params, group, loss_name)
                for loss_name, params in self.algorithm.get_parameters(group).items()
            }
            for group in self.group_map.keys()
        }

        # Initialize mixed precision GradScaler (one per group)
        self.grad_scalers = {}
        if self.config.use_amp and self.config.train_device != "cpu":
            # Determine precision type
            amp_dtype = (
                torch.float16 if self.config.amp_dtype == "float16" else torch.bfloat16
            )
            self.amp_dtype = amp_dtype

            # [修复] 只有 fp16 需要 loss scaling: fp16 指数位少, 梯度易下溢。
            # bf16 的指数位与 fp32 相同(8 bit), 不存在下溢问题,
            # GradScaler 的 scale/unscale/inf-check/update 是纯开销
            # (每 iter 约 600 次调用), 因此 bf16 下不再创建 GradScaler。
            if amp_dtype == torch.float16:
                for group in self.group_map.keys():
                    self.grad_scalers[group] = GradScaler()
            else:
                print(
                    "[AMP] bfloat16 模式: 跳过 GradScaler (bf16 无需 loss scaling)"
                )
        else:
            self.amp_dtype = torch.float32

        # Initialize learning rate schedulers
        self.lr_schedulers = {}
        if self.config.lr_scheduler != "constant":
            # Use manual T_max if specified, otherwise use max_n_iters
            max_iters = (
                self.config.lr_scheduler_T_max
                if self.config.lr_scheduler_T_max is not None
                else self.config.get_max_n_iters(self.on_policy)
            )
            for group in self.group_map.keys():
                schedulers_for_group = {}
                for loss_name, optimizer in self.optimizers[group].items():
                    if self.config.lr_scheduler == "linear":
                        # Linear decay from lr to min_lr
                        scheduler = torch.optim.lr_scheduler.LinearLR(
                            optimizer,
                            start_factor=1.0,
                            end_factor=self.config.lr_scheduler_min_lr / self.config.lr
                            if self.config.lr > 0
                            else 0.0,
                            total_iters=max_iters,
                        )
                    elif self.config.lr_scheduler == "cosine":
                        # Cosine annealing
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=max_iters,
                            eta_min=self.config.lr_scheduler_min_lr,
                        )
                    elif self.config.lr_scheduler == "exponential":
                        # Exponential decay
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(
                            optimizer,
                            gamma=self.config.lr_scheduler_gamma,
                        )
                    else:
                        scheduler = None
                    schedulers_for_group[loss_name] = scheduler
                self.lr_schedulers[group] = schedulers_for_group

    def _setup_collector(self):
        self.policy = self.algorithm.get_policy_for_collection()

        self.group_policies = {}
        for group in self.group_map.keys():
            # [修改开始] 支持复合动作键查找
            # 原代码: group_policy = self.policy.select_subsequence(out_keys=[(group, "action")])

            target_keys = [(group, "action")]
            # 检查策略是否输出了更具体的子动作键 (e.g., ("agent", "action", "continuous"))
            # select_subsequence 需要精确匹配 out_keys
            if hasattr(self.policy, "out_keys"):
                policy_out_keys = self.policy.out_keys
                # 寻找所有以此 group action 为前缀的键
                specific_keys = []
                for key in policy_out_keys:
                    # key 可能是字符串或元组，统一处理
                    key_tuple = (key,) if isinstance(key, str) else key
                    if (
                        len(key_tuple) > 2
                        and key_tuple[0] == group
                        and key_tuple[1] == "action"
                    ):
                        specific_keys.append(key)

                if specific_keys:
                    target_keys = specific_keys

            group_policy = self.policy.select_subsequence(out_keys=target_keys)

            # [修改] 移除长度断言，支持多模块序列
            # assert len(group_policy) == 1

            # [修改] 存储整个序列，而不是只取第一个元素 [0]
            # 这样后续的 exploration check (explore_layer = group_policy[-1]) 才能正确获取到最后的 Actor
            self.group_policies.update({group: group_policy})
            # [修改结束]

        if not self.config.collect_with_grad:
            if self.config.n_workers != 1:
                self.collector = MultiSyncDataCollector(
                    create_env_fn=[self.env_func for _ in range(self.config.n_workers)],
                    policy=self.policy,
                    device=self.config.sampling_device,
                    storing_device=self.config.sampling_device,
                    frames_per_batch=self.config.collected_frames_per_batch(
                        self.on_policy
                    ),
                    total_frames=self.config.get_max_n_frames(self.on_policy),
                    init_random_frames=(
                        self.config.off_policy_init_random_frames
                        if not self.on_policy
                        else -1
                    ),
                    cat_results=0,
                )
            else:
                self.collector = SyncDataCollector(
                    self.env_func,
                    self.policy,
                    device=self.config.sampling_device,
                    storing_device=self.config.sampling_device,
                    frames_per_batch=self.config.collected_frames_per_batch(
                        self.on_policy
                    ),
                    total_frames=self.config.get_max_n_frames(self.on_policy),
                    init_random_frames=(
                        self.config.off_policy_init_random_frames
                        if not self.on_policy
                        else 0
                    ),
                )
            self.collector.set_seed(self.seed)
        else:
            if self.config.off_policy_init_random_frames and not self.on_policy:
                raise TypeError(
                    "Collection via rollouts does not support initial random frames as of now."
                )
            self.rollout_env = self.env_func().to(self.config.sampling_device)

    def _setup_name(self):
        self.algorithm_name = self.algorithm_config.associated_class().__name__.lower()
        self.model_name = self.model_config.associated_class().__name__.lower()
        self.critic_model_name = (
            self.critic_model_config.associated_class().__name__.lower()
        )
        self.environment_name = self.task.env_name().lower()
        self.task_name = self.task.name.lower()
        self._checkpointed_files = deque([])

        if self.config.save_folder is not None:
            # If the user specified a folder for the experiment we use that
            save_folder = Path(self.config.save_folder)
        else:
            # Otherwise, if the user is restoring from a folder, we will save in the folder they are restoring from
            if self.config.restore_file is not None:
                save_folder = Path(
                    self.config.restore_file
                ).parent.parent.parent.resolve()
            # Otherwise, the user is not restoring and did not specify a save_folder so we save in the hydra directory
            # of the experiment or in the directory where the experiment was run (if hydra is not used)
            else:
                if _has_hydra and HydraConfig.initialized():
                    save_folder = Path(HydraConfig.get().runtime.output_dir)
                else:
                    save_folder = Path(os.getcwd())

        if self.config.restore_file is None:
            self.name = generate_exp_name(
                f"{self.algorithm_name}_{self.task_name}_{self.model_name}", ""
            )
            self.folder_name = save_folder / self.name

        else:
            # If restoring, we use the name of the previous experiment
            self.name = Path(self.config.restore_file).parent.parent.resolve().name
            self.folder_name = save_folder / self.name

        self.folder_name.mkdir(parents=False, exist_ok=True)
        with open(self.folder_name / "config.pkl", "wb") as f:
            pickle.dump(self.task, f)
            pickle.dump(self.task.config if self.task.config is not None else {}, f)
            pickle.dump(self.algorithm_config, f)
            pickle.dump(self.model_config, f)
            pickle.dump(self.seed, f)
            pickle.dump(self.config, f)
            pickle.dump(self.critic_model_config, f)
            pickle.dump(self.callbacks, f)

    def _setup_logger(self):
        self.logger = Logger(
            experiment_name=self.name,
            folder_name=str(self.folder_name),
            experiment_config=self.config,
            algorithm_name=self.algorithm_name,
            model_name=self.model_name,
            environment_name=self.environment_name,
            task_name=self.task_name,
            group_map=self.group_map,
            seed=self.seed,
            project_name=self.config.project_name,
            wandb_extra_kwargs=self.config.wandb_extra_kwargs,
        )
        self.logger.log_hparams(
            critic_model_name=self.critic_model_name,
            experiment_config=self.config.__dict__,
            algorithm_config=self.algorithm_config.__dict__,
            model_config=self.model_config.__dict__,
            critic_model_config=self.critic_model_config.__dict__,
            task_config=self.task.config,
            continuous_actions=self.continuous_actions,
            on_policy=self.on_policy,
        )

    def run(self):
        """Run the experiment until completion."""
        try:
            seed_everything(self.seed)
            torch.cuda.empty_cache()
            self._collection_loop()
        except KeyboardInterrupt as interrupt:
            print("\n\nExperiment was closed gracefully\n\n")
            self.close()
            raise interrupt
        except Exception as err:
            print("\n\nExperiment failed and is closing gracefully\n\n")
            self.close()
            raise err

    def evaluate(self):
        """Run just the evaluation loop once."""
        seed_everything(self.seed)
        self._evaluation_loop()
        self.logger.commit()
        print(
            f"Evaluation results logged to loggers={self.config.loggers}"
            f"{' and to a json file in the experiment folder.' if self.config.create_json else ''}"
        )

    def _collection_loop(self):
        pbar = tqdm(
            initial=self.n_iters_performed,
            total=self.config.get_max_n_iters(self.on_policy),
        )

        if not self.config.collect_with_grad:
            iterator = iter(self.collector)
        else:
            reset_batch = self.rollout_env.reset()

        # [PERF] 采集/训练重叠：后台线程预取下一批，与 GPU 训练重叠执行
        _overlap = (
            os.environ.get("OVERLAP_COLLECTION", "0") == "1"
            and not self.config.collect_with_grad
        )
        if _overlap:
            self._prefetcher = _CollectionPrefetcher(iterator, self.collector)
            print(
                "[PERF] OVERLAP_COLLECTION=1 -> 采集与训练重叠执行（后台预取线程），"
                "预取批使用上一次同步的权重（1 次迭代滞后）"
            )
        else:
            self._prefetcher = None

        # [DIAG] 纯滞后臂：串行采集，但延迟一轮消费（数据滞后 1 次更新，无任何并发）
        _delay_consume = (
            os.environ.get("DELAY_CONSUME", "0") == "1" and not _overlap
        )
        _delay_pending = None
        _force_no_train = os.environ.get("FORCE_SKIP_TRAIN", "0") == "1"
        if _force_no_train:
            print("[DIAG] FORCE_SKIP_TRAIN=1 -> 本 run 不执行任何优化器更新（隔离'训练并发'的影响）")
        if _delay_consume:
            print(
                "[DIAG] DELAY_CONSUME=1 -> 串行采集、延迟一轮消费（隔离验证“1 次更新滞后”本身的影响）"
            )

        # Training/collection iterations
        for _ in range(
            self.n_iters_performed, self.config.get_max_n_iters(self.on_policy)
        ):
            iteration_start = time.time()
            torch.cuda.empty_cache()
            if not self.config.collect_with_grad:
                if self._prefetcher is not None:
                    batch, _prefetch_dt = self._prefetcher.get()
                else:
                    self.collector.update_policy_weights_()
            self.policy.eval()
            if not self.config.collect_with_grad:
                if self._prefetcher is None:
                    batch = next(iterator)
            else:
                with set_exploration_type(ExplorationType.RANDOM):
                    batch = self.rollout_env.rollout(
                        max_steps=-(
                            -self.config.collected_frames_per_batch(self.on_policy)
                            // self.rollout_env.batch_size.numel()
                        ),
                        policy=self.policy,
                        break_when_any_done=False,
                        auto_reset=False,
                        tensordict=reset_batch,
                        auto_cast_to_device=True,
                    )
                    reset_batch = step_mdp(
                        batch[..., -1],
                        reward_keys=self.rollout_env.reward_keys,
                        action_keys=self.rollout_env.action_keys,
                        done_keys=self.rollout_env.done_keys,
                    )

            # Logging collection
            if self._prefetcher is not None:
                collection_time = _prefetch_dt
            else:
                collection_time = time.time() - iteration_start
            print(f"collection time: {collection_time}")
            if (
                self._prefetcher is not None
                and os.environ.get("OV_TRIGGER", "before") != "after"
                and _ < (self.config.get_max_n_iters(self.on_policy) - 1)
            ):
                # 本批采集已取回、下一批尚未开始：此刻同步权重并触发下一批后台预取
                self._prefetcher.trigger_next()
            self.policy.train()
            current_frames = batch.numel()
            self.total_frames += current_frames
            self.mean_return = self.logger.log_collection(
                batch,
                total_frames=self.total_frames,
                task=self.task,
                step=self.n_iters_performed,
            )
            pbar.set_description(f"mean return = {self.mean_return}", refresh=False)

            # [DIAG] 延迟消费：本轮采集的批次留待下一轮训练；本轮训练上一轮采集的批次
            _skip_train = _force_no_train
            if _delay_consume:
                if _delay_pending is None:
                    _delay_pending = batch
                    _skip_train = True
                    print("[DIAG] first iteration: collect only (no training)")
                else:
                    batch, _delay_pending = _delay_pending, batch

            # Callback
            self._on_batch_collected(batch)
            batch = batch.detach()
            torch.cuda.empty_cache()
            # Loop over groups
            training_start = time.time()
            # [PROF] 临时分段计时
            _prof = {}
            # [MEM] 显存分段探针（PROF_MEM=1 时启用）
            _mem_on = os.environ.get("PROF_MEM", "0") == "1"
            if _mem_on:
                torch.cuda.reset_peak_memory_stats()

            def _memlog(tag):
                if _mem_on:
                    print(
                        f"[MEM][{tag}] alloc={torch.cuda.memory_allocated() / 2**30:.2f}GB "
                        f"peak_alloc={torch.cuda.max_memory_allocated() / 2**30:.2f}GB "
                        f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GB",
                        flush=True,
                    )

            _active_groups = {} if _skip_train else self.train_group_map
            for group in _active_groups.keys():
                _prof[group] = {}
                _t0 = time.perf_counter()
                group_batch = batch.exclude(*self._get_excluded_keys(group)).to(
                    self.config.train_device
                )
                _memlog(f"{group}:to_gpu")
                _t1 = time.perf_counter()
                group_batch = self.algorithm.process_batch(group, group_batch)
                _b0 = time.perf_counter()
                _memlog(f"{group}:process_batch")
                _t2 = _b0
                if not self.algorithm.has_rnn:
                    group_batch = group_batch.reshape(-1)
                group_buffer = self.replay_buffers[group]
                group_buffer.extend(group_batch.to(group_buffer.storage.device))
                _t3 = time.perf_counter()
                _memlog(f"{group}:extend")
                self._prof_opt = {
                    "sample": 0.0,
                    "forward": 0.0,
                    "bwd_step": 0.0,
                    "tail": 0.0,
                    "count": 0,
                }
                training_tds = []
                for _ in range(self.config.n_optimizer_steps(self.on_policy)):
                    for _ in range(
                        -(
                            -self.config.train_batch_size(self.on_policy)
                            // self.config.train_minibatch_size(self.on_policy)
                        )
                    ):
                        training_tds.append(self._optimizer_loop(group))
                _t4 = time.perf_counter()
                _memlog(f"{group}:opt_loops")
                training_td = torch.stack(training_tds)
                self.logger.log_training(
                    group, training_td, step=self.n_iters_performed
                )
                _t5 = time.perf_counter()
                _prof[group] = {
                    "exclude_to_gpu": _t1 - _t0,
                    "process_batch": _t2 - _t1,
                    "extend": _t3 - _t2,
                    "opt_loops": _t4 - _t3,
                    "stack_log": _t5 - _t4,
                    "opt_detail": dict(self._prof_opt),
                }
                # Callback
                self._on_train_end(training_td, group)
                # Exploration update
                if isinstance(self.group_policies[group], TensorDictSequential):
                    explore_layer = self.group_policies[group][-1]
                else:
                    explore_layer = self.group_policies[group]
                if hasattr(explore_layer, "step"):  # Step exploration annealing
                    explore_layer.step(current_frames)
            # Update policy in collector
            if not self.config.collect_with_grad and self._prefetcher is None:
                # 重叠模式下同步已在 trigger_next() 里做过；此处若再同步会与后台采集竞争
                self.collector.update_policy_weights_()
            lr_log = {}
            for group in _active_groups.keys():
                if group in self.lr_schedulers:
                    for loss_name, scheduler in self.lr_schedulers[group].items():
                        if scheduler is not None:
                            scheduler.step()
                            current_lr = scheduler.get_last_lr()[0]
                            lr_log[f"lr/{group}/{loss_name}"] = current_lr
            if lr_log:
                self.logger.log(lr_log, step=self.n_iters_performed)
            # Training timer
            training_time = time.time() - training_start
            # [PROF] 临时打印分段计时
            for _g, _d in _prof.items():
                _od = _d["opt_detail"]
                print(
                    f"[PROF][{_g}] exclude_to_gpu={_d['exclude_to_gpu']:.1f}s "
                    f"process_batch={_d['process_batch']:.1f}s extend={_d['extend']:.1f}s "
                    f"opt_loops={_d['opt_loops']:.1f}s stack_log={_d['stack_log']:.1f}s || "
                    f"opt_detail(n={_od['count']}) sample={_od['sample']:.1f}s "
                    f"forward={_od['forward']:.1f}s bwd_step={_od['bwd_step']:.1f}s "
                    f"tail={_od['tail']:.1f}s"
                )

            # [DIAG] OV_TRIGGER=after：把触发挪到训练之后（保留后台采集线程机制，但去掉采集与训练的重叠）
            if (
                self._prefetcher is not None
                and os.environ.get("OV_TRIGGER", "before") == "after"
                and _ < (self.config.get_max_n_iters(self.on_policy) - 1)
            ):
                self._prefetcher.trigger_next()

            # Evaluation
            if (
                self.config.evaluation
                and (
                    self.total_frames % self.config.evaluation_interval == 0
                    or self.total_frames - self._last_eval_frames
                    >= self.config.evaluation_interval
                    or self.n_iters_performed == 0
                )
                and (len(self.config.loggers) or self.config.create_json)
            ):
                # 将策略拷贝到 CPU 并序列化
                # 使用 state_dict() 时加上 .to("cpu") 确保子进程不触碰 CUDA
                policy_state_dict = {
                    k: v.cpu().detach() if isinstance(v, torch.Tensor) else v
                    for k, v in self.policy.state_dict().items()
                }

                # 尝试推送到队列（非阻塞）
                try:
                    # 如果队列满了（旧的还没评完），先弹出旧的再放新的，保证评估的是最新权重
                    if self.eval_weight_queue.full():
                        try:
                            self.eval_weight_queue.get_nowait()
                        except:
                            pass

                    self.eval_weight_queue.put_nowait(
                        (policy_state_dict, self.total_frames, self.n_iters_performed)
                    )
                    self._last_eval_frames = self.total_frames
                    print(
                        f"\n[Main]: Sent weights for iteration {self.n_iters_performed} to Eval Worker."
                    )
                except Exception as e:
                    warnings.warn(f"Could not send weights to evaluation worker: {e}")
                # self._evaluation_loop()

            # End of step
            iteration_time = time.time() - iteration_start
            self.total_time += iteration_time
            self.logger.log(
                {
                    "timers/collection_time": collection_time,
                    "timers/training_time": training_time,
                    "timers/iteration_time": iteration_time,
                    "timers/total_time": self.total_time,
                    "counters/current_frames": current_frames,
                    "counters/total_frames": self.total_frames,
                    "counters/iter": self.n_iters_performed,
                },
                step=self.n_iters_performed,
            )
            self.n_iters_performed += 1
            self.logger.commit()
            if (
                self.config.checkpoint_interval > 0
                and (
                    self.total_frames % self.config.checkpoint_interval == 0
                    or self.total_frames - self._last_checkpoint_frames
                    >= self.config.checkpoint_interval
                )
            ):
                self._save_experiment()
                self._last_checkpoint_frames = self.total_frames
            pbar.update()

        if self.config.checkpoint_at_end:
            self._save_experiment()
        self.close()

    def close(self):
        """Close the experiment."""
        # [PERF] 停止采集预取线程（若启用重叠模式）
        prefetcher = getattr(self, "_prefetcher", None)
        if prefetcher is not None:
            prefetcher.close()
            self._prefetcher = None
        if self.evaluation_process is not None:
            print(
                "Waiting for background evaluation worker to finish pending evaluations..."
            )
            # [修复] 不要用 eval_weight_queue.put(None):
            #   该队列 maxsize=1, 队列满时 put 会阻塞, 而 worker 正在评估 (需 1~3 分钟),
            #   会造成主进程卡在 close() 里 (表现为"冒烟卡住")。
            # 只设置 stop_event 即可: worker 的循环会先把队列中 pending 的权重评估完
            # (get 成功 -> 继续评估), 只有当"队列为空且 stop_event 置位"时才退出。
            self.eval_stop_event.set()
            self.evaluation_process.join(timeout=180)
            if self.evaluation_process.is_alive():
                print("Evaluation worker did not finish within 180s, terminating...")
                self.evaluation_process.terminate()
        if not self.config.collect_with_grad:
            self.collector.shutdown()
        else:
            self.rollout_env.close()
        self.test_env.close()
        self.logger.finish()

        for buffer in self.replay_buffers.values():
            if hasattr(buffer.storage, "scratch_dir"):
                shutil.rmtree(buffer.storage.scratch_dir, ignore_errors=False)

    def _get_excluded_keys(self, group: str):
        excluded_keys = []
        for other_group in self.group_map.keys():
            if other_group != group:
                excluded_keys += [other_group, ("next", other_group)]
        excluded_keys += ["info", (group, "info"), ("next", group, "info")]
        return excluded_keys

    # ==================== [NaN 取证] ====================
    def _nan_finite_stats(self, t):
        """返回 (nonfinite 数, numel, finite_min, finite_max)；全部 finite 时返回 None。"""
        f = torch.isfinite(t)
        n_bad = int((~f).sum())
        if n_bad == 0:
            return None
        good = t[f]
        mn = float(good.min()) if good.numel() else float("nan")
        mx = float(good.max()) if good.numel() else float("nan")
        return n_bad, int(t.numel()), mn, mx

    def _iter_float_leaves(self, obj, prefix=""):
        """递归遍历 TensorDict/dict/list，产出 (路径, 浮点张量)。"""
        if torch.is_tensor(obj):
            if obj.is_floating_point():
                yield prefix, obj
            return
        if hasattr(obj, "items"):
            for k, v in obj.items():
                yield from self._iter_float_leaves(v, f"{prefix}/{k}" if prefix else str(k))
            return
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                yield from self._iter_float_leaves(v, f"{prefix}[{i}]")

    def _scan_nonfinite_td(self, td, tag, cap=5):
        """[NaN 取证] 扫描浮点叶子，发现非有限就打印（每次实验最多 cap 条）。"""
        rows = []
        for key, val in self._iter_float_leaves(td):
            st = self._nan_finite_stats(val)
            if st is not None:
                rows.append(
                    f"    {key}: {st[0]}/{st[1]} finite[min,max]=[{st[2]:.4g},{st[3]:.4g}]"
                )
        if not rows:
            return
        self._nan_scan_reports = getattr(self, "_nan_scan_reports", 0)
        if self._nan_scan_reports >= cap:
            return
        self._nan_scan_reports += 1
        print(f"[NaNScan] {tag}: 非有限叶子 {len(rows)} 个")
        for r in rows[:20]:
            print(r)

    def _dump_nan_case(self, group, loss_name, subdata, raw_loss_vals=None, cap=5):
        """[NaN 取证] 非有限 loss 现场：打印 loss 分量 + subdata 非有限叶子（可选保存 subdata）。"""
        self._nan_case_reports = getattr(self, "_nan_case_reports", 0)
        if self._nan_case_reports >= cap:
            return
        self._nan_case_reports += 1
        print(
            f"[NaNCase] iter={self.n_iters_performed} group={group} loss={loss_name}",
            flush=True,
        )
        if raw_loss_vals is not None:
            for key, val in self._iter_float_leaves(raw_loss_vals):
                st = self._nan_finite_stats(val)
                if st is not None:
                    print(f"[NaNCase]   loss[{key}] NONFINITE {st[0]}/{st[1]}")
                else:
                    mn = float(val.min()) if val.numel() else float("nan")
                    mx = float(val.max()) if val.numel() else float("nan")
                    print(f"[NaNCase]   loss[{key}] finite min={mn:.4g} max={mx:.4g}")
        found = 0
        for key, val in self._iter_float_leaves(subdata):
            st = self._nan_finite_stats(val)
            if st is not None:
                found += 1
                print(
                    f"[NaNCase]   subdata {key}: {st[0]}/{st[1]} "
                    f"finite[min,max]=[{st[2]:.4g},{st[3]:.4g}]"
                )
        print(f"[NaNCase]   subdata 非有限叶子合计 {found} 个", flush=True)
        if os.environ.get("NAN_SAVE", "0") == "1":
            path = f"/tmp/nan_case_iter{self.n_iters_performed}_{group}_{loss_name}.pt"
            try:
                torch.save(subdata.detach().cpu(), path)
                print(f"[NaNCase]   saved subdata -> {path}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[NaNCase]   save failed: {e}", flush=True)

    def _dump_nonfinite_grads(self, group, loss_name, optimizer, cap=5):
        """[NaN 取证] 打印含非有限梯度的参数名。"""
        self._nan_grad_reports = getattr(self, "_nan_grad_reports", 0)
        if self._nan_grad_reports >= cap:
            return
        names = {}
        try:
            for name, p in self.losses[group].named_parameters():
                names.setdefault(id(p), name)
        except Exception:  # noqa: BLE001
            pass
        bad = []
        for gi, pg in enumerate(optimizer.param_groups):
            for p in pg.get("params", []):
                if p is None or p.grad is None:
                    continue
                st = self._nan_finite_stats(p.grad)
                if st is not None:
                    bad.append((names.get(id(p), f"pg{gi}"), st))
        if not bad:
            return
        self._nan_grad_reports += 1
        print(
            f"[NaNGrad] iter={self.n_iters_performed} group={group} loss={loss_name} "
            f"非有限梯度参数 {len(bad)} 个",
            flush=True,
        )
        for name, st in bad[:20]:
            print(f"[NaNGrad]   {name}: {st[0]}/{st[1]} finite[min,max]=[{st[2]:.4g},{st[3]:.4g}]")

    def _optimizer_loop(self, group: str) -> TensorDictBase:
        _p0 = time.perf_counter()
        subdata = self.replay_buffers[group].sample().to(self.config.train_device)
        if os.environ.get("NAN_DEBUG", "0") == "1":
            self._scan_nonfinite_td(subdata, f"iter{self.n_iters_performed}/{group}/sample")
        _p1 = time.perf_counter()

        # 1. Forward pass (with mixed precision if enabled)
        if self.config.use_amp and self.config.train_device != "cpu":
            with autocast(device_type="cuda", dtype=self.amp_dtype):
                loss_vals = self.losses[group](subdata)
        else:
            loss_vals = self.losses[group](subdata)
        _p2 = time.perf_counter()

        training_td = loss_vals.detach()
        loss_vals = self.algorithm.process_loss_vals(group, loss_vals)

        # 2. Backward pass and optimization
        for loss_name, loss_value in loss_vals.items():
            if loss_name in self.optimizers[group].keys():
                optimizer = self.optimizers[group][loss_name]

                # [稳定化] 非有限 loss：跳过本次更新并清零梯度，防止 NaN 扩散到参数
                _lv = (
                    loss_value
                    if torch.is_tensor(loss_value)
                    else torch.as_tensor(loss_value)
                )
                if not bool(torch.isfinite(_lv).all()):
                    optimizer.zero_grad(set_to_none=True)
                    self._nonfinite_loss_skips = (
                        getattr(self, "_nonfinite_loss_skips", 0) + 1
                    )
                    print(
                        f"[SafeTrain] 非有限 loss ({group}/{loss_name})，跳过本次更新"
                        f" | 累计 {self._nonfinite_loss_skips} 次"
                    )
                    training_td.set(
                        f"grad_norm_{loss_name}",
                        torch.tensor(float("nan"), device=_lv.device),
                    )
                    self._dump_nan_case(group, loss_name, subdata, training_td)
                    continue

                if self.config.use_amp and self.config.train_device != "cpu":
                    if group in self.grad_scalers:
                        # fp16 流程: 需要 loss scaling
                        scaler = self.grad_scalers[group]

                        # Scale loss and backward
                        scaler.scale(loss_value).backward()

                        # Unscale gradients for clipping
                        scaler.unscale_(optimizer)

                        # Gradient clipping (on unscaled FP32 gradients)
                        grad_norm_tensor = self._grad_clip(optimizer)

                        # Optimizer step with scaler
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        # bf16 流程: 无需 loss scaling, 直接反向+步进
                        loss_value.backward()
                        grad_norm_tensor = self._grad_clip(optimizer)
                        # [稳定化] 梯度非有限则跳过参数更新（清零梯度），避免 NaN 写进权重
                        if bool(torch.isfinite(grad_norm_tensor).all()):
                            optimizer.step()
                        else:
                            optimizer.zero_grad(set_to_none=True)
                            self._nonfinite_grad_skips = (
                                getattr(self, "_nonfinite_grad_skips", 0) + 1
                            )
                            print(
                                f"[SafeTrain] 梯度非有限 ({group}/{loss_name})，跳过 optimizer.step()"
                                f" | 累计 {self._nonfinite_grad_skips} 次"
                            )
                            self._dump_nonfinite_grads(group, loss_name, optimizer)
                        optimizer.zero_grad()
                else:
                    # Original FP32 training flow
                    loss_value.backward()
                    grad_norm_tensor = self._grad_clip(optimizer)
                    # [稳定化] 梯度非有限则跳过参数更新（清零梯度），避免 NaN 写进权重
                    if bool(torch.isfinite(grad_norm_tensor).all()):
                        optimizer.step()
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        self._nonfinite_grad_skips = (
                            getattr(self, "_nonfinite_grad_skips", 0) + 1
                        )
                        print(
                            f"[SafeTrain] 梯度非有限 ({group}/{loss_name})，跳过 optimizer.step()"
                            f" | 累计 {self._nonfinite_grad_skips} 次"
                        )
                        self._dump_nonfinite_grads(group, loss_name, optimizer)
                    optimizer.zero_grad()

                training_td.set(
                    f"grad_norm_{loss_name}",
                    grad_norm_tensor.detach(),
                )
        _p3 = time.perf_counter()

        # 3. Update replay buffer priority and target networks
        self.replay_buffers[group].update_tensordict_priority(subdata)
        if self.target_updaters[group] is not None:
            self.target_updaters[group].step()

        callback_loss = self._on_train_step(subdata, group)
        if callback_loss is not None:
            training_td.update(callback_loss)
        _p4 = time.perf_counter()

        _acc = getattr(self, "_prof_opt", None)
        if _acc is not None:
            _acc["sample"] += _p1 - _p0
            _acc["forward"] += _p2 - _p1
            _acc["bwd_step"] += _p3 - _p2
            _acc["tail"] += _p4 - _p3
            _acc["count"] += 1

        return training_td

    def _grad_clip(self, optimizer: torch.optim.Optimizer) -> float:
        params = []
        for param_group in optimizer.param_groups:
            params += param_group["params"]

        if self.config.clip_grad_norm and self.config.clip_grad_val is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                params, self.config.clip_grad_val
            )
        else:
            norm_type = 2.0
            norms = [
                torch.linalg.vector_norm(p.grad, norm_type)
                for p in params
                if p.grad is not None
            ]
            total_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type)
            if self.config.clip_grad_val is not None:
                torch.nn.utils.clip_grad_value_(params, self.config.clip_grad_val)

        return total_norm

    @local_seed()
    @torch.no_grad()
    def _evaluation_loop(self):
        if self.config.evaluation_static:
            seed_everything(self.seed)
            try:
                self.test_env.set_seed(self.seed)
            except NotImplementedError:
                warnings.warn(
                    "`experiment.evaluation_static` set to true but the environment does not allow to set seeds."
                    "Static evaluation is not guaranteed."
                )
        evaluation_start = time.time()
        with set_exploration_type(
            ExplorationType.DETERMINISTIC
            if self.config.evaluation_deterministic_actions
            else ExplorationType.RANDOM
        ):
            if self.task.has_render(self.test_env) and self.config.render:
                video_frames = []

                def callback(env, td):
                    video_frames.append(
                        self.task.__class__.render_callback(self, env, td)
                    )

            else:
                video_frames = None
                callback = None

            if self.test_env.batch_size == ():
                rollouts = []
                for eval_episode in range(self.config.evaluation_episodes):
                    rollouts.append(
                        self.test_env.rollout(
                            max_steps=self.max_steps,
                            policy=self.policy,
                            callback=callback if eval_episode == 0 else None,
                            auto_cast_to_device=True,
                            break_when_any_done=True,
                        )
                    )
            else:
                rollouts = self.test_env.rollout(
                    max_steps=self.max_steps,
                    policy=self.policy,
                    callback=callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
                )
                rollouts = list(rollouts.unbind(0))
        evaluation_time = time.time() - evaluation_start
        self.logger.log(
            {"timers/evaluation_time": evaluation_time}, step=self.n_iters_performed
        )
        self.logger.log_evaluation(
            rollouts,
            video_frames=video_frames,
            step=self.n_iters_performed,
            total_frames=self.total_frames,
        )
        # Callback
        self._on_evaluation_end(rollouts)

    # Saving experiment state
    def state_dict(self) -> OrderedDict:
        """Get the state_dict for the experiment."""
        state = OrderedDict(
            total_time=self.total_time,
            total_frames=self.total_frames,
            n_iters_performed=self.n_iters_performed,
            mean_return=self.mean_return,
        )
        state_dict = OrderedDict(
            state=state,
            **{f"loss_{k}": item.state_dict() for k, item in self.losses.items()},
            **{
                f"buffer_{k}": item.state_dict() if len(item) else None
                for k, item in self.replay_buffers.items()
            },
        )
        if not self.config.collect_with_grad:
            state_dict.update({"collector": self.collector.state_dict()})

        # Save GradScaler state for mixed precision training
        if self.config.use_amp and self.config.train_device != "cpu":
            state_dict.update(
                {
                    f"grad_scaler_{k}": scaler.state_dict()
                    for k, scaler in self.grad_scalers.items()
                }
            )

        # Save optimizer state
        for group in self.group_map.keys():
            state_dict[f"optimizer_{group}"] = {
                name: opt.state_dict() for name, opt in self.optimizers[group].items()
            }

        # Save learning rate scheduler state
        if self.config.lr_scheduler != "constant":
            for group in self.group_map.keys():
                state_dict[f"lr_scheduler_{group}"] = {
                    name: sched.state_dict() if sched is not None else None
                    for name, sched in self.lr_schedulers[group].items()
                }

        return state_dict

    def _buffer_state_is_compatible(self, group: str, buffer_state: Dict) -> bool:
        """检查 checkpoint 内 replay buffer 的存储形状是否与当前配置匹配。

        跨配置恢复（n_envs / frames_per_batch 改变）时行数或时间维不同，
        直接 load_state_dict 会让后续写入越界（CUDA index out of bounds）。
        """
        try:
            saved = buffer_state["_storage"]["_storage"]["state"]
            saved_rows = int(saved.shape[0])
            saved_t = int(saved.shape[1]) if saved.ndim > 1 else None
        except Exception:
            return True

        on_policy = self.algorithm.on_policy
        n_envs = self.config.n_envs_per_worker(on_policy) * self.config.n_workers
        frames_per_batch = self.config.collected_frames_per_batch(on_policy)
        if self.algorithm.has_rnn:
            expected_rows = n_envs
            expected_t = -(-frames_per_batch // n_envs)
            ok = (saved_rows == expected_rows) and (
                saved_t is None or saved_t == expected_t
            )
            if not ok:
                print(
                    f"[BufferGuard] {group}: ckpt buffer 形状 (rows={saved_rows}, T={saved_t})"
                    f" != 当前配置 (rows={expected_rows}, T={expected_t})"
                )
            return ok
        if saved_rows != frames_per_batch:
            print(
                f"[BufferGuard] {group}: ckpt buffer 行数 {saved_rows}"
                f" != 当前配置 {frames_per_batch}"
            )
            return False
        return True

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load the state_dict for the experiment.

        Args:
            state_dict (dict): the state dict

        """
        for group in self.group_map.keys():
            self.losses[group].load_state_dict(state_dict[f"loss_{group}"])
            if state_dict[f"buffer_{group}"] is not None:
                # [修复] 跨配置恢复 buffer 会导致存储形状不匹配：
                # 例如 ckpt 存于 1500 envs/225k frames，当前配置 2000 envs/300k frames，
                # 恢复后 storage 只有 1500 行，写入 2000 行时触发
                # CUDA "index out of bounds" (IndexKernel.cu) 崩溃。
                # on-policy buffer 内容每轮都会被重写，可直接跳过。
                if self._buffer_state_is_compatible(
                    group, state_dict[f"buffer_{group}"]
                ):
                    self.replay_buffers[group].load_state_dict(
                        state_dict[f"buffer_{group}"]
                    )
                    want = self._configured_buffer_batch_size.get(group)
                    buf = self.replay_buffers[group]
                    got = getattr(buf, "_batch_size", None)
                    if want is not None and got != want:
                        buf._batch_size = want
                        print(
                            f"[BufferGuard] {group}: 采样 batch 大小按当前配置对齐 "
                            f"{got} -> {want}（checkpoint 内旧值与当前 minibatch 配置不一致）"
                        )
                else:
                    print(
                        f"[BufferGuard] 跳过 {group} 的 replay buffer 恢复："
                        f"checkpoint 内存储形状与当前配置不一致"
                        f"（on-policy 缓冲内容可丢弃，不影响训练）。"
                    )

            # Load GradScaler state for mixed precision training
            # [修复] bf16 模式下 grad_scalers 为空 dict, 直接索引会 KeyError
            # (旧 checkpoint 里可能仍带有 grad_scaler_* 状态)
            if self.config.use_amp and self.config.train_device != "cpu":
                scaler_key = f"grad_scaler_{group}"
                if scaler_key in state_dict and group in self.grad_scalers:
                    self.grad_scalers[group].load_state_dict(state_dict[scaler_key])

            # Load optimizer state
            opt_key = f"optimizer_{group}"
            if opt_key in state_dict:
                for name, opt_state in state_dict[opt_key].items():
                    if name in self.optimizers[group]:
                        self.optimizers[group][name].load_state_dict(opt_state)

            # Load learning rate scheduler state
            if self.config.lr_scheduler != "constant":
                sched_key = f"lr_scheduler_{group}"
                if sched_key in state_dict and group in self.lr_schedulers:
                    for name, sched_state in state_dict[sched_key].items():
                        if (
                            sched_state is not None
                            and name in self.lr_schedulers[group]
                        ):
                            self.lr_schedulers[group][name].load_state_dict(sched_state)

        if not self.config.collect_with_grad:
            try:
                self.collector.load_state_dict(state_dict["collector"])
            except RuntimeError as e:
                # [修复] n_envs 变化后 CatFrames 历史缓冲形状不匹配（[1000,2,410] -> [1500,2,410]）
                # 收集器状态（历史缓冲/RNG）非必要，丢弃并用全新状态继续训练
                warnings.warn(
                    "collector state_dict 恢复失败（通常是 n_envs 变了），"
                    f"改用全新收集器状态继续训练: {e}"
                )
        self.total_time = state_dict["state"]["total_time"]
        self.total_frames = state_dict["state"]["total_frames"]
        self.n_iters_performed = state_dict["state"]["n_iters_performed"]
        self.mean_return = state_dict["state"]["mean_return"]
        # [修复] 续训时的增量触发基准：存档从恢复点起每 checkpoint_interval 帧一次；
        # 评测从恢复点起立刻做一次（便于确认评测链路正常），之后每 evaluation_interval 帧一次
        self._last_checkpoint_frames = self.total_frames
        self._last_eval_frames = self.total_frames - self.config.evaluation_interval

    def _save_experiment(self) -> None:
        """Checkpoint trainer"""
        if self.config.keep_checkpoints_num is not None:
            while len(self._checkpointed_files) >= self.config.keep_checkpoints_num:
                file_to_delete = self._checkpointed_files.popleft()
                file_to_delete.unlink(missing_ok=False)

        checkpoint_folder = self.folder_name / "checkpoints"
        checkpoint_folder.mkdir(parents=False, exist_ok=True)
        checkpoint_file = checkpoint_folder / f"checkpoint_{self.total_frames}.pt"
        torch.save(self.state_dict(), checkpoint_file)
        self._checkpointed_files.append(checkpoint_file)

    def _load_experiment(self) -> Experiment:
        """Load trainer from checkpoint"""
        loaded_dict: OrderedDict = torch.load(
            self.config.restore_file, map_location=self.config.restore_map_location
        )
        self.load_state_dict(loaded_dict)
        return self

    @staticmethod
    def reload_from_file(restore_file: str) -> Experiment:
        """
        Restores the experiment from the checkpoint file.

        This method expects the same folder structure created when an experiment is run.
        The checkpoint file (``restore_file``) is in the checkpoints directory and a config.pkl file is
        present a level above at restore_file/../../config.pkl

        Args:
            restore_file (str): The checkpoint file (.pt) of the experiment reload.

        Returns:
            The reloaded experiment.

        """
        experiment_folder = Path(restore_file).parent.parent.resolve()
        config_file = experiment_folder / "config.pkl"
        if not os.path.exists(config_file):
            raise ValueError("config.pkl file not found in experiment folder.")
        with open(config_file, "rb") as f:
            task = pickle.load(f)
            task_config = pickle.load(f)
            algorithm_config = pickle.load(f)
            model_config = pickle.load(f)
            seed = pickle.load(f)
            experiment_config = pickle.load(f)
            critic_model_config = pickle.load(f)
            callbacks = pickle.load(f)
        task.config = task_config
        experiment_config.restore_file = restore_file
        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            seed=seed,
            config=experiment_config,
            callbacks=callbacks,
            critic_model_config=critic_model_config,
        )
        print(f"\nReloaded experiment {experiment.name} from {restore_file}.")
        return experiment
