#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import copy
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import vmas
from tensordict import TensorDictBase
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.transforms import CatFrames, Transform
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import MarlGroupMapType

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

class FlattenHybridAction(Transform):
    """
    动作转换层：
    1.  [Agent -> Env] _inv_call: 将混合动作压扁成 [vx, vy, trigger] 给 VMAS。
    2.  [Env -> Agent] forward: 不做处理 (或者处理 Observation，如果需要)。
    """
    def __init__(self, continuous_dim: int = 2, discrete_dim: int = 1, discrete_n: int = 2):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.discrete_dim = discrete_dim
        self.discrete_n = discrete_n
        self._debug_printed = False

    # [关键修复] forward 是处理 Observation 的，动作处理必须在 _inv_call 中！
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        root_keys = list(tensordict.keys())
        
        for key in root_keys:
            sub_td = tensordict.get(key)
            
            # 检查结构: group -> action
            if isinstance(sub_td, TensorDictBase) and "action" in sub_td.keys():
                action_entry = sub_td.get("action")
                
                # 确认是混合动作容器
                if (isinstance(action_entry, TensorDictBase) 
                    and "continuous" in action_entry.keys() 
                    and "discrete" in action_entry.keys()):
                    
                    act_c = action_entry.get("continuous")
                    act_d = action_entry.get("discrete")
                    
                    # 1. 类型转换 & 维度对齐
                    act_d_float = act_d.float()
                    if act_d_float.dim() == act_c.dim() - 1:
                        act_d_float = act_d_float.unsqueeze(-1)
                    
                    # 2. 拼接
                    flat_action = torch.cat([act_c, act_d_float], dim=-1)
                    
                    # 3. 删除旧容器，写入新 Tensor
                    del sub_td["action"]
                    sub_td.set("action", flat_action)

        return tensordict
    
    # 必须保留 forward，否则 Transform 基类会报错或行为异常
    # 对于 Action Transform，forward 通常是 Identity (原样返回)
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _continuous_bounds(self, original_spec):
        """[速度范围 B] 读取原始动作 spec 前 continuous_dim 维的 (low, high)。

        直接从原始 leaf spec 读取 ⇒ 自动跟随 v_max / action_size 变化（向前兼容）；
        读不到上下界时返回 (None, None)，调用方退回 Unbounded（旧行为）。
        """
        space = getattr(original_spec, "space", None)
        low = getattr(space, "low", None) if space is not None else None
        high = getattr(space, "high", None) if space is not None else None
        if low is None or high is None:
            low = getattr(original_spec, "low", None)
            high = getattr(original_spec, "high", None)
        if low is None or high is None:
            return None, None
        dev = getattr(original_spec, "device", None)
        low = torch.as_tensor(low, dtype=torch.float32, device=dev)[..., : self.continuous_dim]
        high = torch.as_tensor(high, dtype=torch.float32, device=dev)[..., : self.continuous_dim]
        return low, high

    def _hybridize(self, full_action_spec: Composite) -> Composite:
        """把每个 group 的 "action" 叶子 spec 换成 {continuous, discrete} 复合 spec。"""
        for group_key in list(full_action_spec.keys()):
            group_spec = full_action_spec[group_key]
            if "action" not in group_spec.keys(): continue
                
            original_spec = group_spec["action"]
            if isinstance(original_spec, Composite): continue

            # 保留 Batch 维度
            target_shape = original_spec.shape[:-1]
            
            # [速度范围 B] 从原始动作叶子 spec 还原连续分量的物理上下界（±v_max）。
            # 不还原的话 TanhNormal 退化成平凡边界 (−1,1)：策略只能输出 1 m/s，
            # 且 loc 与事件空间失配会把 log_prob 推到 ~1e4（NaN 诱因，见 m08857 取证）。
            cont_low, cont_high = self._continuous_bounds(original_spec)

            new_spec = Composite(shape=target_shape, device=original_spec.device)
            if cont_low is not None:
                new_spec["continuous"] = Bounded(
                    low=cont_low,
                    high=cont_high,
                    shape=target_shape + (self.continuous_dim,),
                    device=original_spec.device,
                    dtype=torch.float32
                )
            else:
                new_spec["continuous"] = Unbounded(
                    shape=target_shape + (self.continuous_dim,),
                    device=original_spec.device,
                    dtype=torch.float32
                )
            new_spec["discrete"] = Categorical(
                n=self.discrete_n,
                shape=target_shape + (self.discrete_dim,),
                device=original_spec.device,
                dtype=torch.long
            )
            full_action_spec[group_key]["action"] = new_spec

        return full_action_spec

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        """
        Step 2: 欺骗 Mappo (保留 Batch 维度)
        """
        if "full_action_spec" not in input_spec.keys():
            return input_spec
        input_spec["full_action_spec"] = self._hybridize(input_spec["full_action_spec"])
        return input_spec

    def transform_action_spec(self, action_spec: Composite) -> Composite:
        # [投篮按键] TransformedEnv.action_spec 走这条；不写的话对外仍是 3 维连续 spec
        return self._hybridize(action_spec)

class VmasEnvWithState(VmasEnv):
    """
    带有全局状态支持的 VMAS 环境封装。
    """
    def _make_specs(
        self, env: vmas.simulator.environment.environment.Environment
    ) -> None:
        super()._make_specs(env)

        try:
            sample_state = self._env.scenario.get_global_state()
        except AttributeError:
            raise AttributeError(
                "The environment's scenario must have a 'get_global_state()' method."
            )
        full_state_spec_unbatched = Composite(device=self.device)
        state_dim_shape = sample_state.shape[1:]
        unbatched_state_spec_value = Unbounded(
            shape=state_dim_shape,
            device=self.device,
            dtype=sample_state.dtype,
        )
            
        full_state_spec_unbatched["state"] = unbatched_state_spec_value
        self.full_state_spec_unbatched = full_state_spec_unbatched
        
        # 将 state 加入 observation spec，以便 BenchMARL 处理
        observation_spec_unbatched = self.observation_spec_unbatched
        observation_spec_unbatched["state"] = unbatched_state_spec_value

        # [投篮按键] 在观测 spec 中声明组级 action_mask: [n_agents, 2] bool
        # （Actor 观测 spec 会把它删掉，单独走 TaskClass.action_mask_spec -> mappo 的 mask 注入）
        n_agents = len(self._env.scenario.world.agents)
        group_key = next(iter(self.group_map))
        observation_spec_unbatched[(group_key, "action_mask")] = Categorical(
            n=2,
            shape=(n_agents, 2),
            dtype=torch.bool,
            device=self.device,
        )
        self.observation_spec_unbatched = observation_spec_unbatched

    def _write_action_mask(self, tensordict_out: TensorDictBase) -> TensorDictBase:
        # [投篮按键] 把场景的动作 mask 写进组子 td（与观测 spec 的 (group,"action_mask") 对齐）
        getter = getattr(self._env.scenario, "get_action_mask", None)
        if getter is None:
            return tensordict_out
        mask = getter()
        has_next = "next" in tensordict_out.keys()
        for group in self.group_map.keys():
            tensordict_out.set((group, "action_mask"), mask)
            # torchrl 的 step 输出约定：新一帧的数据在 "next" 下，step_mdp 之后才提升到根；
            # 只写根的话会被 step_mdp 用旧值覆盖，所以两边都写。
            if has_next:
                tensordict_out.set(("next", group, "action_mask"), mask)
        return tensordict_out

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        tensordict_out = super()._reset(tensordict, **kwargs)
        state = self._env.scenario.get_global_state()
        tensordict_out.set("state", state)
        return self._write_action_mask(tensordict_out)

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        tensordict_out = super()._step(tensordict)
        next_state = self._env.scenario.get_global_state()
        tensordict_out.set("state", next_state)
        return self._write_action_mask(tensordict_out)


class StridedCatFrames(CatFrames):
    """
    [冒烟实验] 带采样间隔的 CatFrames。

    CatFrames 每步都会写入一帧, 因此只能表达"最近 N 个连续步"。
    本类每隔 ``stride`` 步才写入一帧, 用较少的帧覆盖更长的时间:
        history_frames=5, stride=5, dt=0.1s -> 覆盖 5*5*0.1 = 2.0s, 维度只放大 5 倍。

    实现: 继承 CatFrames, 仅在采样步复用其"滚动 + 写入"逻辑;
    非采样步只把当前 buffer 回填到 tensordict (输出保持不变)。
    """

    def __init__(self, N: int, stride: int = 1, **kwargs):
        super().__init__(N=N, **kwargs)
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError(f"stride 必须 >= 1, 得到 {self.stride}")
        self._stride_counter = 0

    def _call(self, next_tensordict: TensorDictBase, _reset=None) -> TensorDictBase:
        if self.stride == 1:
            return super()._call(next_tensordict, _reset=_reset)

        self._stride_counter += 1
        _just_reset = _reset is not None
        if _just_reset and bool(_reset.all()):
            self._stride_counter = 1
        is_sample = (self._stride_counter - 1) % self.stride == 0

        for in_key, out_key in zip(self.in_keys, self.out_keys):
            data = next_tensordict.get(in_key)
            d = data.size(self.dim)
            buffer_name = f"_cat_buffers_{in_key}"
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                buffer = self._make_missing_buffer(data, buffer_name)

            shape = [1] * data.ndim
            shape[self.dim] = self.N

            if _just_reset and bool(_reset.all()):
                # 全量 reset: 用当前帧填满整个历史窗口 (等价 padding="same")
                buffer.copy_(data.repeat(shape))
            else:
                # 1) 采样步: 对"本步未 reset 的 env"执行滚动 + 写入最新帧
                if is_sample:
                    if _just_reset:
                        keep = (~_reset).nonzero(as_tuple=True)[0]
                        if keep.numel():
                            sub = buffer[keep]
                            sub.copy_(torch.roll(sub, shifts=-d, dims=self.dim))
                            sub[..., -d:] = data[keep]
                            buffer[keep] = sub
                    else:
                        buffer.copy_(torch.roll(buffer, shifts=-d, dims=self.dim))
                        buffer[..., -d:] = data
                # 2) 对本步 reset 的 env 用当前帧填满其历史窗口
                if _just_reset:
                    ridx = _reset.nonzero(as_tuple=True)[0]
                    buffer[ridx] = data[ridx].repeat(shape)

            next_tensordict.set(out_key, buffer.clone())
        return next_tensordict


class LayupClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        
        # 1. 基础环境: 告诉 VMAS 这是一个连续动作环境
        # 即使我们想要离散逻辑，底层 VMAS 接口必须是连续的 (continuous_actions=True)
        # [P0 单组化] 全部 4 个 agent 并成一个 "agents" 组，交给同一套共享主干网络。
        # agent 顺序由 layup 场景保证: [attacker_1, attacker_2, defender_1, defender_2]
        # => 角色映射 role_ids = [0(A1), 1(A2), 2(D), 2(D)]。
        base_env_fun = lambda: VmasEnvWithState(
            scenario=self.name.lower(),
            num_envs=num_envs,
            continuous_actions=True, 
            seed=seed,
            device=device, 
            clamp_actions=True,
            group_map=MarlGroupMapType.ALL_IN_ONE_GROUP,
            **config,
        )

        # 2. 包装转换器
        def transformed_env_fun():
            env = base_env_fun()
            
            # --- 核心配置 ---
            # 连续维度 = 2 (vx, vy)
            # 离散维度 = 1 (brake_trigger)
            # 总维度 = 3 (对应 VMAS scenario 里的 action.u 的长度)
            env = TransformedEnv(env)
            env.append_transform(
                FlattenHybridAction(continuous_dim=2, discrete_dim=1, discrete_n=2)
            )
            return env

        # [投篮按键] 启用混合动作适配层（此前误返回 base_env_fun，导致动作 3 通道被
        # 直接当作 3 维连续动作交给 VMAS）
        return transformed_env_fun

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        return {"agents": [agent.name for agent in env.agents]}

    def get_env_transforms(self, env: EnvBase) -> List[Transform]:
        """
        [冒烟实验] 为无记忆的 MLP 提供固定窗口的历史观测。

        对每个 agent group 的 observation 以及全局 state, 沿最后一维拼接最近
        ``history_frames`` 帧 (每 ``history_stride`` 步采样一帧);
        episode 结束时自动重置历史。
        ``history_frames <= 0`` 时不添加任何 transform (原有 RNN 配置不受影响)。

        例: history_frames=5, history_stride=5, dt=0.1s -> 覆盖 2.0s, 维度 x5。
        """
        history_frames = int(self.config.get("history_frames", 0) or 0)
        if history_frames <= 0:
            return []

        history_stride = int(self.config.get("history_stride", 1) or 1)
        in_keys = [(group, "observation") for group in self.group_map(env).keys()]
        in_keys.append("state")
        return [
            StridedCatFrames(
                N=history_frames,
                stride=history_stride,
                dim=-1,
                in_keys=in_keys,
                padding="same",
            )
        ]

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        if "state" in env.observation_spec:
            return Composite({"state": env.full_observation_spec_unbatched["state"].clone()})

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        # [投篮按键] 与 SMACv2 约定一致：mask 放在 (group, "action_mask")；
        # 从观测 spec 中剔除其余键，只保留 mask。
        observation_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            if (group, "observation") in observation_spec.keys(True):
                del observation_spec[(group, "observation")]
        if "state" in observation_spec:
            del observation_spec["state"]
        return observation_spec

    def observation_spec(self, env: EnvBase) -> Composite:
        """
        定义 Actor 的观测空间。
        关键在于，Actor 不应该看到 Critic 的专属信息。
        """
        observation_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            if "info" in observation_spec[group]:
                del observation_spec[(group, "info")]
            # [投篮按键] mask 不属于 Actor 观测
            if (group, "action_mask") in observation_spec.keys(True):
                del observation_spec[(group, "action_mask")]
        if "state" in observation_spec:
            del observation_spec["state"]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        info_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            if (group, "observation") in info_spec.keys(True):
                 del info_spec[(group, "observation")]
            if (group, "critic_obs") in info_spec.keys(True):
                 del info_spec[(group, "critic_obs")]
            # [投篮按键] mask 不属于 info
            if (group, "action_mask") in info_spec.keys(True):
                 del info_spec[(group, "action_mask")]
        for group in self.group_map(env):
            if "info" in info_spec[group]:
                return info_spec
        else:
            return None

    def action_spec(self, env: EnvBase) -> Composite:
        # 必须返回 env.action_spec
        # 因为我们的 Transform 已经在这里修改了 spec，这里返回的就是混合 spec
        return env.full_action_spec_unbatched

    @staticmethod
    def env_name() -> str:
        return "vmas"

    # --- [渲染降分辨率] 评估视频：2x2 块平均降采样，保持帧率不变 ---
    # VMAS 默认把 4 个子场景拼成 (2*700)x(2*700) 的单帧（5.88MB/帧），
    # 评估整段视频在内存里驻留会造成 ~2GB 峰值。这里在帧进入内存前
    # 就降采样到 700x700（1.47MB/帧），再配合流式写盘。
    @staticmethod
    def render_callback(experiment, env: EnvBase, data: TensorDictBase):
        frame = TaskClass.render_callback(experiment, env, data)
        arr = (
            frame.detach().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
        )
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] % 2 == 0 and arr.shape[1] % 2 == 0:
            h, w, c = arr.shape
            arr = (
                arr.reshape(h // 2, 2, w // 2, 2, c)
                .mean(axis=(1, 3))
                .round()
                .astype(np.uint8)
            )
        return torch.from_numpy(arr) if torch.is_tensor(frame) else arr


class LayupTask(Task):
    """Enum for VMAS tasks."""

    LAYUP = None

    @staticmethod
    def associated_class():
        return LayupClass