#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import copy
from typing import Callable, Dict, List, Optional

import torch
import vmas
from tensordict import TensorDictBase
from torchrl.data import Composite, CompositeSpec, DiscreteTensorSpec, UnboundedContinuousTensorSpec
from torchrl.data.tensor_specs import Unbounded
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.transforms import Transform
from torchrl.envs.libs.vmas import VmasEnv

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
        print(f"[DEBUG] FlattenHybridAction initialized. C={continuous_dim}, D={discrete_dim}")

    # [关键修复] forward 是处理 Observation 的，动作处理必须在 _inv_call 中！
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # [DEBUG] 打印 tensordict 结构
        if not self._debug_printed:
            print(f"\n[DEBUG] FlattenHybridAction._inv_call (ACTION processing) CALLED.")
            # 简略打印 Keys 确认数据流
            root_keys = list(tensordict.keys())
            print(f"[DEBUG] Root keys: {root_keys}")
            for k in root_keys:
                if k in ["attacker", "defender", "agents"]: # 打印 Agent Group 的内容
                    item = tensordict.get(k)
                    if isinstance(item, TensorDictBase) and "action" in item.keys():
                        act = item.get("action")
                        if isinstance(act, TensorDictBase):
                            print(f"[DEBUG]   Group '{k}' action keys: {list(act.keys())}")
            self._debug_printed = True

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

    def transform_input_spec(self, input_spec: CompositeSpec) -> CompositeSpec:
        """
        Step 2: 欺骗 Mappo (保留 Batch 维度)
        """
        if "full_action_spec" not in input_spec.keys():
            return input_spec
            
        full_action_spec = input_spec["full_action_spec"]
        
        for group_key in list(full_action_spec.keys()):
            group_spec = full_action_spec[group_key]
            if "action" not in group_spec.keys(): continue
                
            original_spec = group_spec["action"]
            if isinstance(original_spec, CompositeSpec): continue

            # 保留 Batch 维度
            target_shape = original_spec.shape[:-1]
            
            new_spec = CompositeSpec(shape=target_shape, device=original_spec.device)
            new_spec["continuous"] = UnboundedContinuousTensorSpec(
                shape=target_shape + (self.continuous_dim,),
                device=original_spec.device,
                dtype=torch.float32
            )
            new_spec["discrete"] = DiscreteTensorSpec(
                n=self.discrete_n,
                shape=target_shape + (self.discrete_dim,),
                device=original_spec.device,
                dtype=torch.long
            )
            full_action_spec[group_key]["action"] = new_spec

        return input_spec
    
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
        self.observation_spec_unbatched = observation_spec_unbatched

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        tensordict_out = super()._reset(tensordict, **kwargs)
        state = self._env.scenario.get_global_state()
        tensordict_out.set("state", state)
        return tensordict_out

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        tensordict_out = super()._step(tensordict)
        next_state = self._env.scenario.get_global_state()
        tensordict_out.set("state", next_state)
        return tensordict_out


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
        base_env_fun = lambda: VmasEnvWithState(
            scenario=self.name.lower(),
            num_envs=num_envs,
            continuous_actions=True, 
            seed=seed,
            device=device, 
            clamp_actions=True,
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

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        if "state" in env.observation_spec:
            return Composite({"state": env.full_observation_spec_unbatched["state"].clone()})

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        """
        定义 Actor 的观测空间。
        关键在于，Actor 不应该看到 Critic 的专属信息。
        """
        observation_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            if "info" in observation_spec[group]:
                del observation_spec[(group, "info")]
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


class LayupTask(Task):
    """Enum for VMAS tasks."""

    LAYUP = None

    @staticmethod
    def associated_class():
        return LayupClass