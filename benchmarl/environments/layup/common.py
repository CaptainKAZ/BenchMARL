#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import copy
from typing import Callable, Dict, List, Optional

import vmas
from tensordict import TensorDictBase
from torchrl.data import Composite
from torchrl.data.tensor_specs import Unbounded
from torchrl.envs import EnvBase
from torchrl.envs.libs.vmas import VmasEnv

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

class VmasEnvWithState(VmasEnv):
    """
    一个自定义 VmasEnv 封装，为环境添加了全局状态 (state) 支持。

    它将共享的全局状态注入到每个智能体组的观测空间中，以兼容
    BenchMARL 中 CTDE 算法的按组处理机制。
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
        # 基础 state spec，形状为 [21]
        print(f"sample state is {sample_state.shape}")
        state_dim_shape = sample_state.shape[1:]
        unbatched_state_spec_value = Unbounded(
            shape=state_dim_shape,
            device=self.device,
            dtype=sample_state.dtype,
        )
        
            
        full_state_spec_unbatched["state"] = unbatched_state_spec_value
        self.full_state_spec_unbatched = full_state_spec_unbatched
        observation_spec_unbatched = self.observation_spec_unbatched
        observation_spec_unbatched["state"] = unbatched_state_spec_value
        self.observation_spec_unbatched = observation_spec_unbatched
        # print(self.observation_spec)
        # print(self.state_spec)
            

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        """在 reset 返回的 tensordict 中为每个 group 填充初始全局状态。"""
        tensordict_out = super()._reset(tensordict, **kwargs)
        
        # 获取形状为 [num_envs, state_dim] 的共享状态
        state = self._env.scenario.get_global_state()

        tensordict_out.set("state", state)
            
        return tensordict_out

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        """在 step 返回的 tensordict 中为每个 group 填充下一个全局状态。"""
        tensordict_out = super()._step(tensordict)
        # 获取形状为 [num_envs, state_dim] 的 next_state
        next_state = self._env.scenario.get_global_state()
        
        tensordict_out.set(("state"), next_state)
            
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
        return lambda: VmasEnvWithState(
            scenario=self.name.lower(),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device, 
            clamp_actions=True,
            
            **config,
        )

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