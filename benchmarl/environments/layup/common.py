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
            
        # 基础 state spec，形状为 [21]
        state_dim_shape = sample_state.shape[1:]
        unbatched_state_spec_value = Unbounded(
            shape=state_dim_shape,
            device=self.device,
            dtype=sample_state.dtype,
        )

        for group in self.group_map.keys():
            # group_shape 是 [n_agents_in_group], e.g., torch.Size([2])
            group_shape = self.full_observation_spec_unbatched[group].shape

            # --- 关键修正 ---
            # 1. 通过拼接，构建出正确的目标形状
            #    e.g., torch.Size([2]) + torch.Size([21]) -> torch.Size([2, 21])
            target_shape = group_shape + state_dim_shape

            # 2. 将基础 spec 扩展到这个新的、正确的二维形状
            expanded_spec = unbatched_state_spec_value.expand(target_shape)
            
            self.full_observation_spec_unbatched[group, "state"] = expanded_spec

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        """在 reset 返回的 tensordict 中为每个 group 填充初始全局状态。"""
        tensordict_out = super()._reset(tensordict, **kwargs)
        
        # 获取形状为 [num_envs, state_dim] 的共享状态
        state = self._env.scenario.get_global_state()
        
        # 将共享状态扩展并添加到每个 agent group
        for group, agent_names in self.group_map.items():
            n_agents_in_group = len(agent_names)
            # 扩展后形状: [num_envs, n_agents_in_group, state_dim]
            expanded_state = state.unsqueeze(1).expand(-1, n_agents_in_group, -1)
            tensordict_out.set((group, "state"), expanded_state)
            
        return tensordict_out

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        """在 step 返回的 tensordict 中为每个 group 填充下一个全局状态。"""
        tensordict_out = super()._step(tensordict)
        
        # 获取形状为 [num_envs, state_dim] 的 next_state
        next_state = self._env.scenario.get_global_state()
        
        # 将 next_state 扩展并添加到每个 agent group 的 "next" 子字典中
        for group, agent_names in self.group_map.items():
            n_agents_in_group = len(agent_names)
            expanded_next_state = next_state.unsqueeze(1).expand(-1, n_agents_in_group, -1)
            tensordict_out.set((group, "next", "state"), expanded_next_state)
            
        return tensordict_out


class VmasEnvWithCriticState(VmasEnv):
    """
    一个自定义 VmasEnv 封装，为环境添加了每个智能体专属的 critic_obs 支持。

    这个封装假设底层的 vmas scenario 实现了一个名为 `get_all_critic_obs()` 的
    向量化方法，该方法一次性返回所有智能体的 critic 观测矩阵，形状为
    [B, N, D_critic_obs]，从而获得最佳性能。
    """

    def _make_specs(
        self, env: vmas.simulator.environment.environment.Environment
    ) -> None:
        # 关键经验：所有对spec的修改，都必须在带批处理维度的 `self.full_observation_spec` 上进行才能生效。
        super()._make_specs(env)

        if not hasattr(self._env.scenario, "get_all_critic_obs"):
            raise AttributeError(
                "The environment's scenario must have a 'get_all_critic_obs()' method."
            )
        sample_critic_obs_matrix = self._env.scenario.get_all_critic_obs()
        
        unbatched_shape = sample_critic_obs_matrix.shape[1:]
        unbatched_critic_obs_spec = Unbounded(
            shape=unbatched_shape,
            device=self.device,
            dtype=sample_critic_obs_matrix.dtype,
        )
        
        batched_critic_obs_spec = unbatched_critic_obs_spec.expand(
            self.batch_size + unbatched_critic_obs_spec.shape
        )
        
        group = next(iter(self.group_map.keys()))
        
        group_spec = self.full_observation_spec[group]
        group_spec["critic_obs"] = batched_critic_obs_spec
        self.full_observation_spec[group] = group_spec

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        """在 reset 返回的 tensordict 中填充初始的 critic_obs 矩阵。"""
        tensordict_out = super()._reset(tensordict, **kwargs)

        # 直接进行一次向量化调用，获取 [B, N, D] 的完整矩阵
        all_critic_obs = self._env.scenario.get_all_critic_obs()
        group = next(iter(self.group_map.keys()))

        tensordict_out.set((group, "critic_obs"), all_critic_obs)

        return tensordict_out

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        """在 step 返回的 tensordict 中填充下一个 critic_obs 矩阵。"""
        tensordict_out = super()._step(tensordict)

        # 直接进行一次向量化调用，获取 [B, N, D] 的完整矩阵
        next_all_critic_obs = self._env.scenario.get_all_critic_obs()
        group = next(iter(self.group_map.keys()))

        tensordict_out.set(("next", group, "critic_obs"), next_all_critic_obs)

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
            categorical_actions=True,
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
        """
        为中心化 Critic 定义 state spec。
        我们可以直接复用 critic_obs 作为 state，因为它包含了全局信息。
        BenchMARL 的 CTDE 算法期望的 key 是 "state"。
        """
        any_group = next(iter(self.group_map(env).keys()))
        if (any_group, "critic_obs") in env.full_observation_spec_unbatched.keys(True):
            # 获取原始的 critic_obs spec
            raw_spec = env.full_observation_spec_unbatched[any_group, "critic_obs"]
            # 将其包装在 "state" key 下返回
            return Composite({"state": raw_spec})
        return None

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
            # # 从 Actor 的观测空间中移除 critic_obs
            # if "critic_obs" in observation_spec[group]:
            #     del observation_spec[(group, "critic_obs")]
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