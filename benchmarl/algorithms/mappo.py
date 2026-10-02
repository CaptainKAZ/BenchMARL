#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, List, Optional, Tuple, Type

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import (
    IndependentNormal,
    MaskedCategorical,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig

import functools
from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec, OneHotDiscreteTensorSpec, DiscreteTensorSpec
from tensordict.nn import CompositeDistribution

import torch

def install_nan_hunter(model, warning_threshold=1e4):
    """
    即插即用的数值监控工具 (NaN Hunter)。
    功能：
    1. 自动监控模型所有叶子层的输入/输出。
    2. 发现 NaN/Inf 时抛出异常，发现数值过大时打印预警。
    3. 兼容 vmap (自动跳过矢量化内部的数据依赖检查)。
    4. 健壮性：自动处理 Bool、Int 等非浮点类型。
    
    参数:
        model: 要监控的 PyTorch 模型。
        warning_threshold: 数值爆炸预警阈值，默认 1e4。
    """
    try:
        # 用于检测当前 Tensor 是否处于 vmap 矢量化运算内部
        from torch._C._functorch import is_batchedtensor
    except ImportError:
        def is_batchedtensor(t): return False

    def _check_status(data, location_name=""):
        """
        递归检查数据状态。返回: (is_critical, is_warning, message)
        """
        # 1. 处理 TensorDict
        if isinstance(data, TensorDictBase):
            for key, val in data.items(include_nested=True, leaves_only=True):
                crit, warn, msg = _check_status(val, f"{location_name}[Key: '{key}']")
                if crit or warn: return crit, warn, msg
            return False, False, None

        # 2. 处理 Tensor
        elif isinstance(data, torch.Tensor):
            # vmap 内部无法执行 .any() 或 .item() 等控制流操作，必须跳过
            if is_batchedtensor(data):
                return False, False, None
            
            # 2.1 检查致命错误 (NaN/Inf) - 仅针对浮点型
            if torch.is_floating_point(data):
                if torch.isnan(data).any() or torch.isinf(data).any():
                    return True, True, f"{location_name} 发现 NaN/Inf! [Shape={list(data.shape)}]"
            
            # 2.2 检查爆炸预警 (数值过大)
            # 排除布尔型，仅检查浮点型和整型
            if data.dtype != torch.bool:
                # 使用 abs().max() 检查是否接近爆炸
                max_val = data.abs().max().item()
                if max_val > warning_threshold:
                    return False, True, f"{location_name} 数值过大: {max_val:.2e} (超过阈值 {warning_threshold:.0e})"
            
            return False, False, None

        # 3. 处理 Tuple 或 List (常用于多输入 args)
        elif isinstance(data, (tuple, list)):
            for i, item in enumerate(data):
                crit, warn, msg = _check_status(item, f"{location_name}[Idx: {i}]")
                if crit or warn: return crit, warn, msg
        
        return False, False, None

    def _print_stats(data, prefix=""):
        """安全打印数据的统计信息"""
        if isinstance(data, torch.Tensor):
            if is_batchedtensor(data):
                print(f"{prefix} Tensor {data.shape}: <vmap 内部数据，无法计算统计>")
            elif data.dtype == torch.bool:
                print(f"{prefix} Tensor {data.shape} [Bool]: True数量={data.sum().item()}")
            elif torch.is_floating_point(data):
                # 打印浮点数统计，包含标准差以观察分布
                print(f"{prefix} Tensor {data.shape}: Min={data.min().item():.4f}, Max={data.max().item():.4f}, Mean={data.mean().item():.4f}, Std={data.std().item():.4f}")
            else:
                print(f"{prefix} Tensor {data.shape} [{data.dtype}]: Min={data.min().item()}, Max={data.max().item()}")
        
        elif isinstance(data, TensorDictBase):
            for key, val in data.items(include_nested=True, leaves_only=True):
                if isinstance(val, torch.Tensor):
                    _print_stats(val, prefix=f"{prefix} -> '{key}':")
        
        elif isinstance(data, (tuple, list)):
            for i, item in enumerate(data):
                _print_stats(item, prefix=f"{prefix}[Idx: {i}]")

    def _hook(module, args, output):
        # 同时检查输入和输出
        in_crit, in_warn, in_msg = _check_status(args, "Input")
        out_crit, out_warn, out_msg = _check_status(output, "Output")

        if in_warn or out_warn or in_crit or out_crit:
            print(f"\n{'!'*20} 数值异常预警 {'!'*20}")
            print(f"层: {module}")
            print(f"类型: {type(module).__name__}")
            
            if in_msg: print(f"【输入异常】: {in_msg}")
            if out_msg: print(f"【输出异常】: {out_msg}")
            
            print("\n--- 🕵️‍♂️ 现场数据回溯 ---")
            print("1. 输入 (Input Args):")
            _print_stats(args, prefix="   ")
            print("\n2. 输出 (Output):")
            _print_stats(output, prefix="   ")
            
            # 只有发现真实的 NaN/Inf 时才抛出异常停止程序
            if in_crit or out_crit:
                raise RuntimeError(f"🚨 发现 NaN/Inf，为防止破坏 Checkpoint，程序已强制停止。")
            else:
                print(f"{'!'*50}\n")

    # 递归注册到所有实际执行计算的叶子层
    print(f"🕵️‍♂️ NaN Hunter 启动中... (预警阈值: {warning_threshold:.0e})")
    counter = 0
    for name, layer in model.named_modules():
        # 只挂载叶子节点，避免在容器层（如 Sequential）重复触发
        if len(list(layer.children())) == 0: 
            layer.register_forward_hook(_hook)
            counter += 1
    print(f"✅ 监控已就绪，已成功挂载 {counter} 个计算层。")

class Mappo(Algorithm):
    """Multi Agent PPO (from `https://arxiv.org/abs/2103.01955 <https://arxiv.org/abs/2103.01955>`__).

    Args:
        share_param_critic (bool): Whether to share the parameters of the critics withing agent groups
        clip_epsilon (scalar): weight clipping threshold in the clipped PPO loss equation.
        entropy_coef (scalar): entropy multiplier when computing the total loss.
        critic_coef (scalar): critic loss multiplier when computing the total
        loss_critic_type (str): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1".
        lmbda (float): The GAE lambda
        scale_mapping (str): positive mapping function to be used with the std.
            choices: "softplus", "exp", "relu", "biased_softplus_1";
        use_tanh_normal (bool): if ``True``, use TanhNormal as the continuyous action distribution with support bound
            to the action domain. Otherwise, an IndependentNormal is used.
        minibatch_advantage (bool): if ``True``, advantage computation is perfomend on minibatches of size
            ``experiment.config.on_policy_minibatch_size`` instead of the full
            ``experiment.config.on_policy_collected_frames_per_batch``, this helps not exploding memory usage

    """

    def __init__(
        self,
        share_param_critic: bool,
        clip_epsilon: float,
        entropy_coef: bool,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        scale_mapping: str,
        use_tanh_normal: bool,
        minibatch_advantage: bool,
        share_param_actor: bool,
        normalize_advantage: bool = False,
        normalize_advantage_exclude_dims: Optional[List[int]] = None,
        bounded_tanh_params: bool = True,
        loc_bound: float = 3.0,
        scale_min: float = 0.01,
        scale_max: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.share_param_critic = share_param_critic
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.loss_critic_type = loss_critic_type
        self.lmbda = lmbda
        self.scale_mapping = scale_mapping
        self.use_tanh_normal = use_tanh_normal
        self.minibatch_advantage = minibatch_advantage
        self.share_param_actor = share_param_actor
        self.normalize_advantage = normalize_advantage
        self.normalize_advantage_exclude_dims = normalize_advantage_exclude_dims
        # [稳定化] 有界策略头参数（见 BoundedNormalParamExtractor）
        self.bounded_tanh_params = bounded_tanh_params
        self.loc_bound = loc_bound
        self.scale_min = scale_min
        self.scale_max = scale_max

    #############################
    # Overridden abstract methods
    #############################

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        # Loss
        loss_module = ClipPPOLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            # [修复] 参数名曾误写成 entropy_coeff/critic_coeff（双 f）。torchrl 0.8.x 的
            # ClipPPOLoss 只有 entropy_coef/critic_coef（单 f）+ 末尾 **kwargs，而
            # PPOLoss.__init__ 调 super().__init__() 时不转发 kwargs -> 未知关键字被静默
            # 丢弃，系数恒为默认值 (entropy_coef=0.01, critic_coef=1.0)，yaml 里的数值
            # 形同虚设。此处改为单 f 后，yaml 中的值才真正生效。
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            # [稳定化] 默认对 advantage 做标准化。
            # advantage 量级实测 std 14~35, |adv|>10 占比 23%~64%, 与 grad clip=2.0 严重不匹配。
            # 多智能体场景必须传 exclude_dims, 否则会跨 agent 混合统计 (torchrl 官方建议),
            # [-2] 表示 agent 维保持独立 -> 每个 agent 各自标准化, 避免 A1 的弱信号被 A2 淹没。
            normalize_advantage=self.normalize_advantage,
            normalize_advantage_exclude_dims=(
                tuple(self.normalize_advantage_exclude_dims)
                if self.normalize_advantage_exclude_dims
                else ()
            ),
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        return loss_module, False

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
        return {
            "loss_objective": list(loss.actor_network_params.flatten_keys().values()),
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
        }
    
    def _check_specs(self) -> None:
        """
        [关键修复] 覆盖父类检查。
        允许 'action' 为 CompositeSpec (混合动作容器)，而不是强制要求为叶子节点。
        """
        for group in self.group_map.keys():
            try:
                group_spec = self.action_spec[group]
            except KeyError:
                raise ValueError(f"Action spec for group '{group}' is missing.")

            if "action" not in group_spec.keys():
                raise ValueError(
                    f"Action spec for group '{group}' must contain an entry named 'action'."
                )

            action_spec = group_spec["action"]
            
            # 如果是混合动作容器，直接通过
            if isinstance(action_spec, CompositeSpec):
                continue 
            
            # 兼容旧逻辑：如果是普通 Spec，检查是否有 shape
            if not hasattr(action_spec, "shape"):
                 raise ValueError(f"Action spec for group '{group}' is invalid.")

    def _get_composite_policy(self, group, model_config, n_agents, action_spec):
        import functools
        import torch  # [关键] 引入 torch 以使用 compiler 装饰器
        from torch import distributions as d
        from tensordict import TensorDict
        from tensordict.nn import CompositeDistribution, TensorDictSequential, TensorDictModule
        from torchrl.data import CompositeSpec
        from torchrl.modules import IndependentNormal, TanhNormal, MaskedCategorical

        # --- 1. 优化后的 Wrapper (带 Torch Compile Fix) ---

        # [关键修复] 添加 @torch.compiler.disable
        # 这个装饰器告诉 PyTorch Dynamo 不要尝试编译/追踪这个类及其方法。
        # 这解决了由于 Categorical 分布惰性属性 (lazy properties) 和 __getattr__ 代理
        # 导致的 "AssertionError: Guard check failed" 问题。
        @torch.compiler.disable
        class SummedDistributionWrapper(d.Distribution):
            def __init__(self, dist, group_name):
                self.dist = dist
                self.group_name = group_name
                
            def log_prob(self, value):
                # 1. 重构层级结构 (Hierarchy Reconstruction)
                if isinstance(value, TensorDict):
                    # 保持 batch_size 和 device 一致
                    root = TensorDict({}, batch_size=value.batch_size, device=value.device)
                    # 嵌套挂载：root -> group -> action -> values
                    root[self.group_name] = TensorDict({"action": value}, batch_size=value.batch_size, device=value.device)
                    lp = self.dist.log_prob(root)
                else:
                    lp = self.dist.log_prob(value)

                # 2. 扁平化高效求和 (Flattened Sum)
                if isinstance(lp, (dict, TensorDict)):
                    # include_nested=True: 穿透所有层级
                    # leaves_only=True: 只返回 Tensor
                    leaves = list(lp.values(include_nested=True, leaves_only=True))
                    
                    if not leaves:
                        return 0.0
                    
                    return sum(leaves)
                
                return lp

            # 透传 entropy
            def entropy(self):
                ent = self.dist.entropy()
                if isinstance(ent, (dict, TensorDict)):
                    leaves = ent.values(include_nested=True, leaves_only=True)
                    return sum(leaves)
                return ent
            
            # 透传 sample
            def sample(self, *args, **kwargs):
                return self.dist.sample(*args, **kwargs)
                
            @property
            def mode(self):
                return self.dist.mode
            
            def __getattr__(self, name):
                return getattr(self.dist, name)

        class CompositePolicy(TensorDictSequential):
            def __init__(self, prob_actor, summer, group_name):
                super().__init__(prob_actor, summer)
                self.prob_actor = prob_actor
                self.group_name = group_name
                
            def get_dist(self, td, **kwargs):
                if self.prob_actor:
                    dist = self.prob_actor.get_dist(td, **kwargs)
                    return SummedDistributionWrapper(dist, self.group_name)
                raise RuntimeError("ProbabilisticActor not found in CompositePolicy sequence")

        # --- 2. 配置准备 ---

        distribution_map = {}
        name_map = {} 
        modules = []
        flat_action_spec_dict = {}

        total_param_dim = 0
        split_sizes = []
        split_names = []
        
        log_prob_keys = []
        
        for name, sub_spec in action_spec.items():
            # 全局路径 (用于 Sampling)
            full_action_key = (group, "action", name)
            name_map[name] = full_action_key
            flat_action_spec_dict[full_action_key] = sub_spec
            
            # log_prob 路径 (用于 Collection Summer)
            # TorchRL 默认规则: action_key + "_log_prob"
            lp_key = (group, "action", name + "_log_prob")
            log_prob_keys.append(lp_key)

            if isinstance(sub_spec, (BoundedTensorSpec, UnboundedContinuousTensorSpec)):
                dim = sub_spec.shape[-1] * 2 
                distribution_map[name] = IndependentNormal if not self.use_tanh_normal else SafeTanhNormal
            else:
                dim = sub_spec.space.n
                distribution_map[name] = Categorical if self.action_mask_spec is None else MaskedCategorical
            
            total_param_dim += dim
            split_sizes.append(dim)
            split_names.append(name)

        # --- 3. 模块构建 ---

        # (A) Base Model
        actor_input_spec = Composite({group: self.observation_spec[group].clone().to(self.device)})
        actor_output_spec = Composite({group: Composite({"logits": Unbounded(shape=(n_agents, total_param_dim))}, shape=(n_agents,))})
        
        base_model = model_config.get_model(
            input_spec=actor_input_spec, output_spec=actor_output_spec, 
            agent_group=group, input_has_agent_dim=True, n_agents=n_agents, 
            centralised=False, share_params=self.share_param_actor, 
            device=self.device, action_spec=self.action_spec
        )
        modules.append(base_model)

        # (B) Splitter
        split_keys = [f"{name}_raw_params" for name in split_names]
        modules.append(TensorDictModule(
            lambda x: torch.split(x, split_sizes, dim=-1),
            in_keys=[(group, "logits")],
            out_keys=split_keys
        ))

        # (C) Param Extractor
        for name, sub_spec in action_spec.items():
            raw_key = f"{name}_raw_params"
            if isinstance(sub_spec, (BoundedTensorSpec, UnboundedContinuousTensorSpec)):
                loc_key = (group, "params", name, "loc")
                scale_key = (group, "params", name, "scale")
                modules.append(TensorDictModule(
                    (
                        BoundedNormalParamExtractor(
                            loc_bound=self.loc_bound,
                            scale_min=self.scale_min,
                            scale_max=self.scale_max,
                        )
                        if (self.use_tanh_normal and self.bounded_tanh_params)
                        else NormalParamExtractor(scale_mapping=self.scale_mapping)
                    ),
                    in_keys=[raw_key],
                    out_keys=[loc_key, scale_key]
                ))
            else:
                logits_key = (group, "params", name, "logits")
                modules.append(TensorDictModule(
                    lambda x: x, in_keys=[raw_key], out_keys=[logits_key]
                ))

        # --- 4. ProbabilisticActor ---

        dist_constructor = functools.partial(
            CompositeDistribution,
            distribution_map=distribution_map,
            name_map=name_map
        )
        
        check_spec = CompositeSpec(flat_action_spec_dict, shape=action_spec.shape)

        prob_actor = ProbabilisticActor(
            module=TensorDictSequential(*modules),
            spec=check_spec,
            in_keys=[(group, "params")],
            out_keys=list(name_map.values()),
            distribution_class=dist_constructor,
            return_log_prob=True,
            log_prob_keys=log_prob_keys, 
        )
        
        # --- 5. Summer (用于 Collection 阶段) ---
        
        summer = TensorDictModule(
            lambda *args: sum(args),
            in_keys=log_prob_keys,
            out_keys=[(group, "log_prob")]
        )
        
        # --- 6. 返回策略 ---
        return CompositePolicy(prob_actor, summer, group)

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        action_spec = self.action_spec[group, "action"]
        if isinstance(action_spec, CompositeSpec):
            return self._get_composite_policy(group, model_config, n_agents, action_spec)

        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]
        actor_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )

        actor_output_spec = Composite(
            {
                group: Composite(
                    {"logits": Unbounded(shape=logits_shape)},
                    shape=(n_agents,),
                )
            }
        )
        print(f"making actor model for {group}, share param {self.share_param_actor}")
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.share_param_actor,
            device=self.device,
            action_spec=self.action_spec,
            name=f"{group}_actor"
        )

        if continuous:
            if self.use_tanh_normal and self.bounded_tanh_params:
                # [稳定化] 有界策略头：loc∈(−3,3)、scale∈(0.01,1.0)，避免 tanh 饱和把 log_prob 推向 1e6 量级
                _extractor = BoundedNormalParamExtractor(
                    loc_bound=self.loc_bound,
                    scale_min=self.scale_min,
                    scale_max=self.scale_max,
                )
            else:
                _extractor = NormalParamExtractor(
                    scale_mapping=self.scale_mapping, scale_lb=1e-4
                )
            extractor_module = TensorDictModule(
                _extractor,
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            policy = ProbabilisticActor(
                module=TensorDictSequential(actor_module, extractor_module),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=(
                    IndependentNormal if not self.use_tanh_normal else SafeTanhNormal
                ),
                distribution_kwargs=(
                    {
                        "low": self.action_spec[(group, "action")].space.low,
                        "high": self.action_spec[(group, "action")].space.high,
                    }
                    if self.use_tanh_normal
                    else {}
                ),
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )

        else:
            if self.action_mask_spec is None:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "action")],
                    distribution_class=Categorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys={
                        "logits": (group, "logits"),
                        "mask": (group, "action_mask"),
                    },
                    out_keys=[(group, "action")],
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
        # policy=torch.compile(policy)
        # install_nan_hunter(policy)
        return policy

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        # MAPPO uses the same stochastic actor for collection
        return policy_for_loss

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        nested_reward_key = ("next", group, "reward")

        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )

        if nested_reward_key not in keys:
            batch.set(
                nested_reward_key,
                batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)),
            )

        loss = self.get_loss_and_updater(group)[0]
        if self.minibatch_advantage:
            increment = -(
                -self.experiment.config.train_minibatch_size(self.on_policy)
                // batch.shape[1]
            )
        else:
            increment = batch.batch_size[0] + 1
        last_start_index = 0
        start_index = increment
        adv_key = loss.tensor_keys.advantage
        vt_key = loss.tensor_keys.value_target
        adv_parts, vt_parts = [], []
        while last_start_index < batch.shape[0]:
            minimbatch = batch[last_start_index:start_index]
            with torch.no_grad():
                loss.value_estimator(
                    minimbatch,
                    params=loss.critic_network_params,
                    target_params=loss.target_critic_network_params,
                )
            # [内存优化] TensorDict 切片是副本, 只 clone 出体积很小的
            # advantage/value_target, 让切片在本轮结束后即可被释放,
            # 避免在内存中同时保留整批切片副本 (约 1 个完整 batch 的量级)。
            if adv_key is not None and adv_key in minimbatch.keys(True):
                adv_parts.append(minimbatch.get(adv_key).clone())
            if vt_key is not None and vt_key in minimbatch.keys(True):
                vt_parts.append(minimbatch.get(vt_key).clone())
            del minimbatch

            last_start_index = start_index
            start_index += increment

        if adv_parts:
            batch.set(adv_key, torch.cat(adv_parts, dim=0))
        if vt_parts:
            batch.set(vt_key, torch.cat(vt_parts, dim=0))
        return batch

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        loss_vals.set(
            "loss_objective", loss_vals["loss_objective"] + loss_vals["loss_entropy"]
        )
        del loss_vals["loss_entropy"]
        return loss_vals

    #####################
    # Custom new methods
    #####################

    def get_critic(self, group: str) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        if self.share_param_critic:
            critic_output_spec = Composite({"state_value": Unbounded(shape=(1,))})
        else:
            critic_output_spec = Composite(
                {
                    group: Composite(
                        {"state_value": Unbounded(shape=(n_agents, 1))},
                        shape=(n_agents,),
                    )
                }
            )

        if self.state_spec is not None:
            input_has_agent_dim = False
            critic_input_spec = self.state_spec

        else:
            input_has_agent_dim = True
            critic_input_spec = Composite(
                {group: self.observation_spec[group].clone().to(self.device)}
            )
        print(f"making critic model for {group}, share_param {self.share_param_critic}")
        value_module = self.critic_model_config.get_model(
            input_spec=critic_input_spec,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=True,
            input_has_agent_dim=input_has_agent_dim,
            agent_group=group,
            share_params=self.share_param_critic,
            device=self.device,
            action_spec=self.action_spec,
            name=f"{group}_critic"
        )

        if self.share_param_critic:
            expand_module = TensorDictModule(
                lambda value: value.unsqueeze(-2).expand(
                    *value.shape[:-1], n_agents, 1
                ),
                in_keys=["state_value"],
                out_keys=[(group, "state_value")],
            )
            value_module = TensorDictSequential(value_module, expand_module)
        # [性能] critic 的 torch.compile 用环境变量控制，便于 A/B：
        #   CRITIC_COMPILE_MODE = off | default | reduce-overhead | max-autotune-no-cudagraphs
        # 实测（3 轮冒烟，/tmp/opencode 日志）：off 的 opt_loops 更小更稳（attacker 28.7-29.1s，
        # 编译时 28.9-35.4s 抖动很大），故默认关闭；需要时用环境变量打开。
        import os as _os
        critic_compile_mode = _os.environ.get("CRITIC_COMPILE_MODE", "off")
        if critic_compile_mode != "off":
            value_module = torch.compile(value_module, mode=critic_compile_mode)
        return value_module


# ---------------------------------------------------------------------------
# [稳定化] 有界参数头 + 安全 TanhNormal
#   v27 崩溃回溯：策略头输出 loc 漂移到 ±30、scale 到 90，动作长期贴在 tanh 边界，
#   训练侧反解 log_prob 在饱和点得到 ~1e6 量级的值 → importance ratio 变 0/∞ → 0×∞ → NaN。
#   下面两个类分别从"分布参数范围"和"log_prob 数值"两侧封死这条路径。
# ---------------------------------------------------------------------------


class BoundedNormalParamExtractor(torch.nn.Module):
    """把网络输出 (..., 2D) 映射为有界的 loc / scale。

    - loc   = loc_bound * tanh(raw_loc / loc_bound)                     ∈ (−loc_bound, loc_bound)
    - scale = scale_min + (scale_max − scale_min) * sigmoid(raw_scale)  ∈ (scale_min, scale_max)

    采样是 upscale·tanh(z)：只要 |loc| 有界、scale 有下限，动作就贴不到 ±upscale 的饱和点，
    log_prob 也始终有限（scale 有下限、loc 有界 ⇒ 上界可控）。
    """

    def __init__(
        self,
        loc_bound: float = 3.0,
        scale_min: float = 0.01,
        scale_max: float = 1.0,
    ):
        super().__init__()
        self.loc_bound = float(loc_bound)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

    def forward(self, logits: torch.Tensor):
        loc_raw, scale_raw = logits.chunk(2, dim=-1)
        loc = self.loc_bound * torch.tanh(loc_raw / self.loc_bound)
        scale = self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(
            scale_raw
        )
        return loc, scale


class SafeTanhNormal(TanhNormal):
    """log_prob 前把动作夹进开区间 (low, high)，避免 atanh 在饱和点反解出 ±inf/巨大值。"""

    def log_prob(self, value):
        low = torch.as_tensor(self.low, device=value.device, dtype=value.dtype)
        high = torch.as_tensor(self.high, device=value.device, dtype=value.dtype)
        margin = (high - low) * 1e-3
        value = value.clamp(min=low + margin, max=high - margin)
        return super().log_prob(value)


@dataclass
class MappoConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Mappo`."""

    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING
    minibatch_advantage: bool = MISSING
    share_param_actor: bool = MISSING
    # [稳定化] advantage 标准化开关; exclude_dims 指定保持独立统计的维度 ([-2] = agent 维)
    normalize_advantage: bool = MISSING
    normalize_advantage_exclude_dims: Optional[List[int]] = None
    # [稳定化] 有界策略头 (仅 use_tanh_normal 时生效): loc 用 tanh 限幅、scale 用 sigmoid 限幅
    bounded_tanh_params: bool = True
    loc_bound: float = 3.0
    scale_min: float = 0.01
    scale_max: float = 1.0

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Mappo

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def has_centralized_critic() -> bool:
        return True
