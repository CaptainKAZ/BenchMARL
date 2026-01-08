#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Tuple, Type

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
from tensordict import TensorDictBase

def install_nan_hunter(model):
    """
    给模型的所有子层安装 NaN 监控钩子 (支持 TensorDict 和 Nested Structure)。
    无需 autograd，在 inference/rollout 模式下完全有效。
    """
    
    def _check_nan(data, location_name=""):
        """
        递归检查 data 中是否包含 NaN/Inf。
        返回: (has_nan, details_string)
        """
        # 1. 如果是 TensorDict
        if isinstance(data, TensorDictBase):
            # 遍历所有叶子节点 (leaves_only=True 会自动递归嵌套的 TensorDict)
            for key, val in data.items(include_nested=True, leaves_only=True):
                if isinstance(val, torch.Tensor):
                    if torch.isnan(val).any() or torch.isinf(val).any():
                        return True, f"{location_name}[TensorDict Key: '{key}']"
            return False, None

        # 2. 如果是 Tensor
        elif isinstance(data, torch.Tensor):
            if torch.isnan(data).any() or torch.isinf(data).any():
                return True, f"{location_name}[Tensor shape={data.shape}]"
            return False, None

        # 3. 如果是 Tuple 或 List
        elif isinstance(data, (tuple, list)):
            for i, item in enumerate(data):
                found, msg = _check_nan(item, f"{location_name}[Seq index: {i}]")
                if found:
                    return True, msg
            return False, None

        # 4. 其他类型忽略 (如 None, int, str)
        return False, None

    def _print_stats(data, prefix=""):
        """辅助函数：打印数据的统计信息"""
        if isinstance(data, torch.Tensor):
             print(f"{prefix} Tensor {data.shape}: Min={data.min():.4f}, Max={data.max():.4f}, Mean={data.mean():.4f}, HasNaN={torch.isnan(data).any()}")
        elif isinstance(data, TensorDictBase):
            print(f"{prefix} TensorDict Keys: {data.keys(include_nested=True)}")
            # 简单打印第一个 key 的状态作为示例，防止刷屏
            for key, val in data.items(include_nested=True, leaves_only=True):
                 if isinstance(val, torch.Tensor):
                    print(f"{prefix}   -> Key '{key}': Min={val.min():.4f}, Max={val.max():.4f}, HasNaN={torch.isnan(val).any()}")

    def _hook(module, args, output):
        # args 是输入 (tuple), output 是输出 (可以是 Tensor, TensorDict, Tuple 等)
        
        # --- 1. 检查输入 (Input) ---
        # args 永远是一个 tuple，比如 (tensordict, ) 或者 (tensor_a, tensor_b)
        for i, arg in enumerate(args):
            has_nan, loc = _check_nan(arg, location_name=f"Input arg {i}")
            if has_nan:
                # 发现输入就有 NaN，通常意味着上一层或者是数据源的问题
                # 我们可以选择忽略，或者打印警告
                # print(f"⚠️ Warning: Layer {type(module).__name__} received NaN at {loc}")
                pass 

        # --- 2. 检查输出 (Output) ---
        has_nan_out, loc_out = _check_nan(output, location_name="Output")

        if has_nan_out:
            print(f"\n{'='*60}")
            print(f"🚨 抓到了！NaN 产生于层: {module}")
            print(f"   类型: {type(module).__name__}")
            print(f"   具体位置: {loc_out}")
            print(f"{'='*60}")
            
            print("\n--- 🕵️‍♂️ 现场数据分析 ---")
            
            print("1. 输入数据统计:")
            for i, arg in enumerate(args):
                _print_stats(arg, prefix=f"Arg[{i}]:")

            print("\n2. 输出数据统计:")
            _print_stats(output, prefix="Output:")
            
            # 抛出异常，暂停程序
            raise RuntimeError(f"NaN detected in forward pass of {type(module).__name__}")

    # 递归注册到所有子模块
    print(f"🕵️‍♂️ NaN Hunter (TensorDict版) 正在启动...")
    for name, layer in model.named_modules():
        # 我们可以跳过一些不进行计算的容器层，比如 Sequential，只监控实际的叶子层
        # 但为了保险，监控所有层也行，除了本身就是容器的
        if len(list(layer.children())) == 0: 
            # print(f"  -> 监控层：{name} ({type(layer).__name__})")
            layer.register_forward_hook(_hook)
    print("✅ 监控已就绪。")

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
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            # normalize_advantage=True,
            # normalize_advantage_exclude_dims=[1]
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
                distribution_map[name] = IndependentNormal if not self.use_tanh_normal else TanhNormal
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
            centralised=False, share_params=self.experiment_config.share_policy_params, 
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
                    NormalParamExtractor(scale_mapping=self.scale_mapping),
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
        print(f"making actor model for {group}")
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
        )
        print(actor_module)

        if continuous:
            extractor_module = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping, scale_lb=1e-4),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            policy = ProbabilisticActor(
                module=TensorDictSequential(actor_module, extractor_module),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=(
                    IndependentNormal if not self.use_tanh_normal else TanhNormal
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
        minibatches = []
        while last_start_index < batch.shape[0]:
            minimbatch = batch[last_start_index:start_index]
            minibatches.append(minimbatch)
            with torch.no_grad():
                loss.value_estimator(
                    minimbatch,
                    params=loss.critic_network_params,
                    target_params=loss.target_critic_network_params,
                )
            last_start_index = start_index
            start_index += increment

        batch = torch.cat(minibatches, dim=0)
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
        print(f"making critic model for {group}")
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
        )
        print(value_module)
        if self.share_param_critic:
            expand_module = TensorDictModule(
                lambda value: value.unsqueeze(-2).expand(
                    *value.shape[:-1], n_agents, 1
                ),
                in_keys=["state_value"],
                out_keys=[(group, "state_value")],
            )
            value_module = TensorDictSequential(value_module, expand_module)
        value_module = torch.compile(value_module)
        return value_module


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
