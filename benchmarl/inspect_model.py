#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""
详细的模型检查工具
用于验证 MAPPO 网络的内部机制：centralized、input_has_agent_dim、share_param 等
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from typing import Dict, List, Any
import warnings


class ModelInspector:
    """
    详细检查 MAPPO 模型的内部机制

    功能：
    1. 打印模型的详细配置参数
    2. 使用实际数据跟踪前向传播
    3. 打印每一步的张量形状和数据流
    4. 验证参数共享情况
    5. 检查 attention 的具体实现
    """

    def __init__(self):
        self.indent_level = 0

    def print_section(self, title: str, level: int = 1):
        """打印分节标题"""
        if level == 1:
            print("\n" + "=" * 80)
            print(f"  {title}")
            print("=" * 80)
        elif level == 2:
            print("\n" + "-" * 80)
            print(f"  {title}")
            print("-" * 80)
        else:
            print(f"\n{'  ' * (level - 2)}[{title}]")

    def print_indent(self, text: str, extra_indent: int = 0):
        """缩进打印"""
        indent = "  " * (self.indent_level + extra_indent)
        print(f"{indent}{text}")

    def inspect_model_config(self, model, model_name: str):
        """检查模型的配置参数"""
        self.print_section(f"模型配置: {model_name}", level=1)

        # 检查关键属性
        attrs_to_check = [
            'n_agents',
            'centralised',
            'share_params',
            'input_has_agent_dim',
            'is_critic',
            'agent_group',
            'device',
        ]

        for attr in attrs_to_check:
            if hasattr(model, attr):
                value = getattr(model, attr)
                self.print_indent(f"✓ {attr}: {value}")
            else:
                self.print_indent(f"✗ {attr}: NOT FOUND")

        # 检查输入输出规格
        if hasattr(model, 'input_spec'):
            self.print_indent(f"\n输入规格:")
            self.print_indent(f"  {model.input_spec}", extra_indent=1)

        if hasattr(model, 'output_spec'):
            self.print_indent(f"\n输出规格:")
            self.print_indent(f"  {model.output_spec}", extra_indent=1)

        # 检查输入输出键
        if hasattr(model, 'in_keys'):
            self.print_indent(f"\n输入键: {model.in_keys}")
        if hasattr(model, 'out_keys'):
            self.print_indent(f"输出键: {model.out_keys}")

    def trace_forward_pass(self, model, input_data: TensorDictBase, model_name: str):
        """跟踪前向传播，打印每一步的形状"""
        self.print_section(f"前向传播跟踪: {model_name}", level=1)

        # 打印输入数据
        self.print_indent("输入数据:")
        self._print_tensordict(input_data, indent=1)

        # 如果是 SequenceModel，逐层跟踪
        from benchmarl.models.common import SequenceModel

        if isinstance(model, SequenceModel):
            self.print_indent("\n检测到 SequenceModel，逐层跟踪...")

            current_td = input_data.clone()

            if hasattr(model, 'models') and hasattr(model.models, 'module'):
                submodules = list(model.models.module)

                for idx, submodule in enumerate(submodules):
                    self.print_section(f"Layer {idx}: {submodule.__class__.__name__}", level=3)

                    # 检查子模块配置
                    if hasattr(submodule, 'centralised'):
                        self.print_indent(f"  centralised: {submodule.centralised}")
                    if hasattr(submodule, 'input_has_agent_dim'):
                        self.print_indent(f"  input_has_agent_dim: {submodule.input_has_agent_dim}")
                    if hasattr(submodule, 'share_params'):
                        self.print_indent(f"  share_params: {submodule.share_params}")

                    # 打印输入
                    self.print_indent(f"\n  输入:")
                    self._print_tensordict(current_td, indent=2)

                    try:
                        # 执行前向传播
                        with torch.no_grad():
                            current_td = submodule(current_td)

                        # 打印输出
                        self.print_indent(f"\n  输出:")
                        self._print_tensordict(current_td, indent=2)

                    except Exception as e:
                        self.print_indent(f"  ✗ 前向传播失败: {e}")
                        break
        else:
            # 非 SequenceModel，直接执行
            try:
                with torch.no_grad():
                    output = model(input_data)

                self.print_indent("\n输出数据:")
                self._print_tensordict(output, indent=1)

            except Exception as e:
                self.print_indent(f"✗ 前向传播失败: {e}")

    def _print_tensordict(self, td: TensorDictBase, indent: int = 0):
        """打印 TensorDict 的详细信息"""
        if not isinstance(td, TensorDictBase):
            self.print_indent(f"类型: {type(td)}, 值: {td}", extra_indent=indent)
            return

        try:
            keys = sorted(td.keys())
        except TypeError:
            keys = list(td.keys())

        for key in keys:
            value = td.get(key)

            if isinstance(value, torch.Tensor):
                self.print_indent(
                    f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}, "
                    f"device={value.device}, range=[{value.min().item():.3f}, {value.max().item():.3f}]",
                    extra_indent=indent
                )
            elif isinstance(value, TensorDictBase):
                self.print_indent(f"{key}: (nested TensorDict)", extra_indent=indent)
                self._print_tensordict(value, indent + 1)
            else:
                self.print_indent(f"{key}: {type(value)} = {value}", extra_indent=indent)

    def check_parameter_sharing(self, models: Dict[str, nn.Module]):
        """检查参数共享情况"""
        self.print_section("参数共享检查", level=1)

        # 收集所有参数的 ID
        param_ids = {}

        for model_name, model in models.items():
            self.print_indent(f"\n模型: {model_name}")

            for param_name, param in model.named_parameters():
                param_id = id(param)

                if param_id in param_ids:
                    # 发现共享参数
                    self.print_indent(
                        f"  ✓ SHARED: {param_name} <--> {param_ids[param_id]}",
                        extra_indent=1
                    )
                else:
                    param_ids[param_id] = f"{model_name}.{param_name}"
                    self.print_indent(
                        f"  • {param_name}: shape={tuple(param.shape)}, params={param.numel()}",
                        extra_indent=1
                    )

    def inspect_attention_module(self, attention_module, module_name: str):
        """详细检查 Attention 模块的实现"""
        self.print_section(f"Attention 模块详细检查: {module_name}", level=1)

        # 检查是否是自定义的 Attention
        self.print_indent(f"类名: {attention_module.__class__.__name__}")
        self.print_indent(f"模块类型: {type(attention_module)}")

        # 打印完整的模块结构
        self.print_indent("\n完整模块结构:")
        for line in str(attention_module).split('\n'):
            self.print_indent(line, extra_indent=1)

        # 检查内部子模块
        self.print_indent("\n内部子模块:")
        for name, submodule in attention_module.named_children():
            self.print_indent(f"  {name}: {submodule.__class__.__name__}")

            # 如果是 AttentionBlock 或 AgentNet，打印更多细节
            if hasattr(submodule, 'embedding_dim'):
                self.print_indent(f"    embedding_dim: {submodule.embedding_dim}", extra_indent=1)
            if hasattr(submodule, 'attention'):
                attn = submodule.attention
                if hasattr(attn, 'embed_dim'):
                    self.print_indent(f"    attention.embed_dim: {attn.embed_dim}", extra_indent=1)
                if hasattr(attn, 'num_heads'):
                    self.print_indent(f"    attention.num_heads: {attn.num_heads}", extra_indent=1)

        # 检查关键配置
        config_attrs = [
            'centralised',
            'input_has_agent_dim',
            'share_params',
            'n_agents',
        ]

        self.print_indent("\n配置参数:")
        for attr in config_attrs:
            if hasattr(attention_module, attr):
                self.print_indent(f"  {attr}: {getattr(attention_module, attr)}")

        # 检查编码器（如果存在）
        if hasattr(attention_module, 'encoders'):
            self.print_indent("\n编码器 (Encoders):")
            encoders = attention_module.encoders
            if isinstance(encoders, nn.ModuleDict):
                for enc_name, encoder in encoders.items():
                    if isinstance(encoder, nn.Linear):
                        self.print_indent(
                            f"  {enc_name}: Linear(in={encoder.in_features}, out={encoder.out_features})"
                        )
                    else:
                        self.print_indent(f"  {enc_name}: {encoder.__class__.__name__}")

        # 检查 attention layers
        if hasattr(attention_module, 'attention_layers'):
            self.print_indent("\nAttention Layers:")
            layers = attention_module.attention_layers
            if isinstance(layers, nn.ModuleList):
                self.print_indent(f"  数量: {len(layers)}")
                for i, layer in enumerate(layers):
                    self.print_indent(f"  Layer {i}: {layer.__class__.__name__}")

        # 检查 MLP（如果存在）
        if hasattr(attention_module, 'mlp'):
            self.print_indent("\nMLP:")
            mlp = attention_module.mlp
            self.print_indent(f"  类型: {mlp.__class__.__name__}")
            if hasattr(mlp, 'features'):
                self.print_indent(f"  结构: {mlp.features}")

    def create_sample_input(self, model, n_agents: int = 2, batch_size: int = 4):
        """为模型创建示例输入"""
        # 根据模型的 input_spec 创建输入
        if not hasattr(model, 'input_spec'):
            warnings.warn("模型没有 input_spec，无法创建示例输入")
            return None

        input_spec = model.input_spec
        sample_td = TensorDict({}, batch_size=[batch_size])

        # 如果模型需要 agent 维度
        if hasattr(model, 'input_has_agent_dim') and model.input_has_agent_dim:
            batch_shape = [batch_size, n_agents]
        else:
            batch_shape = [batch_size]

        # 遍历 input_spec 创建数据
        for key in input_spec.keys():
            spec = input_spec[key]
            if hasattr(spec, 'shape'):
                shape = list(batch_shape) + list(spec.shape)
                sample_td[key] = torch.randn(*shape)

        return sample_td

    def full_inspection(
        self,
        actor_models: Dict[str, nn.Module],
        critic_models: Dict[str, nn.Module],
        sample_batch: TensorDictBase = None,
    ):
        """完整检查：actor + critic"""

        print("\n" + "🔍" * 40)
        print("  MAPPO 模型详细检查报告")
        print("🔍" * 40)

        # 1. Actor 检查
        for group_name, actor in actor_models.items():
            self.inspect_model_config(actor, f"{group_name} Actor")

            # 递归查找 Attention 模块
            attention_modules = self._find_modules_by_type(actor, 'Attention')
            for i, attn_module in enumerate(attention_modules):
                self.inspect_attention_module(attn_module, f"{group_name} Actor Attention {i}")

            # 如果有示例输入，进行前向传播跟踪
            if sample_batch is not None:
                group_input = sample_batch.get(group_name, None)
                if group_input is not None:
                    self.trace_forward_pass(actor, group_input, f"{group_name} Actor")

        # 2. Critic 检查
        for group_name, critic in critic_models.items():
            self.inspect_model_config(critic, f"{group_name} Critic")

            attention_modules = self._find_modules_by_type(critic, 'Attention')
            for i, attn_module in enumerate(attention_modules):
                self.inspect_attention_module(attn_module, f"{group_name} Critic Attention {i}")

        # 3. 参数共享检查
        all_models = {}
        for name, model in actor_models.items():
            all_models[f"{name}_actor"] = model
        for name, model in critic_models.items():
            all_models[f"{name}_critic"] = model

        self.check_parameter_sharing(all_models)

        print("\n" + "🔍" * 40)
        print("  检查完成")
        print("🔍" * 40 + "\n")

    def _find_modules_by_type(self, module: nn.Module, type_name: str) -> List[nn.Module]:
        """递归查找特定类型的模块"""
        found = []

        # 检查当前模块
        if type_name.lower() in module.__class__.__name__.lower():
            found.append(module)

        # 递归检查子模块
        for child in module.children():
            found.extend(self._find_modules_by_type(child, type_name))

        return found
