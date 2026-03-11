from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from typing import Set, Tuple, Type, Dict, List, Optional, Any

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MLP, MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from benchmarl.models.debug_utils import debug_print, debug_separator

class AttentionBlock(nn.Module):
    """标准的 Transformer 编码器层，支持多维 Batch 自动压扁"""
    def __init__(self, embedding_dim: int, num_heads: int, ffn_multiplier: int = 4, dropout_prob: float = 0.1, device: str | torch.device = "cpu"):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, device=device
        )
        self.norm1 = nn.LayerNorm(embedding_dim, device=device)
        self.norm2 = nn.LayerNorm(embedding_dim, device=device)

        ffn_hidden_dim = embedding_dim * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim, device=device),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(ffn_hidden_dim, embedding_dim, device=device),
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 兼容 RNN 场景：将前置维度 (Batch, Step, ...) 压扁为单一 Batch 轴
        # 确保传给 MultiheadAttention 的是 3D 张量
        orig_shape = x.shape
        x_flat = x.flatten(0, -3)
        x_norm = self.norm1(x_flat)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x_flat = x_flat + self.dropout(attn_output)
        x_norm = self.norm2(x_flat)
        x_flat = x_flat + self.dropout(self.ffn(x_norm))
        return x_flat.view(orig_shape)

class Attention(Model):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_attention_layers: int,
        ffn_multiplier: int,
        final_mlp_hidden_layers: List[int],
        dropout_prob: float,
        input_feature_order: List[str],
        roles: Dict[str, List[str]],
        definitions: Dict[str, Dict[str, int]],
        use_ego_embedding: bool,
        encoder_groups: Dict[str, Dict[str, List[str]]] = None,
        ignore_features: List[str] = None,
        share_params_override: Optional[bool] = None,
        share_params_final_mlp: Optional[bool] = None,
        **kwargs,
    ):
        # ✅ 先保存配置（_perform_checks 需要用到）
        self.embedding_dim = embedding_dim
        self.input_feature_order = input_feature_order
        self.roles = roles
        self.definitions = definitions
        self.use_ego_embedding = use_ego_embedding
        self.ignore_features = set(ignore_features) if ignore_features else set()
        self.encoder_groups_config = encoder_groups or {}
        compile_attention_blocks = kwargs.pop('compile_attention_blocks', False)
        compile_mode = kwargs.pop('compile_mode', 'default')

        # ✅ 解析输入切片（_perform_checks 需要 total_expected_dim）
        self._parse_input_slices()

        # ✅ 然后调用基类初始化（会调用 _perform_checks）
        super().__init__(**kwargs)

        # ✅ [NEW] 允许覆盖 share_params (实现混合架构的关键：Shared Attention + Independent GRU)
        if share_params_override is not None:
            self.share_params = share_params_override

        # ✅ [NEW] 保存 final MLP 的独立共享配置（如果未设置则跟随 self.share_params）
        self.share_params_final_mlp = share_params_final_mlp if share_params_final_mlp is not None else self.share_params

        # 初始化编码器
        self.encoders = nn.ModuleDict()
        self.grouped_features: Set[str] = set()
        self.feature_map: Dict[str, Tuple[str, int]] = {}
        self._init_encoders()

        # Transformer 主干
        # ✅ 根据 share_params 决定创建 1 组还是 n_agents 组 attention layers
        if not self.share_params:
            self.attention_layers = nn.ModuleList([
                nn.ModuleList([
                    AttentionBlock(embedding_dim, num_heads, ffn_multiplier, dropout_prob, device=self.device)
                    for _ in range(num_attention_layers)
                ])
                for _ in range(self.n_agents)
            ])
        else:
            self.attention_layers = nn.ModuleList([
                AttentionBlock(embedding_dim, num_heads, ffn_multiplier, dropout_prob, device=self.device)
                for _ in range(num_attention_layers)
            ])

        if compile_attention_blocks:
            if not self.share_params:
                # 编译每个 agent 的每一层
                for agent_idx in range(self.n_agents):
                    for layer_idx in range(num_attention_layers):
                        self.attention_layers[agent_idx][layer_idx] = torch.compile(
                            self.attention_layers[agent_idx][layer_idx],
                            mode=compile_mode
                        )
            else:
                # 编译共享层
                for layer_idx in range(num_attention_layers):
                    self.attention_layers[layer_idx] = torch.compile(
                        self.attention_layers[layer_idx],
                        mode=compile_mode
                    )

        # 输出 MLP（✅ 使用 MultiAgentMLP）
        self._init_final_mlp(final_mlp_hidden_layers)

        # ✅ [NEW] 预计算 forward 中需要的固定值（修复 1）
        self._precompute_constants()

    def _parse_input_slices(self):
        self.slices = {}
        current_idx = 0
        for name in self.input_feature_order:
            if name not in self.definitions: continue
            f_def = self.definitions[name]
            length = f_def['dim'] * f_def['num']
            self.slices[name] = slice(current_idx, current_idx + length)
            current_idx += length
        self.total_expected_dim = current_idx

    def _perform_checks(self):
        super()._perform_checks()
        if self.centralised and self.use_ego_embedding:
            raise ValueError("Attention 在 centralised=True 时不允许开启 use_ego_embedding。")
        actual_dim = self.input_spec[self.in_key].shape[-1]
        if self.total_expected_dim != actual_dim:
            raise ValueError(f"Attention 配置维度 {self.total_expected_dim} 与输入 Spec {actual_dim} 不符。")

    def _precompute_constants(self):
        """✅ [NEW] 预计算 forward 中需要的所有固定值（修复 1）"""
        # 1. 实体特征的有序列表（排除 ignore）
        self._entity_names_ordered = [
            n for n in self.input_feature_order
            if n in self.roles.get('entity', [])
            and n not in self.ignore_features
        ]

        # 2. 全局特征的有序列表
        self._global_names_ordered = [
            n for n in self.roles.get('global', [])
            if n not in self.ignore_features
        ]

        # 3. 实体总数
        self._num_entities = sum(
            self.definitions[n]['num']
            for n in self._entity_names_ordered
        )

        # 4. 全局特征总维度
        self._global_dim = sum(
            self.definitions[n]['dim'] * self.definitions[n]['num']
            for n in self._global_names_ordered
        )

        # 5. 输出特征维度（聚合后）
        if self.use_ego_embedding:
            self._aggregated_dim = self.embedding_dim
        else:
            self._aggregated_dim = self._num_entities * self.embedding_dim

        # 6. 全局特征拼接模式（静态决定）
        if self.input_has_agent_dim and not self.output_has_agent_dim:
            self._global_concat_mode = 0
        elif not self.input_has_agent_dim and self.output_has_agent_dim:
            self._global_concat_mode = 1
        else:
            self._global_concat_mode = 2

    def _init_encoders(self):
        """✅ 正确处理参数共享：share_params=False 时为每个 agent 创建独立编码器"""
        # 处理分组编码器
        for group_name, config in self.encoder_groups_config.items():
            features = [f for f in config['features'] if f not in self.ignore_features]
            if not features: continue
            num_types = len(features)
            base_dim = self.definitions[features[0]]['dim']
            for i, f in enumerate(features):
                self.grouped_features.add(f)
                self.feature_map[f] = (group_name, i)
            self.register_buffer(f"{group_name}_id", torch.eye(num_types, device=self.device))

            # ✅ 根据 share_params 决定创建 1 个还是 n_agents 个编码器
            if not self.share_params:
                self.encoders[group_name] = nn.ModuleList([
                    nn.Linear(base_dim + num_types, self.embedding_dim, device=self.device)
                    for _ in range(self.n_agents)
                ])
            else:
                self.encoders[group_name] = nn.Linear(base_dim + num_types, self.embedding_dim, device=self.device)

        # 处理独立编码器
        for name in self.roles.get('entity', []):
            if name not in self.ignore_features and name not in self.grouped_features:
                # ✅ 根据 share_params 决定创建 1 个还是 n_agents 个编码器
                if not self.share_params:
                    self.encoders[name] = nn.ModuleList([
                        nn.Linear(self.definitions[name]['dim'], self.embedding_dim, device=self.device)
                        for _ in range(self.n_agents)
                    ])
                else:
                    self.encoders[name] = nn.Linear(self.definitions[name]['dim'], self.embedding_dim, device=self.device)

    def _init_final_mlp(self, hidden_layers):
        """✅ 参考 CNN/GRU，使用 MultiAgentMLP 正确处理参数共享"""

        # 计算全局特征维度
        global_dim = sum(self.definitions[n]['dim'] * self.definitions[n]['num']
                         for n in self.roles.get('global', []) if n not in self.ignore_features)

        # 计算实体数量
        valid_entities = [n for n in self.roles.get('entity', []) if n not in self.ignore_features]
        num_entities = sum(self.definitions[n]['num'] for n in valid_entities)

        # 计算 MLP 输入维度
        if self.use_ego_embedding:
            mlp_in = self.embedding_dim + global_dim
        else:
            mlp_in = (num_entities * self.embedding_dim) + global_dim

        mlp_out = self.output_leaf_spec.shape[-1]

        # ✅ 关键：使用 output_has_agent_dim 属性决定 MLP 类型
        if self.output_has_agent_dim:
            # 大部分情况：Actor 或带 agent 维度的 Critic
            self.final_mlp = MultiAgentMLP(
                n_agent_inputs=mlp_in,
                n_agent_outputs=mlp_out,
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params_final_mlp,  # ✅ 使用独立的 final MLP 参数共享配置
                device=self.device,
                num_cells=hidden_layers,
                activation_class=nn.GELU,
                # ✅ 添加 LayerNorm 以稳定训练
                norm_class=nn.LayerNorm,
                norm_kwargs={"eps": 1e-5, "normalized_shape": hidden_layers[0]},
            )
        else:
            # 特殊情况：centralised + share_params 的 Critic
            self.final_mlp = nn.ModuleList(
                [
                    MLP(
                        in_features=mlp_in,
                        out_features=mlp_out,
                        num_cells=hidden_layers,
                        activation_class=nn.GELU,
                        device=self.device,
                        # ✅ 添加 LayerNorm 以稳定训练
                        norm_class=nn.LayerNorm,
                        norm_kwargs={"eps": 1e-5, "normalized_shape": hidden_layers[0]},
                    )
                    for _ in range(self.n_agents if not self.share_params_final_mlp else 1)
                ]
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        debug_separator(self.name, "FORWARD START")

        # 1. 拼接输入（排除 RNN 相关键）
        in_keys = [k for k in self.in_keys if k not in getattr(self, "rnn_keys", [])]
        input_tensor = torch.cat([tensordict.get(key) for key in in_keys], dim=-1)

        debug_print(self.name, "Input tensor", input_tensor,
                   f"share_params={self.share_params}, input_has_agent_dim={self.input_has_agent_dim}, "
                   f"centralised={self.centralised}, output_has_agent_dim={self.output_has_agent_dim}")

        # ✅ 修正：根据 input_has_agent_dim 和 share_params 选择处理路径
        if self.input_has_agent_dim:
            # 路径 A: 输入有 agent 维度
            if self.share_params:
                # A1: 共享参数 - 所有 agent 使用相同编码器
                features = self._forward_shared(input_tensor)
            else:
                # A2: 非共享参数 - 每个 agent 使用独立编码器
                features = self._forward_unshared(input_tensor)
        else:
            # 路径 B: 全局状态输入（无 agent 维度）
            # 参考 GRU/CNN 的实现：为每个 agent 独立运行相同输入
            features = self._forward_global_input(input_tensor)

        debug_print(self.name, "Features after attention", features)

        # ✅ 关键修复：移除 agent 维度（参考 CNN 实现）
        if not self.output_has_agent_dim and features.shape[-2] == self.n_agents:
            features = features[..., 0, :]
            debug_print(self.name, "Features after removing agent dim", features)

        # 5. 拼接全局特征
        # ✅ 使用预计算的全局特征列表和拼接模式（修复 1.2）
        if self._global_dim > 0:
            # 提取全局特征
            if len(self._global_names_ordered) == 1:
                global_cat = input_tensor[..., self.slices[self._global_names_ordered[0]]]
            else:
                global_tensors = [input_tensor[..., self.slices[n]] for n in self._global_names_ordered]
                global_cat = torch.cat(global_tensors, dim=-1)

            # ✅ 使用预计算的拼接模式（避免运行时分支）
            if self._global_concat_mode == 0:  # input_agent & ~output_agent
                global_cat = global_cat[..., 0, :]
            elif self._global_concat_mode == 1:  # ~input_agent & output_agent
                global_cat = global_cat.unsqueeze(-2).expand(
                    *global_cat.shape[:-1], self.n_agents, global_cat.shape[-1]
                )
            # else: mode == 2, 维度已经对齐，无需变换

            features = torch.cat([features, global_cat], dim=-1)
            debug_print(self.name, "Features after global concat", features)

        # 6. MLP 输出
        if self.output_has_agent_dim:
            output = self.final_mlp(features)
        else:
            # 特殊情况：centralised + share_params
            if not self.share_params:
                output = torch.stack(
                    [net(features) for net in self.final_mlp],
                    dim=-2,
                )
            else:
                output = self.final_mlp[0](features)

        debug_print(self.name, "Final output", output)
        debug_separator(self.name, "FORWARD END")

        tensordict.set(self.out_key, output)
        return tensordict

    def _forward_shared(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """共享参数模式：所有 agent 使用相同的编码器和注意力层"""
        debug_print(self.name, "[_forward_shared] Input tensor", input_tensor)

        # 2. Embedding 处理
        embedded_entities = []
        # ✅ 使用预计算的实体列表（修复 1.2）
        for name in self._entity_names_ordered:
            raw = input_tensor[..., self.slices[name]]
            d = self.definitions[name]
            entity_data = raw.view(*raw.shape[:-1], d['num'], d['dim'])
            debug_print(self.name, f"[_forward_shared] Entity '{name}'", entity_data)

            if name in self.grouped_features:
                g_name, t_idx = self.feature_map[name]
                encoder = self.encoders[g_name]
                if isinstance(encoder, nn.ModuleList):
                    raise RuntimeError(
                        f"_forward_shared 被调用，但编码器 '{g_name}' 是 ModuleList！"
                    )
                ids = getattr(self, f"{g_name}_id")[t_idx]
                ids = ids.view(*([1]*(entity_data.dim()-1)), -1).expand(*entity_data.shape[:-1], -1)
                embedded = encoder(torch.cat([entity_data, ids], dim=-1))
                embedded_entities.append(embedded)
            else:
                encoder = self.encoders[name]
                if isinstance(encoder, nn.ModuleList):
                    raise RuntimeError(
                        f"_forward_shared 被调用，但编码器 '{name}' 是 ModuleList！"
                    )
                embedded = encoder(entity_data)
                embedded_entities.append(embedded)

        sequence = torch.cat(embedded_entities, dim=-2)
        debug_print(self.name, "[_forward_shared] Sequence after embedding", sequence)

        # 3. Attention 处理
        pre_attn_shape = sequence.shape
        flat_sequence = sequence.flatten(0, -3)
        debug_print(self.name, "[_forward_shared] Flat sequence", flat_sequence)

        for i, layer in enumerate(self.attention_layers):
            flat_sequence = layer(flat_sequence)
            debug_print(self.name, f"[_forward_shared] After attention layer {i}", flat_sequence)

        sequence = flat_sequence.view(pre_attn_shape)

        # 4. 特征聚合
        if self.use_ego_embedding:
            features = sequence[..., 0, :]
        else:
            features = sequence.flatten(-2, -1)

        debug_print(self.name, "[_forward_shared] Features output", features)
        return features

    def _forward_global_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """全局状态输入模式：输入无 agent 维度，为每个 agent 独立处理"""
        debug_print(self.name, "[_forward_global_input] Input tensor", input_tensor,
                   f"share_params={self.share_params}")

        # ✅ 当 share_params=False 时，需要为每个 agent 分别运行
        if not self.share_params:
            # 为每个 agent 独立处理相同的全局输入
            agent_outputs = []
            for agent_idx in range(self.n_agents):
                embedded_entities = []
                # ✅ 使用预计算的实体列表（修复 1.2）
                for name in self._entity_names_ordered:
                    raw = input_tensor[..., self.slices[name]]
                    d = self.definitions[name]
                    entity_data = raw.view(*raw.shape[:-1], d['num'], d['dim'])

                    if name in self.grouped_features:
                        g_name, t_idx = self.feature_map[name]
                        ids = getattr(self, f"{g_name}_id")[t_idx]
                        ids = ids.view(*([1]*(entity_data.dim()-1)), -1).expand(*entity_data.shape[:-1], -1)
                        encoder = self.encoders[g_name][agent_idx]
                        embedded = encoder(torch.cat([entity_data, ids], dim=-1))
                        embedded_entities.append(embedded)
                    else:
                        encoder = self.encoders[name][agent_idx]
                        embedded = encoder(entity_data)
                        embedded_entities.append(embedded)

                # 拼接实体
                sequence = torch.cat(embedded_entities, dim=-2)

                # Attention 处理（使用 agent 特定的注意力层）
                pre_attn_shape = sequence.shape
                flat_sequence = sequence.flatten(0, -3)

                for layer in self.attention_layers[agent_idx]:
                    flat_sequence = layer(flat_sequence)

                sequence = flat_sequence.view(pre_attn_shape)

                # 特征聚合
                if self.use_ego_embedding:
                    agent_features = sequence[..., 0, :]
                else:
                    agent_features = sequence.flatten(-2, -1)

                agent_outputs.append(agent_features)

            # 堆叠所有 agent 的输出
            features = torch.stack(agent_outputs, dim=-2)
            debug_print(self.name, "[_forward_global_input] Stacked agent outputs", features)

        else:
            # share_params=True: 所有 agent 使用相同的编码器和注意力层
            # 首先编码实体
            embedded_entities = []
            # ✅ 使用预计算的实体列表（修复 1.2）
            for name in self._entity_names_ordered:
                raw = input_tensor[..., self.slices[name]]
                d = self.definitions[name]
                entity_data = raw.view(*raw.shape[:-1], d['num'], d['dim'])

                # printf"[_forward_global_input] entity '{name}' shape: {entity_data.shape}")

                if name in self.grouped_features:
                    g_name, t_idx = self.feature_map[name]
                    ids = getattr(self, f"{g_name}_id")[t_idx]
                    ids = ids.view(*([1]*(entity_data.dim()-1)), -1).expand(*entity_data.shape[:-1], -1)
                    encoder = self.encoders[g_name]
                    embedded = encoder(torch.cat([entity_data, ids], dim=-1))
                    embedded_entities.append(embedded)
                else:
                    encoder = self.encoders[name]
                    embedded = encoder(entity_data)
                    embedded_entities.append(embedded)

            # 拼接所有实体
            sequence = torch.cat(embedded_entities, dim=-2)

            # Attention 处理
            pre_attn_shape = sequence.shape
            flat_sequence = sequence.flatten(0, -3)

            for layer in self.attention_layers:
                flat_sequence = layer(flat_sequence)

            sequence = flat_sequence.view(pre_attn_shape)

            # 特征聚合
            if self.use_ego_embedding:
                single_output = sequence[..., 0, :]
            else:
                single_output = sequence.flatten(-2, -1)

            # 为每个 agent 复制输出（expand agent 维度）
            features = single_output.unsqueeze(-2).expand(
                *single_output.shape[:-1], self.n_agents, single_output.shape[-1]
            )
            debug_print(self.name, "[_forward_global_input] Expanded features", features)

        return features

    def _forward_unshared(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """非共享参数模式：每个 agent 使用独立的编码器和注意力层"""
        debug_print(self.name, "[_forward_unshared] Input tensor", input_tensor)

        agent_outputs = []
        for agent_idx in range(self.n_agents):
            # 提取当前 agent 的输入
            agent_input = input_tensor[..., agent_idx, :]  # (..., feature_dim)

            # 2. Embedding 处理（使用当前 agent 的编码器）
            embedded_entities = []
            # ✅ 使用预计算的实体列表（修复 1.2）
            for name in self._entity_names_ordered:
                raw = agent_input[..., self.slices[name]]
                d = self.definitions[name]
                entity_data = raw.view(*raw.shape[:-1], d['num'], d['dim'])

                if name in self.grouped_features:
                    # 分组编码器 - 使用 agent 特定的编码器
                    g_name, t_idx = self.feature_map[name]
                    ids = getattr(self, f"{g_name}_id")[t_idx]
                    ids = ids.view(*([1]*(entity_data.dim()-1)), -1).expand(*entity_data.shape[:-1], -1)
                    encoder = self.encoders[g_name][agent_idx]  # ✅ 使用 agent 特定的编码器
                    embedded_entities.append(encoder(torch.cat([entity_data, ids], dim=-1)))
                else:
                    # 独立编码器 - 使用 agent 特定的编码器
                    encoder = self.encoders[name][agent_idx]  # ✅ 使用 agent 特定的编码器
                    embedded_entities.append(encoder(entity_data))

            # 拼接所有实体
            sequence = torch.cat(embedded_entities, dim=-2)

            # 3. Attention 处理（使用当前 agent 的注意力层）
            # ✅ 确保维度兼容：AttentionBlock 期望至少 3 维 (..., seq_len, embed)
            pre_attn_shape = sequence.shape
            if sequence.dim() < 3:
                # 如果只有 2 维 (seq_len, embed)，添加 batch 维度
                sequence = sequence.unsqueeze(0)

            flat_sequence = sequence.flatten(0, -3)  # (B*T*..., seq_len, embed)

            for layer in self.attention_layers[agent_idx]:  # ✅ 使用 agent 特定的注意力层
                flat_sequence = layer(flat_sequence)

            sequence = flat_sequence.view(*sequence.shape[:-2], *flat_sequence.shape[-2:])  # 恢复形状

            # 如果之前添加了维度，现在移除
            if pre_attn_shape != sequence.shape:
                sequence = sequence.squeeze(0)

            # 4. 特征聚合
            if self.use_ego_embedding:
                agent_features = sequence[..., 0, :]  # 只取第一个 token（ego）
            else:
                agent_features = sequence.flatten(-2, -1)  # 展平最后两维（seq_len, embed）

            agent_outputs.append(agent_features)

        # 重新组合所有 agent 的输出：(..., n_agents, aggregated_dim)
        features = torch.stack(agent_outputs, dim=-2)
        debug_print(self.name, "[_forward_unshared] Stacked features", features)
        return features

@dataclass
class AttentionConfig(ModelConfig):
    embedding_dim: int = 64
    num_heads: int = 4
    num_attention_layers: int = 2
    ffn_multiplier: int = 4
    final_mlp_hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    dropout_prob: float = 0.0
    use_ego_embedding: bool = False
    input_feature_order: List[str] = field(default_factory=list)
    roles: Dict[str, List[str]] = field(default_factory=dict)
    definitions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    encoder_groups: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    ignore_features: List[str] = field(default_factory=list)
    share_params_override: Optional[bool] = None
    share_params_final_mlp: Optional[bool] = None

    # ✅ 编译配置（学习自 GRU）
    compile_attention_blocks: bool = True  # 是否编译 AttentionBlock
    compile_mode: str = "default"  # 编译模式: "default", "reduce-overhead", "max-autotune"

    @staticmethod
    def associated_class() -> Type[Model]:
        return Attention
