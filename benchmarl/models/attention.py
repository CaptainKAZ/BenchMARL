from __future__ import annotations
from dataclasses import dataclass, field, MISSING
from typing import Type, Dict, List, Any

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MLP

from benchmarl.models.common import Model, ModelConfig

class AttentionBlock(nn.Module):
    """
    一个标准的 Transformer 编码器层，包含多头自注意力、前馈网络和残差连接。
    """
    def __init__(self, embedding_dim: int, num_heads: int, ffn_multiplier: int = 4, dropout_prob:float = 0.1, device=None):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, device=device
        )
        self.norm1 = nn.LayerNorm(embedding_dim, device=device)
        self.norm2 = nn.LayerNorm(embedding_dim, device=device)
        
        ffn_hidden_dim = embedding_dim * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim, device=device),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(ffn_hidden_dim, embedding_dim, device=device),
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output)) # Add dropout to attention output as well
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class Attention(Model):
    """
    一个完全由 YAML 配置驱动的、用于 MARL 的注意力网络。
    它能够动态解析扁平化的输入向量，区分实体特征和全局特征，
    并为 Actor 和 Critic 角色构建合适的网络结构。
    """
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.embedding_dim = embedding_dim
        self.input_feature_order = input_feature_order
        self.roles = roles
        self.definitions = definitions
        
        self.entity_names = self.roles.get('entity', [])
        self.global_names = self.roles.get('global', [])
        
        self.slices = {}
        current_idx = 0
        for feature_name in self.input_feature_order:
            feature_def = self.definitions[feature_name]
            length = feature_def['dim'] * feature_def['num']
            self.slices[feature_name] = slice(current_idx, current_idx + length)
            current_idx += length

        input_features = sum(s.shape[-1] for s in self.input_spec.values(True, True))
        if input_features != current_idx:
             raise ValueError(
                f"模型配置计算出的总输入维度为 {current_idx}，"
                f"但从环境接收到的维度为 {input_features}。请检查 YAML 配置和环境观测。"
             )

        self.embed_layers = nn.ModuleDict({
            name: nn.Linear(self.definitions[name]['dim'], self.embedding_dim, device=self.device)
            for name in self.entity_names
        })
            
        self.attention_layers = nn.ModuleList(
            [
                AttentionBlock(
                    self.embedding_dim, num_heads, ffn_multiplier, dropout_prob, device=self.device
                ) for _ in range(num_attention_layers)
            ]
        )
        
        self.num_entities = sum(self.definitions[name]['num'] for name in self.entity_names)
        global_features_dim = sum(self.definitions[name]['dim'] for name in self.global_names)

        # Actor 的输入维度 = (所有实体的数量 * embedding_dim) + 全局特征维度
        actor_mlp_in_features = (self.num_entities * self.embedding_dim) + global_features_dim

        self.output_features = self.output_leaf_spec.shape[-1]
        
        if self.centralised:
            # Critic 的输入维度现在也基于 actor 的维度计算
            critic_input_features = self.n_agents * actor_mlp_in_features
            self.final_mlp = MLP(
                in_features=critic_input_features,
                out_features=self.output_features,
                num_cells=final_mlp_hidden_layers,
                activation_class=nn.Tanh,
                device=self.device,
            )
        else:
            self.final_mlp = MLP(
                in_features=actor_mlp_in_features,
                out_features=self.output_features,
                num_cells=final_mlp_hidden_layers,
                activation_class=nn.Tanh,
                device=self.device,
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        input_tensor = torch.cat([tensordict.get(key) for key in self.in_keys], dim=-1)
        batch_shape = input_tensor.shape[:-1]

        unpacked_data = {name: input_tensor[..., s] for name, s in self.slices.items()}
        
        embedded_entities = []
        for name in self.entity_names:
            feature_def = self.definitions[name]
            entity_data = unpacked_data[name].view(*batch_shape, feature_def['num'], feature_def['dim'])
            embedded_entities.append(self.embed_layers[name](entity_data))
        entity_sequence = torch.cat(embedded_entities, dim=-2)
        
        original_shape = entity_sequence.shape
        if self.input_has_agent_dim:
            num_total_entities = original_shape[-2]
            processed_sequence = entity_sequence.view(-1, num_total_entities, self.embedding_dim)
        else:
            processed_sequence = entity_sequence
        
        for layer in self.attention_layers:
            processed_sequence = layer(processed_sequence)
        
        attn_output = processed_sequence.view(*original_shape)

        # 将所有实体和 embedding 维度展平，为每个 agent 创建一个特征向量
        final_attn_features = attn_output.reshape(*batch_shape, -1)

        # 准备全局特征
        global_features_list = [unpacked_data[name] for name in self.global_names]
        global_features = torch.cat(global_features_list, dim=-1)
        
        # 将展平后的 attention 特征与全局特征拼接
        final_mlp_input = torch.cat([final_attn_features, global_features], dim=-1)
        
        if self.centralised:
            # Critic: 展平所有 agent 的特征
            critic_input = final_mlp_input.reshape(*batch_shape[:-1], -1)
            output = self.final_mlp(critic_input)
        else:
            # Actor: 每个 agent 的特征向量直接输入 MLP
            output = self.final_mlp(final_mlp_input)

        tensordict.set(self.out_key, output)
        return tensordict

@dataclass
class AttentionConfig(ModelConfig):
    """Attention 模型的配置类，完全由 YAML 驱动。"""
    # --- 模型结构超参数 ---
    embedding_dim: int = MISSING
    num_heads: int = MISSING
    num_attention_layers: int = 1
    ffn_multiplier: int = 2
    final_mlp_hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    dropout_prob: float = 0.0

    # --- 输入数据结构定义 ---
    input_feature_order: List[str] = field(default_factory=list)
    roles: Dict[str, List[str]] = field(default_factory=dict)
    definitions: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @staticmethod
    def associated_class() -> Type[Model]:
        return Attention