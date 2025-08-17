from __future__ import annotations
from dataclasses import dataclass, MISSING, field
from typing import Type, Dict, List

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MLP

from benchmarl.models.common import Model, ModelConfig

class AttentionBlock(nn.Module):
    """
    标准的 Transformer 编码器层。
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
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # 添加 dropout 层
            nn.Linear(ffn_hidden_dim, embedding_dim, device=device),
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x

class Attention(Model):
    """
    一个可配置的、鲁棒的多头注意力网络模型，能正确处理 Actor 和 Critic 角色。
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        entity_configs: Dict[str, List[int]],
        num_attention_layers: int = 1,
        ffn_multiplier: int = 4,
        final_mlp_hidden_layers: List[int] = [256, 128],
        dropout_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.entity_configs = entity_configs
        
        # 动态验证逻辑
        self.num_total_entities = sum(c[1] for c in entity_configs.values())
        self.entity_dim = 4
        expected_input_dim = self.num_total_entities * self.entity_dim
        self.input_features = sum(s.shape[-1] for s in self.input_spec.values(True, True))
        if self.input_features != expected_input_dim:
             raise ValueError(f"模型配置期望输入维度 {expected_input_dim}，但从环境接收到 {self.input_features}。")

        # 嵌入层
        self.embed_layers = nn.ModuleDict({
            name: nn.Linear(cfg[0], embedding_dim, device=self.device) for name, cfg in entity_configs.items()
        })
            
        # 注意力层
        self.attention_layers = nn.ModuleList(
            [AttentionBlock(embedding_dim, num_heads, ffn_multiplier, dropout_prob, device=self.device) for _ in range(num_attention_layers)]
        )
        
        self.output_features = self.output_leaf_spec.shape[-1]
        
        # --- 核心修改：根据角色定义输出层 ---
        if self.centralised:
            # Critic 场景: 聚合所有信息后通过一个MLP输出价值
            if self.input_has_agent_dim:
                critic_input_features = self.n_agents * self.num_total_entities * self.embedding_dim
            else:
                critic_input_features = self.num_total_entities * self.embedding_dim

            self.final_mlp = MLP(
                in_features=critic_input_features,
                out_features=self.output_features,
                num_cells=final_mlp_hidden_layers,
                activation_class=nn.Tanh,
                device=self.device,
            )
        else:
            # Actor 场景: 每个智能体独立输出动作
            self.final_mlp = nn.Linear(self.num_total_entities * self.embedding_dim, self.output_features, device=self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        input_tensor = torch.cat([tensordict.get(key) for key in self.in_keys], dim=-1)
        batch_shape = input_tensor.shape[:-1]
        
        reshaped_tensor = input_tensor.view(*batch_shape, self.num_total_entities, self.entity_dim)
        
        embedded_entities = []
        current_slot = 0
        for name, cfg in self.entity_configs.items():
            dim, num = cfg
            slots = reshaped_tensor[..., current_slot:current_slot+num, :dim]
            embedded_entities.append(self.embed_layers[name](slots))
            current_slot += num
            
        entity_sequence = torch.cat(embedded_entities, dim=-2)
        
        # 将 batch 和 agent 维度合并 (如果存在)
        original_shape = entity_sequence.shape
        if self.input_has_agent_dim:
            processed_sequence = entity_sequence.view(-1, self.num_total_entities, self.embedding_dim)
        else:
            processed_sequence = entity_sequence
        
        for layer in self.attention_layers:
            processed_sequence = layer(processed_sequence)
        
        attn_output = processed_sequence.view(*original_shape)
        
        if self.centralised:
            # Critic: 展平所有 agent 和 entity 的信息
            if self.input_has_agent_dim:
                # 展平 agent, entity, embedding 维度
                critic_input = attn_output.reshape(*batch_shape[:-1], -1)
            else:
                # 展平 entity, embedding 维度
                critic_input = attn_output.reshape(*batch_shape, -1)
            output = self.final_mlp(critic_input)
        else:
            # Actor: 每个 agent 独立输出
            actor_input = attn_output.reshape(*batch_shape, -1)
            output = self.final_mlp(actor_input)

        tensordict.set(self.out_key, output)
        return tensordict

@dataclass
class AttentionConfig(ModelConfig):
    """AttentionModel 的配置数据类。"""
    embedding_dim: int = MISSING
    num_heads: int = MISSING
    num_attention_layers: int = 1
    ffn_multiplier: int = 4
    final_mlp_hidden_layers: List[int] = field(default_factory=lambda: [256, 128])
    entity_configs: Dict[str, List[int]] = field(default_factory=dict)
    dropout_prob: float = 0.1

    @staticmethod
    def associated_class():
        return Attention