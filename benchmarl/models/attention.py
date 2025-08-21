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
    标准的 Transformer 编码器层 (回退到使用 nn.MultiheadAttention)。
    - 采用 Pre-Norm 结构，训练更稳定。
    - dropout 位置经过修正，符合标准实践。
    """
    def __init__(self, embedding_dim: int, num_heads: int, ffn_multiplier: int = 4, dropout_prob:float = 0.1, device=None):
        super().__init__()

        # 1. 第一个子层：多头自注意力
        self.norm1 = nn.LayerNorm(embedding_dim, device=device)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_prob,  # nn.MultiheadAttention 内置了对 attention weights 的 dropout
            batch_first=True,
            device=device
        )
        
        # 2. 第二个子层：前馈网络 (FFN)
        self.norm2 = nn.LayerNorm(embedding_dim, device=device)
        ffn_hidden_dim = embedding_dim * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim, device=device),
            nn.ReLU(), # 下次冷启动换成GELU
            nn.Linear(ffn_hidden_dim, embedding_dim, device=device),
        )

        # 3. 定义一个用于子层输出的 dropout
        # 这个 dropout 会应用在 attention 的输出和 FFN 的输出上
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- 第一个子层：多头注意力 ---
        # Pre-Norm 结构: Norm -> Attention -> Dropout -> Add
        residual = x
        x_norm = self.norm1(x)
        
        # nn.MultiheadAttention 需要 query, key, value
        # 对于自注意力，这三者是相同的
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        
        # 将 dropout 应用在子层输出上，然后进行残差连接
        x = residual + self.dropout(attn_output)
        
        # --- 第二个子层：前馈网络 ---
        # Pre-Norm 结构: Norm -> FFN -> Dropout -> Add
        residual = x
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        
        # 将 dropout 应用在子层输出上，然后进行残差连接
        x = residual + self.dropout(ffn_output)
        
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
        
        # --- 核心修改 1: 动态验证逻辑 ---
        # 移除 self.entity_dim = 4
        self.num_total_entities = sum(c[1] for c in entity_configs.values())
        
        # 根据 entity_configs 精确计算期望的输入维度
        expected_input_dim = sum(cfg[0] * cfg[1] for cfg in self.entity_configs.values())
        
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
        
        # --- 核心修改：根据角色定义输出层 (这部分逻辑不变) ---
        if self.centralised:
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
            self.final_mlp = MLP(
                in_features=self.num_total_entities * self.embedding_dim,
                out_features=self.output_features,
                num_cells=final_mlp_hidden_layers,
                activation_class=nn.Tanh,
                device=self.device,
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        input_tensor = torch.cat([tensordict.get(key) for key in self.in_keys], dim=-1)
        batch_shape = input_tensor.shape[:-1]
        
        # --- 核心修改 2: 动态切分和重塑输入张量 ---
        # 移除老的 reshape 逻辑: 
        # reshaped_tensor = input_tensor.view(*batch_shape, self.num_total_entities, self.entity_dim)
        
        embedded_entities = []
        current_dim_idx = 0
        for name, cfg in self.entity_configs.items():
            # cfg 格式: [有效特征维度, 槽的数量]
            effective_dim, num_slots = cfg
            
            # 计算当前实体类型在扁平向量中占用的总特征数
            total_features_for_entity = effective_dim * num_slots
            
            # 从扁平的 input_tensor 中切片出当前实体的数据
            entity_chunk = input_tensor[..., current_dim_idx : current_dim_idx + total_features_for_entity]
            
            # 将切出的数据块重塑为 (..., 槽数, 特征维度)
            reshaped_chunk = entity_chunk.view(*batch_shape, num_slots, effective_dim)
            
            # 应用对应的嵌入层
            embedded_entities.append(self.embed_layers[name](reshaped_chunk))
            
            # 更新下一次切片的起始位置
            current_dim_idx += total_features_for_entity
            
        entity_sequence = torch.cat(embedded_entities, dim=-2)
        
        # 后续逻辑保持不变
        original_shape = entity_sequence.shape
        if self.input_has_agent_dim:
            processed_sequence = entity_sequence.view(-1, self.num_total_entities, self.embedding_dim)
        else:
            processed_sequence = entity_sequence
        
        for layer in self.attention_layers:
            processed_sequence = layer(processed_sequence)
        
        attn_output = processed_sequence.view(*original_shape)
        
        if self.centralised:
            if self.input_has_agent_dim:
                critic_input = attn_output.reshape(*batch_shape[:-1], -1)
            else:
                critic_input = attn_output.reshape(*batch_shape, -1)
            output = self.final_mlp(critic_input)
        else:
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