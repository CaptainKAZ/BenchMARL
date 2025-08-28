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
    一个标准的 Transformer 编码器层。

    该模块包含多头自注意力（Multi-Head Self-Attention）、前馈网络（Feed-Forward Network），
    以及围绕它们构建的残差连接（Residual Connections）和层归一化（Layer Normalization）。
    """
    def __init__(self, embedding_dim: int, num_heads: int, ffn_multiplier: int = 4, dropout_prob: float = 0.1, device: str | torch.device = "cpu"):
        """
        初始化 AttentionBlock。

        Args:
            embedding_dim (int): 输入和输出的特征维度。
            num_heads (int): 多头注意力机制中的头数。
            ffn_multiplier (int): 前馈网络中间隐藏层的维度与 embedding_dim 的倍数关系。
            dropout_prob (float): 在注意力和前馈网络中使用的 Dropout 概率。
            device (str | torch.device): 模型所在的设备（如 "cpu" 或 "cuda"）。
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, device=device
        )
        self.norm1 = nn.LayerNorm(embedding_dim, device=device)
        self.norm2 = nn.LayerNorm(embedding_dim, device=device)
        
        ffn_hidden_dim = embedding_dim * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim, device=device),
            nn.GELU(), # 使用 GELU 激活函数，这是现代 Transformer 的常见选择
            nn.Dropout(dropout_prob),
            nn.Linear(ffn_hidden_dim, embedding_dim, device=device),
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        AttentionBlock 的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (Batch, SequenceLength, EmbeddingDim)。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        # 注意力模块 + 残差和归一化
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络模块 + 残差和归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class Attention(Model):
    """
    一个完全由 YAML 配置驱动的、用于 MARL 的注意力网络。

    它能够动态解析扁平化的输入向量，区分实体特征和全局特征，
    并为 Actor 和 Critic 角色构建合适的网络结构。
    支持两种模式：'展平所有实体' 或 '仅使用Ego中心嵌入'。
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
        use_ego_embedding: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # 保存配置
        self.embedding_dim = embedding_dim
        self.input_feature_order = input_feature_order
        self.roles = roles
        self.definitions = definitions
        self.use_ego_embedding = use_ego_embedding
        self.num_attention_layers = num_attention_layers
        
        # 配置健壮性检查
        self.entity_names = self.roles.get('entity', [])
        self.global_names = self.roles.get('global', [])

        if self.use_ego_embedding and not self.entity_names:
            raise ValueError("Ego-Embedding (CLS) 模式已启用, 但 'roles' 中没有定义任何 'entity'。")
        if self.use_ego_embedding and self.num_attention_layers == 0:
            raise ValueError("Ego-Embedding (CLS) 模式需要至少1个注意力层才能聚合信息。")

        # 解析输入观测向量的切片
        self._parse_input_slices()
        
        # 初始化实体嵌入层
        self.embed_layers = nn.ModuleDict({
            name: nn.Linear(self.definitions[name]['dim'], self.embedding_dim, device=self.device)
            for name in self.entity_names
        })
            
        # 初始化注意力层
        self.attention_layers = nn.ModuleList(
            [
                AttentionBlock(
                    self.embedding_dim, num_heads, ffn_multiplier, dropout_prob, device=self.device
                ) for _ in range(self.num_attention_layers)
            ]
        )
        
        # 初始化最终的决策MLP
        self._init_final_mlp(final_mlp_hidden_layers)

    def _parse_input_slices(self):
        """根据配置解析输入向量的切片信息。"""
        self.slices = {}
        current_idx = 0
        for feature_name in self.input_feature_order:
            if feature_name not in self.definitions:
                raise KeyError(f"特征 '{feature_name}' 在 input_feature_order 中定义，但在 definitions 中找不到。")
            feature_def = self.definitions[feature_name]
            length = feature_def['dim'] * feature_def['num']
            self.slices[feature_name] = slice(current_idx, current_idx + length)
            current_idx += length

        # 校验配置计算的维度与环境提供的维度是否一致
        input_features = sum(s.shape[-1] for s in self.input_spec.values(True, True))
        if input_features != current_idx:
             raise ValueError(
                f"模型配置计算出的总输入维度为 {current_idx}，"
                f"但从环境接收到的维度为 {input_features}。请检查 YAML 配置和环境观测。"
             )

    def _init_final_mlp(self, final_mlp_hidden_layers: List[int]):
        """根据配置和模式初始化最终的决策MLP网络。"""
        global_features_dim = sum(self.definitions[name]['dim'] * self.definitions[name]['num'] for name in self.global_names)

        if self.use_ego_embedding:
            # Ego模式: MLP输入 = 单个ego实体嵌入 + 全局特征
            actor_mlp_in_features = self.embedding_dim + global_features_dim
        else:
            # 展平模式: MLP输入 = (所有实体嵌入) + 全局特征
            num_entities = sum(self.definitions[name]['num'] for name in self.entity_names)
            actor_mlp_in_features = (num_entities * self.embedding_dim) + global_features_dim

        self.output_features = self.output_leaf_spec.shape[-1]
        
        mlp_in_features = self.n_agents * actor_mlp_in_features if self.centralised else actor_mlp_in_features
        
        self.final_mlp = MLP(
            in_features=mlp_in_features,
            out_features=self.output_features,
            num_cells=final_mlp_hidden_layers,
            activation_class=nn.GELU,
            device=self.device,
        )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # 1. 从 tensordict 中拼接和解析输入特征
        input_tensor = torch.cat([tensordict.get(key) for key in self.in_keys], dim=-1)
        batch_shape = input_tensor.shape[:-1]
        unpacked_data = {name: input_tensor[..., s] for name, s in self.slices.items()}
        
        # 2. 对所有实体进行嵌入并构建序列
        embedded_entities = []
        for name in self.entity_names:
            feature_def = self.definitions[name]
            entity_data = unpacked_data[name].view(*batch_shape, feature_def['num'], feature_def['dim'])
            embedded_entities.append(self.embed_layers[name](entity_data))
        
        # 如果没有实体，直接跳到全局特征处理
        if not embedded_entities:
            final_attn_features = torch.zeros(*batch_shape, 0, device=self.device)
        else:
            entity_sequence = torch.cat(embedded_entities, dim=-2)
            original_shape = entity_sequence.shape

            # 统一处理 batch 维度，以兼容有无 agent 维度的情况
            processed_sequence = entity_sequence.view(-1, original_shape[-2], self.embedding_dim)

            # 3. 根据配置选择计算路径
            if self.use_ego_embedding:
                # Ego-Embedding (CLS) 模式计算路径
                # 先通过 N-1 层自注意力，让信息在所有实体间充分流动
                for layer in self.attention_layers[:-1]:
                    processed_sequence = layer(processed_sequence)
                
                # 【优化】在最后一层，只计算 Ego 实体（第0个）的输出
                ego_token = processed_sequence[..., 0:1, :]  # Query
                last_attn_layer = self.attention_layers[-1]
                attn_output, _ = last_attn_layer.attention(query=ego_token, key=processed_sequence, value=processed_sequence)
                x = last_attn_layer.norm1(ego_token + last_attn_layer.dropout(attn_output))
                ffn_output = last_attn_layer.ffn(x)
                final_ego_embedding = last_attn_layer.norm2(x + last_attn_layer.dropout(ffn_output))
                
                # 将最终的 Ego 嵌入还原为正确的 batch 形状
                final_attn_features = final_ego_embedding.view(*original_shape[:-2], self.embedding_dim)

            else:
                # 原始展平模式计算路径
                for layer in self.attention_layers:
                    processed_sequence = layer(processed_sequence)
                
                attn_output = processed_sequence.view(*original_shape)
                final_attn_features = attn_output.reshape(*batch_shape, -1)
        
        # 4. 准备全局特征并与注意力输出拼接
        global_features_list = [unpacked_data[name] for name in self.global_names]
        if global_features_list:
            global_features = torch.cat(global_features_list, dim=-1)
            final_mlp_input = torch.cat([final_attn_features, global_features], dim=-1)
        else:
            final_mlp_input = final_attn_features

        # 5. 通过最终的 MLP 进行决策
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
    """
    Attention 模型的配置类，用于从 YAML 文件加载参数。
    """
    # --- 模型结构超参数 ---
    embedding_dim: int = field(metadata={"help": "实体被映射到的内部嵌入维度。"})
    num_heads: int = field(metadata={"help": "多头注意力机制中的头数。"})
    num_attention_layers: int = field(default=2, metadata={"help": "堆叠的 AttentionBlock 层数。"})
    ffn_multiplier: int = field(default=4, metadata={"help": "前馈网络中间层的维度乘数。"})
    final_mlp_hidden_layers: List[int] = field(default_factory=lambda: [256, 128], metadata={"help": "最终决策MLP的隐藏层尺寸。"})
    dropout_prob: float = field(default=0.0, metadata={"help": "在注意力和FFN中使用的Dropout概率。"})

    # --- 输入数据结构定义 ---
    input_feature_order: List[str] = field(default_factory=list, metadata={"help": "定义扁平化观测向量中各个特征的出现顺序。"})
    roles: Dict[str, List[str]] = field(default_factory=dict, metadata={"help": "将特征名称分为 'entity' (参与注意力计算) 和 'global' (直接拼接)。"})
    definitions: Dict[str, Dict[str, int]] = field(default_factory=dict, metadata={"help": "提供每个特征的详细定义，包括 'dim' (单个单位维度) 和 'num' (单位数量)。"})

    # --- 可选高级功能 ---
    use_ego_embedding: bool = field(default=False, metadata={"help": "是否启用Ego-Embedding (CLS) 模式。若为True，仅使用第一个实体的输出进行决策。"})

    @staticmethod
    def associated_class() -> Type[Model]:
        return Attention