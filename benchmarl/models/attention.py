from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from typing import Set, Tuple, Type, Dict, List, Any

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
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output) # 残差连接在 Norm 之外
        
        # 前馈网络模块 + 残差和归一化
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        return x


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
        encoder_groups: Dict[str, Dict[str, List[str]]] = None, # 新增: 分组配置
        ignore_features: List[str] = None,                      # 新增: 忽略列表
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # --- 基础配置保存 ---
        self.embedding_dim = embedding_dim
        self.input_feature_order = input_feature_order
        self.roles = roles
        self.definitions = definitions
        self.use_ego_embedding = use_ego_embedding
        self.num_attention_layers = num_attention_layers
        
        # 健壮性检查
        self.entity_names = self.roles.get('entity', [])
        self.global_names = self.roles.get('global', [])
        
        # --- 1. 处理忽略列表与切片解析 ---
        self.ignore_features = set(ignore_features) if ignore_features else set()
        self._parse_input_slices() # 解析 slices
        
        # --- 2. 构建编码器 (含优化) ---
        self.encoder_groups_config = encoder_groups or {}
        
        # 存储: 特征名 -> (组名, 组内ID索引)
        self.feature_map: Dict[str, Tuple[str, int]] = {} 
        self.grouped_features: Set[str] = set()
        
        # 存储所有的 Linear 层
        self.encoders = nn.ModuleDict()

        # [A] 处理共享分组 (Shared Groups)
        for group_name, config in self.encoder_groups_config.items():
            features_in_group = [f for f in config['features'] if f not in self.ignore_features]
            if not features_in_group:
                continue

            num_types = len(features_in_group)
            base_dim = self.definitions[features_in_group[0]]['dim']
            
            # 校验维度一致性
            for f_idx, fname in enumerate(features_in_group):
                if self.definitions[fname]['dim'] != base_dim:
                    raise ValueError(f"Group '{group_name}' 维度不匹配: {fname} vs {features_in_group[0]}")
                
                self.grouped_features.add(fname)
                self.feature_map[fname] = (group_name, f_idx)

            # [优化] 注册 ID Buffer (One-Hot Matrix)
            # 形状: (Num_Types, Num_Types) -> 这是一个单位阵
            # register_buffer 保证它会随模型存取且移动到 GPU，但不是可训练参数
            id_matrix = torch.eye(num_types, device=self.device)
            self.register_buffer(f"{group_name}_id_buffer", id_matrix)

            # 创建共享 Linear: 输入 = 原始特征 + ID向量
            self.encoders[group_name] = nn.Linear(base_dim + num_types, self.embedding_dim, device=self.device)

        # [B] 处理独立特征 (Independent Features)
        for name in self.entity_names:
            if name in self.ignore_features or name in self.grouped_features:
                continue
            
            # 独立特征不需要 ID，直接映射
            self.encoders[name] = nn.Linear(self.definitions[name]['dim'], self.embedding_dim, device=self.device)

        # --- 3. Transformer 主干 ---
        self.attention_layers = nn.ModuleList([
            AttentionBlock(embedding_dim, num_heads, ffn_multiplier, dropout_prob, device=self.device) 
            for _ in range(self.num_attention_layers)
        ])

        # --- 4. 决策 MLP ---
        self._init_final_mlp(final_mlp_hidden_layers)

    def _parse_input_slices(self):
        """解析输入向量切片"""
        self.slices = {}
        current_idx = 0
        for feature_name in self.input_feature_order:
            if feature_name not in self.definitions: continue # 容错
            
            feature_def = self.definitions[feature_name]
            length = feature_def['dim'] * feature_def['num']
            
            # 只有不在忽略列表里的才记录切片(或者全部记录，但在forward里跳过，推荐后者以防索引错乱)
            self.slices[feature_name] = slice(current_idx, current_idx + length)
            current_idx += length

    def _init_final_mlp(self, hidden_layers):
        """初始化输出层"""
        # 计算全局特征维度
        global_dim = sum(self.definitions[n]['dim'] * self.definitions[n]['num'] 
                         for n in self.global_names if n not in self.ignore_features)
        
        # 计算 MLP 输入维度
        if self.use_ego_embedding:
            mlp_in = self.embedding_dim + global_dim
        else:
            # 只有未被忽略的实体才会计数
            valid_entities = [n for n in self.entity_names if n not in self.ignore_features]
            num_entities = sum(self.definitions[n]['num'] for n in valid_entities)
            mlp_in = (num_entities * self.embedding_dim) + global_dim

        self.output_features = self.output_leaf_spec.shape[-1]
        
        # Critic (Centralised) vs Actor
        # final_in = self.n_agents * mlp_in if self.centralised else mlp_in
        # print(final_in)
        
        self.final_mlp = MLP(
            norm_class=nn.LayerNorm,
            norm_kwargs={"eps": 1e-5, "normalized_shape":hidden_layers[0]},
            in_features=mlp_in, out_features=self.output_features,
            num_cells=hidden_layers, activation_class=nn.GELU, device=self.device
        )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # 1. 拼接并解析输入
        input_tensor = torch.cat([tensordict.get(key) for key in self.in_keys], dim=-1)

        # input_tensor: (Batch, Total_Raw_Features)
        
        embedded_entities = []
        
        # 严格按照 input_feature_order 遍历，保证 Sequence 顺序
        for name in self.input_feature_order:
            if name not in self.roles['entity']: continue # 跳过 global
            if name in self.ignore_features: continue     # 跳过 ignored
            
            # 获取原始数据并 Reshape 为 (Batch..., Num, Dim)
            raw_flat = input_tensor[..., self.slices[name]]
            def_ = self.definitions[name]
            # (Batch..., Num_Entities, Feature_Dim)
            entity_data = raw_flat.view(*raw_flat.shape[:-1], def_['num'], def_['dim']) 
            
            if name in self.grouped_features:
                # --- [A] 共享组逻辑 (One-Hot 优化版) ---
                group_name, type_idx = self.feature_map[name]
                encoder = self.encoders[group_name]
                
                # 1. 从 Buffer 获取预计算好的 ID 向量 (Dim: Num_Types)
                # 使用 getattr 动态获取 buffer
                id_buffer = getattr(self, f"{group_name}_id_buffer")
                id_vec = id_buffer[type_idx] # 取出第 type_idx 行
                
                # 2. 扩展维度以匹配数据: (1, 1, ID_Dim) -> (Batch, Num, ID_Dim)
                # view: (1...1, ID_Dim)
                target_shape = entity_data.shape[:-1] # (Batch..., Num)
                id_expanded = id_vec.view(*([1] * len(target_shape)), -1).expand(*target_shape, -1)
                
                # 3. 拼接并编码
                inp = torch.cat([entity_data, id_expanded], dim=-1)
                embedded_entities.append(encoder(inp))
                
            else:
                # --- [B] 独立逻辑 ---
                encoder = self.encoders[name]
                embedded_entities.append(encoder(entity_data))
        
        # 2. 此时 embedded_entities 是一个列表，里面每个元素是 (Batch, Num_i, Embed_Dim)
        if not embedded_entities:
            # 极端情况处理
            raise ValueError(
                "Attention 模型接收到的实体列表为空！\n"
                "请检查 YAML 配置：\n"
                "1. 'input_feature_order' 中是否包含属于 'entity'角色的特征？\n"
                "2. 是否所有的实体特征都被误放入了 'ignore_features'？"
            )
        else:
            sequence = torch.cat(embedded_entities, dim=-2)
            
        # 3. Transformer 处理
        # 展平 Batch 维度供 Transformer 使用: (B*Agents, Seq_Len, Dim)
        batch_dims = sequence.shape[:-2]
        flat_sequence = sequence.flatten(0, len(batch_dims)-1) 
        
        for layer in self.attention_layers:
            flat_sequence = layer(flat_sequence)
        
        # 4. 聚合策略 (Ego vs Flatten)
        # 还原 Batch 维度
        sequence = flat_sequence.view(*batch_dims, -1, self.embedding_dim)
        
        if self.use_ego_embedding:
            # 取第一个 token (Ego)
            # 前提：input_feature_order 第一个是 self_embed
            features = sequence[..., 0, :] 
        else:
            features = sequence.flatten(-2, -1) # 展平所有
            
        # 5. 拼接 Global 特征
        global_feats = []
        for name in self.global_names:
            if name not in self.ignore_features:
                global_feats.append(input_tensor[..., self.slices[name]])
        
        if global_feats:
            features = torch.cat([features, *global_feats], dim=-1)

        # print(f"before flatten {features.shape}")
        # # 6. Critic / Actor 输出
        # if self.centralised:
        #     # Critic: 简单的 Reshape 拼接所有 Agent (注意：这可能导致维度很大，可改为 Mean Pooling)
        #     features = features.flatten(-2, -1)

        output = self.final_mlp(features)
        tensordict.set(self.out_key, output)
        return tensordict
    
   

@dataclass
class AttentionConfig(ModelConfig):
    embedding_dim: int = field(default=64, metadata={"help": "Embedding 维度"})
    num_heads: int = field(default=4, metadata={"help": "Attention 头数"})
    num_attention_layers: int = field(default=2, metadata={"help": "Transformer 层数"})
    ffn_multiplier: int = field(default=4, metadata={"help": "FFN 隐层倍数"})
    final_mlp_hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    dropout_prob: float = field(default=0.0)
    
    use_ego_embedding: bool = field(default=True, metadata={"help": "是否只使用 Ego Token 进行决策"})
    
    # 结构定义
    input_feature_order: List[str] = field(default_factory=list)
    roles: Dict[str, List[str]] = field(default_factory=dict)
    definitions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # 新增配置
    encoder_groups: Dict[str, Dict[str, List[str]]] = field(default_factory=dict, metadata={"help": "同构共享组配置"})
    ignore_features: List[str] = field(default_factory=list, metadata={"help": "忽略的特征名"})

    @staticmethod
    def associated_class() -> Type[Model]:
        return Attention