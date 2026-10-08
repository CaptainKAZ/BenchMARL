from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from typing import Set, Tuple, Type, Dict, List, Optional, Any

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MLP, MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig, output_has_agent_dim
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


class RoleConditionedMLP(nn.Module):
    """[P0 单组化] 共享主干的"角色条件化"输出头。

    输入 ``[..., n_agents, in_dim]``，输出 ``[..., n_agents, out_dim]``：
      1. 角色嵌入拼接到每个 agent 的特征后面（共享主干随后处理）；
      2. 输出层按角色分组（n_roles 个独立 Linear），使 A1/A2/D 各有自己的输出头。
    这样 4 个 agent 共享同一主干、只在最后的输出层按角色特化（3 个策略/价值头）。

    [投篮按键] ``role_out_dims`` 可给每个角色不同的输出宽度（如 [6, 4, 4]：
    A1 输出 6 个动作参数 = 4 个连续 loc/scale + 2 个离散 logits，其余角色只输出
    4 个连续参数）；宽度不足的输出用 0 补齐到 ``out_dim``（= max(role_out_dims)），
    由适配层与 layup 场景忽略补齐位。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: List[int],
        role_ids: List[int],
        n_roles: int,
        role_embedding_dim: int = 16,
        device: str | torch.device = "cpu",
        role_out_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.role_ids = list(role_ids)
        self.n_roles = int(n_roles)
        # [投篮按键] 每个角色各自的输出宽度（缺省全部 = out_dim）
        if role_out_dims is not None and len(role_out_dims) > 0:
            assert len(role_out_dims) == self.n_roles, (
                f"role_out_dims 长度 {len(role_out_dims)} 与 n_roles {self.n_roles} 不一致"
            )
            self.role_out_dims = [int(d) for d in role_out_dims]
        else:
            self.role_out_dims = [int(out_dim)] * self.n_roles
        self.out_dim = max(self.role_out_dims)
        self.role_embedding = nn.Embedding(self.n_roles, role_embedding_dim, device=device)
        hidden = list(hidden_layers) if hidden_layers else []
        trunk_hidden = hidden if len(hidden) > 0 else [in_dim + role_embedding_dim]
        self.trunk = MLP(
            in_features=in_dim + role_embedding_dim,
            out_features=trunk_hidden[-1],
            num_cells=trunk_hidden,
            activation_class=nn.GELU,
            device=device,
            norm_class=nn.LayerNorm,
            norm_kwargs={"eps": 1e-5, "normalized_shape": trunk_hidden[0]},
        )
        # [P0-A] per-role 输出头加宽为 2 层（共享 trunk 之后按角色特化）
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(trunk_hidden[-1], trunk_hidden[-1], device=device),
                    nn.GELU(),
                    nn.Linear(trunk_hidden[-1], self.role_out_dims[r], device=device),
                )
                for r in range(self.n_roles)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(self.role_ids, device=x.device)
        emb = self.role_embedding(ids)  # [n_agents, role_emb_dim]
        emb = emb.expand(*x.shape[:-2], -1, -1)  # [..., n_agents, role_emb_dim]
        h = self.trunk(torch.cat([x, emb], dim=-1))  # [..., n_agents, trunk_hidden]
        outs = []
        for i, rid in enumerate(self.role_ids):
            o = self.heads[rid](h[..., i, :])
            pad = self.out_dim - o.shape[-1]
            if pad > 0:
                # [投篮按键] 非 A1 角色只输出连续参数，补齐 0 到统一宽度
                o = torch.cat(
                    [o, torch.zeros(*o.shape[:-1], pad, device=o.device, dtype=o.dtype)],
                    dim=-1,
                )
            outs.append(o)
        return torch.stack(outs, dim=-2)


class RoleFiLM(nn.Module):
    """[P0-A] 逐层角色 FiLM 调制（AdaLN 风格）。

    对 attention 层输出的 token 序列按角色做 ``x -> x * (1 + scale) + shift``：
    角色嵌入由共享的 ``role_embedding_ego`` 提供，每个调制层拥有自己的线性头。
    ``to_film`` 零初始化 -> 训练初期等价于恒等映射，不破坏已学表示。
    """

    def __init__(self, embedding_dim: int, role_embedding_dim: int, device: str | torch.device = "cpu"):
        super().__init__()
        self.to_film = nn.Linear(role_embedding_dim, embedding_dim * 2, device=device)
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)

    def forward(self, x: torch.Tensor, role_emb: torch.Tensor) -> torch.Tensor:
        # x: [rows, tokens, dim]；role_emb: [rows, role_emb_dim]
        scale, shift = self.to_film(role_emb).chunk(2, dim=-1)
        return x * (1.0 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)


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
        ego_pool: str = "none",
        encoder_groups: Dict[str, Dict[str, List[str]]] = None,
        ignore_features: List[str] = None,
        share_params_override: Optional[bool] = None,
        share_params_final_mlp: Optional[bool] = None,
        role_ids: Optional[List[int]] = None,
        n_roles: int = 0,
        role_embedding_dim: int = 32,
        use_role_film: bool = True,
        encoders_per_role: bool = False,
        **kwargs,
    ):
        # ✅ 先保存配置（_perform_checks 需要用到）
        self.embedding_dim = embedding_dim
        self.input_feature_order = input_feature_order
        self.roles = roles
        self.definitions = definitions
        self.use_ego_embedding = use_ego_embedding
        # "none" = 只取 ego token（旧行为）；"mean" = ego ⊕ 其余 token 均值（下游带宽翻倍）
        self.ego_pool = ego_pool
        self.ignore_features = set(ignore_features) if ignore_features else set()
        self.encoder_groups_config = encoder_groups or {}
        # [P0 单组化] 角色条件化：role_ids[i] 表示第 i 个 agent 的角色编号（如 A1=0/A2=1/D=2）
        self.role_ids = list(role_ids) if role_ids else []
        self.n_roles = int(n_roles)
        self.role_embedding_dim = int(role_embedding_dim)
        self._role_enabled = len(self.role_ids) > 0 and self.n_roles > 0
        # [P0-A] 是否启用逐层角色 FiLM 调制（默认开；可关做消融）
        self.use_role_film = bool(use_role_film)
        compile_attention_blocks = kwargs.pop('compile_attention_blocks', False)
        compile_mode = kwargs.pop('compile_mode', 'default')

        # ✅ 解析输入切片（_perform_checks 需要 total_expected_dim）
        self._parse_input_slices()

        # ✅ 然后调用基类初始化（会调用 _perform_checks）
        super().__init__(**kwargs)

        # [P0 单组化] 校验角色映射与 agent 数量一致
        if self._role_enabled and len(self.role_ids) != self.n_agents:
            raise ValueError(
                f"role_ids 长度 {len(self.role_ids)} 与 n_agents {self.n_agents} 不一致"
            )

        # [P0-B] 记录 override 之前的 share_params：用于判断"输出是否应带 agent 维"。
        # 共享 trunk + 角色价值头的 critic 需要 share_params=True（一次前向 + 展开），
        # 但输出仍是每角色一个价值 [n_agents, 1]（由 spec 决定），二者必须解耦。
        # share_params 经 **kwargs 传给基类，这里先从 kwargs 取出用于输出几何判断
        self._spec_share_params = kwargs.get("share_params", False)

        # ✅ [NEW] 允许覆盖 share_params (实现混合架构的关键：Shared Attention + Independent GRU)
        if share_params_override is not None:
            self.share_params = share_params_override

        # [P0-B+] 观察者角色专属编码器：每个角色一套 encoder（前端就区分"谁在看什么"）。
        # 仅在"共享主干 + 有 agent 维 + 启用角色"时生效；critic 的全局状态输入无 agent 维，不适用。
        self.use_role_encoders = bool(
            encoders_per_role
            and self._role_enabled
            and self.share_params
            and self.input_has_agent_dim
        )
        # 角色 -> agent 索引（role_ids=[0,1,2,2] -> [[0],[1],[2,3]]），供按角色分组编码使用
        self.role_agent_indices = (
            [[i for i, r in enumerate(self.role_ids) if r == rr] for rr in range(self.n_roles)]
            if self._role_enabled
            else []
        )

        # ✅ [NEW] 保存 final MLP 的独立共享配置（如果未设置则跟随 self.share_params）
        self.share_params_final_mlp = share_params_final_mlp if share_params_final_mlp is not None else self.share_params

        # 初始化编码器
        self.encoders = nn.ModuleDict()
        self.grouped_features: Set[str] = set()
        self.feature_map: Dict[str, Tuple[str, int]] = {}
        self._init_encoders()

        # [P0 单组化] 角色嵌入（注入 actor 的 ego token，让共享主干知道"我是谁"）
        if self._role_enabled and self.role_embedding_dim > 0:
            self.role_embedding_ego = nn.Embedding(
                self.n_roles, self.role_embedding_dim, device=self.device
            )
            # 投影到 token 维度，避免与 ego token 直接相加时维度不匹配
            if self.role_embedding_dim != self.embedding_dim:
                self.role_proj_ego = nn.Sequential(
                    nn.Linear(
                        self.role_embedding_dim,
                        self.embedding_dim,
                        device=self.device,
                    ),
                    nn.SiLU(),
                )

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

        # [P0-A] 逐层角色 FiLM 调制模块（仅在"有 agent 维 + 共享主干 + 启用角色"时创建；
        # 非共享分支的 attention_layers 是 ModuleList of ModuleList，故不加；
        # 全局状态输入的 critic 走 broadcast 展开，FiLM 无法按角色调制，同样不加）
        self.role_film = None
        if (
            self._role_enabled
            and self.use_role_film
            and self.share_params
            and self.input_has_agent_dim
        ):
            self.role_film = nn.ModuleList([
                RoleFiLM(self.embedding_dim, self.role_embedding_dim, device=self.device)
                for _ in range(num_attention_layers)
            ])

        import os as _os
        _attn_env_mode = _os.environ.get("ATTENTION_COMPILE_MODE", None)
        if _attn_env_mode == "off":
            compile_attention_blocks = False
        elif _attn_env_mode:
            compile_mode = _attn_env_mode
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
            # ego_pool 非 none 时额外拼接池化向量（带宽翻倍）
            self._aggregated_dim = self.embedding_dim * (
                2 if self.ego_pool in ("mean", "mean_all") else 1
            )
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
            if getattr(self, "use_role_encoders", False):
                # [P0-B+] 每个角色一套编码器（前端即区分"谁在看"）
                self.encoders[group_name] = nn.ModuleList([
                    nn.Linear(base_dim + num_types, self.embedding_dim, device=self.device)
                    for _ in range(self.n_roles)
                ])
            elif not self.share_params:
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
                if getattr(self, "use_role_encoders", False):
                    # [P0-B+] 每个角色一套编码器
                    self.encoders[name] = nn.ModuleList([
                        nn.Linear(self.definitions[name]['dim'], self.embedding_dim, device=self.device)
                        for _ in range(self.n_roles)
                    ])
                elif not self.share_params:
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
            mlp_in = self.embedding_dim * (
                2 if self.ego_pool in ("mean", "mean_all") else 1
            ) + global_dim
        else:
            mlp_in = (num_entities * self.embedding_dim) + global_dim

        mlp_out = self.output_leaf_spec.shape[-1]

        # [P0 单组化] 角色条件化输出头：共享主干 + 3 个按角色分组的输出头
        if self._role_enabled and self.output_has_agent_dim:
            self.final_mlp = RoleConditionedMLP(
                in_dim=mlp_in,
                out_dim=mlp_out,
                hidden_layers=hidden_layers,
                role_ids=self.role_ids,
                n_roles=self.n_roles,
                role_embedding_dim=self.role_embedding_dim,
                device=self.device,
            )
            return

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

    @property
    def output_has_agent_dim(self) -> bool:
        """[P0-B] 覆盖基类属性：用 ``share_params_override`` 之前的 share_params 计算。

        基类用 ``share_params``/``centralised`` 动态判断输出是否带 agent 维。当 critic
        通过 ``share_params_override=True`` 启用"共享 trunk + 角色价值头"时，输出仍必须
        是每个 agent 一个价值（[n_agents, 1]，由 spec 声明），因此这里用 override 之前的
        share_params 判断，使"参数共享"与"输出几何"解耦。
        """
        return output_has_agent_dim(
            getattr(self, "_spec_share_params", self.share_params), self.centralised
        )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        debug_separator(self.name, "FORWARD START")

        # 1. 拼接输入（排除 RNN 相关键）
        in_keys = [k for k in self.in_keys if k not in getattr(self, "rnn_keys", [])]
        # [fp16 观测] 先转 fp32 再拼接：环境可能输出半精度观测（省内存）。
        # 必须在 cat 之前转换 —— CPU autocast 下对 fp16 张量做 cat 会崩：
        #   RuntimeError: Unexpected floating ScalarType in at::autocast::prioritize
        input_list = []
        for key in in_keys:
            t = tensordict.get(key)
            input_list.append(t if t.dtype == torch.float32 else t.float())
        input_tensor = torch.cat(input_list, dim=-1)

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

    def _aggregate(self, sequence: torch.Tensor) -> torch.Tensor:
        """特征聚合。

        - use_ego_embedding=False: 展平全部 token；
        - ego_pool="none": 只取 self token（旧行为，128 维瓶颈）；
        - ego_pool="mean": self token ⊕ 其余 token 均值（如 128→256 维），
          把「点位/篮筐相对位置、队友与两名防守者的显式表示」带进下游；
        - ego_pool="mean_all": self token ⊕ 全部 token（含 self）均值，
          池化覆盖整组实体，self 在均值中占 1/num_entities；
          注意力主体参数不变，只加宽末端 MLP 与 GRU 的输入。
        """
        if not self.use_ego_embedding:
            return sequence.flatten(-2, -1)
        ego = sequence[..., 0, :]
        if self.ego_pool == "mean_all":
            pooled = sequence.mean(dim=-2)
            return torch.cat([ego, pooled], dim=-1)
        if self.ego_pool == "mean" and sequence.shape[-2] > 1:
            others = sequence[..., 1:, :].mean(dim=-2)
            return torch.cat([ego, others], dim=-1)
        return ego

    def _encode_entity(self, name: str, inp: torch.Tensor) -> torch.Tensor:
        """实体特征编码。

        - 普通模式：共享 encoder（或 share_params=False 时由 _forward_unshared 处理）；
        - 角色专属模式（``use_role_encoders``）：按 agent 的角色分组，各角色用自己的 encoder。
          ``inp`` 的 agent 维为 -3（形状 [..., n_agents, num_entities, feat_dim]）。
        """
        enc = self.encoders[name]
        if isinstance(enc, nn.ModuleList):
            if not getattr(self, "use_role_encoders", False):
                raise RuntimeError(
                    f"_forward_shared 被调用，但编码器 '{name}' 是 ModuleList 且未启用角色专属编码！"
                )
            parts, idx_all = [], []
            for rr, idx in enumerate(self.role_agent_indices):
                if not idx:
                    continue
                idx_t = torch.as_tensor(idx, device=inp.device, dtype=torch.long)
                parts.append(enc[rr](inp.index_select(-3, idx_t)))
                idx_all.append(idx_t)
            idx_all = torch.cat(idx_all)
            # 按角色顺序拼接，再用 argsort 还原成原始 agent 顺序（纯索引操作，无原地写入）
            return torch.cat(parts, dim=-3).index_select(-3, torch.argsort(idx_all))
        return enc(inp)

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
                ids = getattr(self, f"{g_name}_id")[t_idx]
                ids = ids.view(*([1]*(entity_data.dim()-1)), -1).expand(*entity_data.shape[:-1], -1)
                embedded = self._encode_entity(g_name, torch.cat([entity_data, ids], dim=-1))
            else:
                embedded = self._encode_entity(name, entity_data)
            embedded_entities.append(embedded)

        sequence = torch.cat(embedded_entities, dim=-2)
        debug_print(self.name, "[_forward_shared] Sequence after embedding", sequence)

        # [P0 单组化] 角色嵌入注入 ego token（token 0 = self 实体），
        # 让共享的注意力主干知道当前 agent 的角色（A1 / A2 / 防守者）。
        if getattr(self, "_role_enabled", False) and hasattr(self, "role_embedding_ego"):
            ids = torch.as_tensor(self.role_ids, device=sequence.device)
            ego_emb = self.role_embedding_ego(ids)  # [n_agents, role_emb_dim]
            if hasattr(self, "role_proj_ego"):
                ego_emb = self.role_proj_ego(ego_emb)  # [n_agents, embedding_dim]
            sequence = sequence.clone()
            sequence[..., 0, :] = sequence[..., 0, :] + ego_emb

        # 3. Attention 处理
        pre_attn_shape = sequence.shape
        flat_sequence = sequence.flatten(0, -3)
        debug_print(self.name, "[_forward_shared] Flat sequence", flat_sequence)

        # [P0-A] 逐层角色 FiLM：flat 展平顺序为 (..., n_agents)，agent 在最快变化的维度
        role_emb_rows = None
        if getattr(self, "role_film", None) is not None:
            ids_r = torch.as_tensor(self.role_ids, device=flat_sequence.device)
            emb_r = self.role_embedding_ego(ids_r)  # [n_agents, role_emb_dim]
            n_lead = flat_sequence.shape[0] // self.n_agents
            role_emb_rows = emb_r.repeat(n_lead, 1)  # [rows, role_emb_dim]

        for i, layer in enumerate(self.attention_layers):
            flat_sequence = layer(flat_sequence)
            if role_emb_rows is not None and i < len(self.role_film):
                flat_sequence = self.role_film[i](flat_sequence, role_emb_rows)
            debug_print(self.name, f"[_forward_shared] After attention layer {i}", flat_sequence)

        sequence = flat_sequence.view(pre_attn_shape)

        # 4. 特征聚合
        features = self._aggregate(sequence)

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
                agent_features = self._aggregate(sequence)

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
            single_output = self._aggregate(sequence)

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
            agent_features = self._aggregate(sequence)

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
    ego_pool: str = "none"
    input_feature_order: List[str] = field(default_factory=list)
    roles: Dict[str, List[str]] = field(default_factory=dict)
    definitions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    encoder_groups: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    ignore_features: List[str] = field(default_factory=list)
    share_params_override: Optional[bool] = None
    share_params_final_mlp: Optional[bool] = None
    # [P0 单组化] 角色条件化：role_ids[i] = 第 i 个 agent 的角色编号
    role_ids: List[int] = field(default_factory=list)
    n_roles: int = 0
    role_embedding_dim: int = 32
    # [P0-A] 逐层角色 FiLM 调制开关（消融用）
    use_role_film: bool = True
    # [P0-B+] 观察者角色专属编码器开关（每个角色一套 encoder，仅 actor 的带 agent 维输入生效）
    encoders_per_role: bool = False

    # ✅ 编译配置（学习自 GRU）
    compile_attention_blocks: bool = True  # 是否编译 AttentionBlock
    compile_mode: str = "default"  # 编译模式: "default", "reduce-overhead", "max-autotune"

    @staticmethod
    def associated_class() -> Type[Model]:
        return Attention
