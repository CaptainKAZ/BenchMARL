#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from typing import List, Optional, Sequence, Type

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import expand_as_right, unravel_key_list
from torch import nn
from torchrl.data.tensor_specs import Composite, Unbounded

from torchrl.modules import GRUCell, MLP, MultiAgentMLP

from benchmarl.models.attention import RoleConditionedMLP
from benchmarl.models.common import Model, ModelConfig
from benchmarl.utils import DEVICE_TYPING
from benchmarl.models.debug_utils import debug_print, debug_separator


class GRU(torch.nn.Module):
    """多层 GRU。

    [性能改造] 旧实现用 torchrl 的 GRUCell 在 Python 里逐时间步循环（T=150 ⇒ 每步都要发一次
    kernel），训练时吃掉 actor 前向的一大块时间，而且 vmap 下无法走 cuDNN 融合核。
    新实现用 ``nn.GRU``（cuDNN 融合实现），把"逐时间步"改成"整段一次调用"：
    按 ``is_init`` 把每个 batch 行切成若干片段（run），所有片段拼成一个大 batch 一次算完，
    再 gather 回原时间轴。数值上与旧实现完全等价（T=1 采集路径/中途重置/非零 h_0/多层
    共 6 组用例逐位一致，见 /tmp/opencode/probe_fused_gru.py）。

    形状约定：
        input:   (B, T, F)
        is_init: (B, T, 1)，True 表示该时刻开启新片段（隐状态清零）
        h:       (B, L, H)，片段起点处的隐状态（训练时通常全 0）
    返回：
        output:  (B, T, H)（最后一层输出）
        h_n:     (B, L, H)（每行最后一个片段的终态）
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        device: DEVICE_TYPING,
        n_layers: int,
        dropout: float,
        bias: bool,
        time_dim: int = -2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.time_dim = time_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bias = bias

        self.grus = torch.nn.ModuleList(
            [
                torch.nn.GRU(
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    num_layers=1,
                    bias=self.bias,
                    batch_first=True,
                    device=self.device,
                )
                for i in range(self.n_layers)
            ]
        )

    def forward(
        self,
        input,
        is_init,
        h,
    ):
        B, T, F_ = input.shape
        dev = input.device
        init = is_init[..., 0].reshape(B, T)

        # ---- 1) 向量化切分：行内新段起点 = 行首 or 任意 reset 帧 ----
        t_grid = torch.arange(T, device=dev).expand(B, T)
        new_seg = init.clone()
        new_seg[:, 0] = True
        seg_in_row = new_seg.long().cumsum(1) - 1  # (B,T) 段在本行内的序号(0起)
        seg_start = torch.cummax(
            torch.where(new_seg, t_grid, torch.zeros_like(t_grid)), 1
        ).values  # 每个位置所属段的起始时刻
        pos_in_seg = t_grid - seg_start  # (B,T) 段内偏移
        n_seg = seg_in_row[:, -1] + 1  # (B,)
        seg_base = torch.cumsum(n_seg, 0) - n_seg  # (B,) 每行第一段的全局段号
        seg_id = (seg_base.unsqueeze(-1) + seg_in_row).reshape(B * T)
        R = int(n_seg.sum().item())
        max_len = int(pos_in_seg.max().item()) + 1

        # 目标位置：每个 (b,t) -> 展平后的 (段, 段内偏移)
        dest = seg_id * max_len + pos_in_seg.reshape(-1)
        input_2d = input.reshape(B * T, F_)
        x_flat = input_2d.new_zeros(R * max_len, F_)
        x_flat.index_copy_(0, dest, input_2d)
        x = x_flat.view(R, max_len, F_)

        # 每行最后一段的终态位置（用于 h_n）
        last_in_seg = torch.ones_like(new_seg)
        last_in_seg[:, :-1] = seg_in_row[:, :-1] != seg_in_row[:, 1:]
        last_row = (last_in_seg & (t_grid == T - 1)).reshape(-1)

        output = None
        h_n_list = []
        for layer in range(self.n_layers):
            # 段初始隐状态：只给各行的第一段（行首若是 reset 则取 0），其余从 0 开始
            h_init = h[:, layer, :]
            h_init = torch.where(
                init[:, 0].unsqueeze(-1), torch.zeros_like(h_init), h_init
            )
            h_init_runs = torch.zeros(
                R, self.hidden_size, device=dev, dtype=h_init.dtype
            )
            h_init_runs[seg_base] = h_init

            y = self.grus[layer](x, h_init_runs.unsqueeze(0))[0]  # (R, max_len, H)

            y_flat = y.reshape(R * max_len, self.hidden_size)
            output = y_flat[dest].view(B, T, self.hidden_size)
            h_n_list.append(y_flat[dest[last_row]].unsqueeze(1))

            if layer < self.n_layers - 1:
                x = (
                    F.dropout(y, p=self.dropout, training=self.training)
                    if self.dropout
                    else y
                )

        h_n = torch.cat(h_n_list, dim=1)
        return output, h_n


def get_net(input_size, hidden_size, n_layers, bias, device, dropout, compile):
    gru = GRU(
        input_size,
        hidden_size,
        n_layers=n_layers,
        bias=bias,
        device=device,
        dropout=dropout,
    )
    if compile:
        gru = torch.compile(gru, mode="reduce-overhead")
    return gru


class MultiAgentGRU(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_agents: int,
        device: DEVICE_TYPING,
        centralised: bool,
        share_params: bool,
        n_layers: int,
        dropout: float,
        bias: bool,
        compile: bool,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_agents = n_agents
        self.hidden_size = hidden_size
        self.device = device
        self.centralised = centralised
        self.share_params = share_params
        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.compile = compile

        if self.centralised:
            input_size = input_size * self.n_agents

        agent_networks = [
            get_net(
                input_size=input_size,
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
                bias=self.bias,
                device=self.device,
                dropout=self.dropout,
                compile=self.compile,
            )
            for _ in range(self.n_agents if not self.share_params else 1)
        ]
        self._make_params(agent_networks)

        with torch.device("meta"):
            empty_gru = get_net(
                input_size=input_size,
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
                bias=self.bias,
                device="meta",
                dropout=self.dropout,
                compile=self.compile,
            )
            # Remove all parameters
            TensorDict.from_module(empty_gru).data.to("meta").to_module(empty_gru)
        # [关键] 把 meta 模块移出 nn.Module 的注册表：否则它的 meta 参数会被
        # named_parameters()/state_dict()/collector 的权重搬运看到，而 meta tensor
        # 无法 .to(device)（torchrl SyncDataCollector 建收集器时就会因此报
        # "Cannot copy out of meta tensor"）。真正的权重在 self.params 里，
        # 每次前向用 params.to_module() 热插拔进去。
        object.__setattr__(self, "_empty_gru", empty_gru)

    def forward(
        self,
        input,
        is_init,
        h_0=None,
    ):
        # Input and output always have the multiagent dimension
        # Hidden states always have it apart from when it is centralized and share params
        # is_init never has it

        assert is_init is not None, "We need to pass is_init"
        training = h_0 is None

        missing_batch = False
        if (
            not training and len(input.shape) < 3
        ):  # In evaluation the batch might be missing
            missing_batch = True
            input = input.unsqueeze(0)
            h_0 = h_0.unsqueeze(0)
            is_init = is_init.unsqueeze(0)

        if (
            not training
        ):  # In collection we emulate the sequence dimension and we have the hidden state
            input = input.unsqueeze(1)

        # Check input
        batch = input.shape[0]
        seq = input.shape[1]
        assert input.shape == (batch, seq, self.n_agents, self.input_size)

        if not training:  # Collection
            h_0 = torch.where(
                expand_as_right(is_init, h_0), 0, h_0
            )  # Set hidden to 0 when is_init
            is_init = is_init.unsqueeze(
                1
            )  # If in collection emulate the sequence dimension

        assert is_init.shape == (batch, seq, 1)
        is_init = is_init.unsqueeze(-2).expand(batch, seq, self.n_agents, 1)

        if training:
            if self.centralised and self.share_params:
                shape = (
                    batch,
                    self.n_layers,
                    self.hidden_size,
                )
            else:
                shape = (
                    batch,
                    self.n_agents,
                    self.n_layers,
                    self.hidden_size,
                )
            h_0 = torch.zeros(
                shape,
                device=self.device,
                dtype=torch.float,
            )
        if self.centralised:
            input = input.view(batch, seq, self.n_agents * self.input_size)
            is_init = is_init[..., 0, :]

        output, h_n = self.run_net(input, is_init, h_0)

        if self.centralised and self.share_params:
            output = output.unsqueeze(-2).expand(
                batch, seq, self.n_agents, self.hidden_size
            )

        if not training:
            output = output.squeeze(1)
        if missing_batch:
            output = output.squeeze(0)
            h_n = h_n.squeeze(0)
        return output, h_n

    def run_net(self, input, is_init, h_0):
        # [性能改造] 原来用 torch.vmap 逐 agent 跑（vmap 里走不了 cuDNN 融合 RNN）。
        # 现在改成显式循环：每个 agent 把参数热插拔进 meta 模块单独调用（agent 数只有 1~2）。
        # 实测 meta 热插拔相对真参数无开销（2.85ms vs 2.99ms），且数值完全一致。
        if not self.share_params:
            outputs, h_ns = [], []
            for agent_idx in range(self.n_agents):
                with self.params[agent_idx].to_module(self._empty_gru):
                    if self.centralised:
                        out, h_n = self._empty_gru(input, is_init, h_0[:, agent_idx])
                    else:
                        out, h_n = self._empty_gru(
                            input[..., agent_idx, :],
                            is_init[..., agent_idx, :],
                            h_0[:, agent_idx],
                        )
                outputs.append(out)
                h_ns.append(h_n)
            output = torch.stack(outputs, dim=-2)
            h_n = torch.stack(h_ns, dim=-3)
        else:
            with self.params.to_module(self._empty_gru):
                if self.centralised:
                    output, h_n = self._empty_gru(input, is_init, h_0)
                else:
                    # 把 agent 维折进 batch，一次融合调用（先 permute 让 agent 位在 time 前）
                    batch, seq, n_agents, feat = input.shape
                    out, h_n = self._empty_gru(
                        input.permute(0, 2, 1, 3).reshape(batch * n_agents, seq, feat),
                        is_init.permute(0, 2, 1, 3).reshape(batch * n_agents, seq, 1),
                        h_0.reshape(batch * n_agents, *h_0.shape[2:]),
                    )
                    output = out.view(batch, n_agents, seq, self.hidden_size).permute(
                        0, 2, 1, 3
                    )
                    h_n = h_n.view(batch, n_agents, *h_n.shape[1:])

        return output, h_n

    def vmap_func_module(self, module, *args, **kwargs):
        def exec_module(params, *input):
            with params.to_module(module):
                return module(*input)

        return torch.vmap(exec_module, *args, **kwargs)

    def _make_params(self, agent_networks):
        if self.share_params:
            self.params = TensorDict.from_module(agent_networks[0], as_module=True)
        else:
            self.params = TensorDict.from_modules(*agent_networks, as_module=True)


class Gru(Model):
    r"""A multi-layer Gated Recurrent Unit (GRU) RNN like the one from
    `torch <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`__ .

    The BenchMARL GRU accepts multiple inputs of type array: Tensors of shape ``(*batch,F)``

    Where `F` is the number of features. These arrays will be concatenated along the F dimensions,
    which will be processed to features of `hidden_size` by the GRU.
    
    Args:
        hidden_size (int): The number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        bias (bool): If ``False``, then the GRU layers do not use bias.
        dropout (float): If non-zero, introduces a `Dropout` layer.
        compile (bool): If ``True``, compiles underlying gru model.
        use_input_passthrough (bool): If ``True``, concatenates raw input with GRU output before MLP.
    """

    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        bias: bool,
        dropout: float,
        compile: bool,
        use_input_passthrough: bool, # <--- 接收新参数
        **kwargs,
    ):

        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        self.hidden_state_name = (self.agent_group, f"_hidden_gru_{self.model_index}")
        self.rnn_keys = unravel_key_list(["is_init", self.hidden_state_name])
        self.in_keys += self.rnn_keys

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.compile = compile
        self.use_input_passthrough = use_input_passthrough # <--- 保存配置

        # [P0-C] 角色条件化动作头：role_ids[i] = 第 i 个 agent 的角色编号
        self.role_ids = list(kwargs.pop("role_ids", []) or [])
        self.n_roles = int(kwargs.pop("n_roles", 0))
        self.role_embedding_dim = int(kwargs.pop("role_embedding_dim", 32))
        # [投篮按键] 每个角色各自的输出宽度（默认空 = 全部用 output_features）
        self.role_out_dims = list(kwargs.pop("role_out_dims", []) or [])
        self._role_enabled = (
            len(self.role_ids) > 0
            and self.n_roles > 0
            and len(self.role_ids) == self.n_agents
        )

        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True)]
        )
        self.output_features = self.output_leaf_spec.shape[-1]

        if self.input_has_agent_dim:
            self.gru = MultiAgentGRU(
                self.input_features,
                self.hidden_size,
                self.n_agents,
                self.device,
                bias=self.bias,
                n_layers=self.n_layers,
                centralised=self.centralised,
                share_params=self.share_params,
                dropout=self.dropout,
                compile=self.compile,
            )
        else:
            self.gru = nn.ModuleList(
                [
                    get_net(
                        input_size=self.input_features,
                        hidden_size=self.hidden_size,
                        n_layers=self.n_layers,
                        bias=self.bias,
                        device=self.device,
                        dropout=self.dropout,
                        compile=self.compile,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

        # 【核心逻辑】：根据配置决定 MLP 的输入维度
        if self.use_input_passthrough:
            # 开启时：MLP 输入 = GRU输出(hidden_size) + 原始输入(input_features)
            mlp_input_dim = self.hidden_size + self.input_features
        else:
            # 关闭时：MLP 输入 = GRU输出(hidden_size)
            mlp_input_dim = self.hidden_size

        mlp_net_kwargs = {
            "_".join(k.split("_")[1:]): v
            for k, v in kwargs.items()
            if k.startswith("mlp_")
        }
        
        if self.output_has_agent_dim:
            if self._role_enabled:
                # [P0-C] 角色条件化动作头：共享 trunk + per-role 输出头（loc/scale 按角色分叉）
                hidden = list(mlp_net_kwargs.get("num_cells") or [mlp_input_dim])
                self.mlp = RoleConditionedMLP(
                    in_dim=mlp_input_dim,
                    out_dim=self.output_features,
                    hidden_layers=hidden,
                    role_ids=self.role_ids,
                    n_roles=self.n_roles,
                    role_embedding_dim=self.role_embedding_dim,
                    device=self.device,
                    role_out_dims=self.role_out_dims if len(self.role_out_dims) > 0 else None,
                )
            else:
                self.mlp = MultiAgentMLP(
                    n_agent_inputs=mlp_input_dim, # 使用动态计算的维度
                    n_agent_outputs=self.output_features,
                    n_agents=self.n_agents,
                    centralised=self.centralised,
                    share_params=self.share_params,
                    device=self.device,
                    **mlp_net_kwargs,
                )
        else:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_features=mlp_input_dim, # 使用动态计算的维度
                        out_features=self.output_features,
                        device=self.device,
                        **mlp_net_kwargs,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

    def _perform_checks(self):
        super()._perform_checks()

        input_shape = None
        for input_key, input_spec in self.input_spec.items(True, True):
            if (self.input_has_agent_dim and len(input_spec.shape) == 2) or (
                not self.input_has_agent_dim and len(input_spec.shape) == 1
            ):
                if input_shape is None:
                    input_shape = input_spec.shape[:-1]
                else:
                    if input_spec.shape[:-1] != input_shape:
                        raise ValueError(
                            f"GRU inputs should all have the same shape up to the last dimension, got {self.input_spec}"
                        )
            else:
                raise ValueError(
                    f"GRU input value {input_key} from {self.input_spec} has an invalid shape, maybe you need a CNN?"
                )
        if self.input_has_agent_dim:
            if input_shape[-1] != self.n_agents:
                raise ValueError(
                    "If the GRU input has the agent dimension,"
                    f" the second to last spec dimension should be the number of agents, got {self.input_spec}"
                )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the GRU output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        debug_separator(self.name, "FORWARD START")

        # Gather in_key
        input = torch.cat(
            [
                tensordict.get(in_key)
                for in_key in self.in_keys
                if in_key not in self.rnn_keys
            ],
            dim=-1,
        )
        # [fp16 观测] 统一转 fp32 (环境可能输出半精度观测以节省内存)
        if input.dtype != torch.float32:
            input = input.float()
        debug_print(self.name, "Input tensor", input,
                   f"input_has_agent_dim={self.input_has_agent_dim}, "
                   f"output_has_agent_dim={self.output_has_agent_dim}, "
                   f"share_params={self.share_params}")

        h_0 = tensordict.get(self.hidden_state_name, None)
        is_init = tensordict.get("is_init")
        training = h_0 is None

        # Has multi-agent input dimension
        if self.input_has_agent_dim:
            debug_print(self.name, "Processing with agent dimension", input)
            output, h_n = self.gru(input, is_init, h_0)
            debug_print(self.name, "GRU output (with agent dim)", output)

            if not self.output_has_agent_dim:
                output = output[..., 0, :]
                debug_print(self.name, "Output after removing agent dim", output)
        else:  # Is a global input, this is a critic
            debug_print(self.name, "Processing global input (critic)", input)

            batch = input.shape[0]
            seq = input.shape[1]
            assert input.shape == (batch, seq, self.input_features)
            assert is_init.shape == (batch, seq, 1)

            h_0 = torch.zeros(
                (batch, self.n_layers, self.hidden_size),
                device=self.device,
                dtype=torch.float,
            )
            if self.share_params:
                output, _ = self.gru[0](input, is_init, h_0)
                debug_print(self.name, "GRU output (shared params)", output)
            else:
                outputs = []
                for i, net in enumerate(self.gru):
                    agent_output, _ = net(input, is_init, h_0)
                    debug_print(self.name, f"GRU output for agent {i}", agent_output)
                    outputs.append(agent_output)
                output = torch.stack(outputs, dim=-2)
                debug_print(self.name, "Stacked GRU outputs", output)

        # 【核心逻辑】：执行直通拼接 (Passthrough Concatenation)
        if self.use_input_passthrough:
            input_for_mlp = input
            if self.input_has_agent_dim and not self.output_has_agent_dim:
                 input_for_mlp = input[..., 0, :]

            debug_print(self.name, "Before passthrough concat - GRU output", output)
            debug_print(self.name, "Before passthrough concat - input", input_for_mlp)

            output = torch.cat([output, input_for_mlp], dim=-1)
            debug_print(self.name, "After passthrough concatenation", output)

        # Mlp
        if self.output_has_agent_dim:
            output = self.mlp.forward(output)
            debug_print(self.name, "MLP output (with agent dim)", output)
        else:
            if not self.share_params:
                output = torch.stack(
                    [net(output) for net in self.mlp],
                    dim=-2,
                )
                debug_print(self.name, "MLP output (unshared params)", output)
            else:
                output = self.mlp[0](output)
                debug_print(self.name, "MLP output (shared params)", output)

        debug_print(self.name, "Final output", output)
        debug_separator(self.name, "FORWARD END")

        tensordict.set(self.out_key, output)
        if not training:
            tensordict.set(("next", *self.hidden_state_name), h_n)
        return tensordict


@dataclass
class GruConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Gru`."""

    hidden_size: int = MISSING
    n_layers: int = MISSING
    bias: bool = MISSING
    dropout: float = MISSING
    compile: bool = MISSING
    
    # 【新增配置类字段】
    use_input_passthrough: bool = MISSING

    mlp_num_cells: Sequence[int] = MISSING
    mlp_layer_class: Type[nn.Module] = MISSING
    mlp_activation_class: Type[nn.Module] = MISSING

    # [P0-C] 角色条件化动作头（role_ids[i] = 第 i 个 agent 的角色编号）
    # 注意：dataclass 要求带默认值的字段排在无默认值字段之后
    role_ids: List[int] = field(default_factory=list)
    n_roles: int = 0
    role_embedding_dim: int = 32
    # [投篮按键] per-role 输出宽度（如 [6, 4, 4]：A1 出 4 连续参数 + 2 离散 logits，
    # 其余角色只出 4 个连续参数，补齐 0 后由适配层/layup 忽略）
    role_out_dims: List[int] = field(default_factory=list)

    mlp_activation_kwargs: Optional[dict] = None
    mlp_norm_class: Type[nn.Module] = None
    mlp_norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return Gru

    @property
    def is_rnn(self) -> bool:
        return True

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        spec = Composite(
            {
                f"_hidden_gru_{model_index}": Unbounded(
                    shape=(self.n_layers, self.hidden_size)
                )
            }
        )
        return spec