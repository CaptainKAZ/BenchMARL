#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Type

import torch
from torch import nn
from tensordict import TensorDictBase
from tensordict.utils import unravel_key_list
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.modules import MLP, MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from benchmarl.utils import DEVICE_TYPING

# 尝试导入 Mamba2
try:
    from mamba_ssm import Mamba2
except ImportError:
    raise ImportError("请先安装 mamba_ssm: `pip install mamba-ssm`")


class MultiAgentMamba(nn.Module):
    """
    适配 BenchMARL 的多智能体 Mamba 模块。
    """
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        headdim: int,
        n_agents: int,
        device: DEVICE_TYPING,
        share_params: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_agents = n_agents
        self.device = device
        self.share_params = share_params
        
        # Mamba2 参数
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.d_inner = expand * d_model
        
        # 检查维度兼容性
        assert self.d_inner % self.headdim == 0, (
            f"Mamba2 维度不匹配: (hidden_size={d_model} * expand={expand}) = {self.d_inner} "
            f"必须能被 headdim={headdim} 整除。请调整 hidden_size 或 headdim。"
        )

        # 如果共享参数，只创建一个 Mamba 层
        if self.share_params:
            self.mamba_layers = nn.ModuleList([
                Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    device=device,
                )
            ])
        else:
            self.mamba_layers = nn.ModuleList([
                Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    device=device,
                )
                for _ in range(self.n_agents)
            ])

    def forward(
        self,
        input: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        is_init: torch.Tensor,
    ):
        # input: (Batch, Sequence, Agents, Features)
        
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        has_agent_dim = input.dim() == 4
        
        output_list = []
        next_conv_state_list = []
        next_ssm_state_list = []

        if self.share_params:
            # === 参数共享模式 ===
            layer = self.mamba_layers[0]
            
            if has_agent_dim:
                # Flatten: (Batch, Seq, Agents, Feat) -> (Batch * Agents, Seq, Feat)
                x = input.reshape(batch_size * self.n_agents, seq_len, -1)
                
                # State Flatten: (Batch, Agents, d_inner, d_conv) -> (Batch * Agents, d_inner, d_conv)
                c_state = conv_state.reshape(batch_size * self.n_agents, self.d_inner, self.d_conv)
                s_state = ssm_state.reshape(batch_size * self.n_agents, self.d_inner, self.d_state)
                
                # Init Mask Handling
                if is_init.dim() == 3 and is_init.shape[-1] == 1:
                    init_mask = is_init.expand(-1, -1, self.n_agents)
                elif is_init.dim() == 4:
                    init_mask = is_init.squeeze(-1)
                else:
                    init_mask = is_init

                init_mask = init_mask.reshape(batch_size * self.n_agents, seq_len)
                
            else:
                x = input
                c_state = conv_state if conv_state.dim() == 3 else conv_state.squeeze(1)
                s_state = ssm_state if ssm_state.dim() == 3 else ssm_state.squeeze(1)
                init_mask = is_init.squeeze(-1)

            # --- Step 模式处理 ---
            if seq_len == 1:
                # Reset State logic
                reset_indices = init_mask.reshape(-1) > 0.5
                if reset_indices.any():
                    c_state = c_state.clone()
                    s_state = s_state.clone()
                    c_state[reset_indices] = 0
                    s_state[reset_indices] = 0

                # === 修复点 1: 直接传入带 seq 维度的 x (B, 1, D) ===
                # Mamba2.step 要求 input 为 (B, 1, D)
                out, next_c, next_s = layer.step(x, c_state, s_state)
                # out 已经是 (B, 1, D)，不需要 unsqueeze
            else:
                # Sequence training
                out = layer(x)
                next_c, next_s = c_state, s_state 

            # 恢复形状
            if has_agent_dim:
                output_list.append(out.view(batch_size, seq_len, self.n_agents, -1))
                next_conv_state_list.append(next_c.view(batch_size, self.n_agents, self.d_inner, self.d_conv))
                next_ssm_state_list.append(next_s.view(batch_size, self.n_agents, self.d_inner, self.d_state))
            else:
                return out, next_c.unsqueeze(1), next_s.unsqueeze(1)

        else:
            # === 不共享参数模式 ===
            if not has_agent_dim:
                raise ValueError("非共享参数模式要求输入必须包含 Agent 维度")

            for i, layer in enumerate(self.mamba_layers):
                x = input[:, :, i, :] 
                c_state = conv_state[:, i, :, :]
                s_state = ssm_state[:, i, :, :] 
                
                if is_init.dim() == 3: 
                     init_mask = is_init.squeeze(-1)
                elif is_init.dim() == 4:
                     init_mask = is_init[:, :, i, 0]

                if seq_len == 1:
                    reset_indices = init_mask > 0.5
                    if reset_indices.any():
                        c_state = c_state.clone()
                        s_state = s_state.clone()
                        c_state[reset_indices] = 0
                        s_state[reset_indices] = 0
                    
                    # === 修复点 2: 同样直接传入 x (B, 1, D) ===
                    out, next_c, next_s = layer.step(x, c_state, s_state)
                    # out 保持 (B, 1, D)
                else:
                    out = layer(x)
                    next_c, next_s = c_state, s_state

                output_list.append(out)
                next_conv_state_list.append(next_c)
                next_ssm_state_list.append(next_s)
            
        if self.share_params and has_agent_dim:
             return output_list[0], next_conv_state_list[0], next_ssm_state_list[0]
        elif not self.share_params:
             output = torch.stack(output_list, dim=2) 
             next_conv_state = torch.stack(next_conv_state_list, dim=1)
             next_ssm_state = torch.stack(next_ssm_state_list, dim=1)
             return output, next_conv_state, next_ssm_state
        
        return output_list[0], next_conv_state_list[0], next_ssm_state_list[0]


class Mamba(Model):
    def __init__(
        self,
        hidden_size: int,
        d_state: int,
        d_conv: int,
        expand: int,
        headdim: int,
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
        
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.d_inner = self.expand * self.hidden_size 

        self.conv_state_key = (self.agent_group, f"_conv_state_mamba_{self.model_index}")
        self.ssm_state_key = (self.agent_group, f"_ssm_state_mamba_{self.model_index}")
        
        self.rnn_keys = unravel_key_list(["is_init", self.conv_state_key, self.ssm_state_key])
        self.in_keys += self.rnn_keys

        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True)]
        )
        self.output_features = self.output_leaf_spec.shape[-1]
        
        # 1. Projection
        self.feature_extractor = nn.Linear(self.input_features, self.hidden_size, device=self.device)
        
        # 2. Mamba
        mamba_share_params = self.share_params
        if not self.input_has_agent_dim: 
            mamba_share_params = True 

        self.mamba = MultiAgentMamba(
            d_model=self.hidden_size,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            headdim=self.headdim,
            n_agents=self.n_agents,
            device=self.device,
            share_params=mamba_share_params,
        )

        # 3. MLP
        mlp_net_kwargs = {
            "_".join(k.split("_")[1:]): v
            for k, v in kwargs.items()
            if k.startswith("mlp_")
        }
        
        self.use_mlp = (self.hidden_size != self.output_features) or len(mlp_net_kwargs) > 0
        
        if self.use_mlp:
            if self.output_has_agent_dim:
                self.mlp = MultiAgentMLP(
                    n_agent_inputs=self.hidden_size,
                    n_agent_outputs=self.output_features,
                    n_agents=self.n_agents,
                    centralised=self.centralised,
                    share_params=self.share_params,
                    device=self.device,
                    **mlp_net_kwargs,
                )
            else:
                 self.mlp = nn.ModuleList([
                    MLP(
                        in_features=self.hidden_size,
                        out_features=self.output_features,
                        device=self.device,
                        **mlp_net_kwargs,
                    ) for _ in range(self.n_agents if not self.share_params else 1)
                ])

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        input = torch.cat(
            [
                tensordict.get(in_key)
                for in_key in self.in_keys
                if in_key not in self.rnn_keys
            ],
            dim=-1,
        )
        
        input = self.feature_extractor(input)
        
        conv_state = tensordict.get(self.conv_state_key, None)
        ssm_state = tensordict.get(self.ssm_state_key, None)
        is_init = tensordict.get("is_init")
        
        if conv_state is None or ssm_state is None:
            batch_size = input.shape[0]
            if self.input_has_agent_dim:
                 n_agents_dim = self.n_agents
            else:
                 n_agents_dim = 1
            
            shape_conv = (batch_size, n_agents_dim, self.d_inner, self.d_conv)
            shape_ssm = (batch_size, n_agents_dim, self.d_inner, self.d_state)
            
            conv_state = torch.zeros(shape_conv, device=self.device, dtype=input.dtype)
            ssm_state = torch.zeros(shape_ssm, device=self.device, dtype=input.dtype)

        # 维度调整
        if input.dim() == 3: 
             input = input.unsqueeze(1)
             if is_init.dim() == 2:
                is_init = is_init.unsqueeze(1)
        elif input.dim() == 2 and not self.input_has_agent_dim:
             input = input.unsqueeze(1)
             if is_init.dim() == 2:
                 is_init = is_init.unsqueeze(1)
        
        output, next_c, next_s = self.mamba(input, conv_state, ssm_state, is_init)

        if self.use_mlp:
             if self.output_has_agent_dim:
                output = self.mlp.forward(output)
             else:
                if not self.share_params:
                     output_stack = []
                     for i, net in enumerate(self.mlp):
                         output_stack.append(net(output[..., i, :]))
                     output = torch.stack(output_stack, dim=-2)
                else:
                    output = self.mlp[0](output)

        if output.shape[1] == 1:
            output = output.squeeze(1)

        tensordict.set(self.out_key, output)
        
        tensordict.set(("next", *self.conv_state_key), next_c)
        tensordict.set(("next", *self.ssm_state_key), next_s)
        
        return tensordict

@dataclass
class MambaConfig(ModelConfig):
    hidden_size: int = MISSING
    d_state: int = MISSING
    d_conv: int = MISSING
    expand: int = MISSING
    headdim: int = MISSING
    
    mlp_num_cells: Sequence[int] = MISSING
    mlp_layer_class: Type[nn.Module] = nn.Linear
    mlp_activation_class: Type[nn.Module] = nn.Tanh
    mlp_activation_kwargs: Optional[dict] = None
    mlp_norm_class: Type[nn.Module] = None
    mlp_norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return Mamba

    @property
    def is_rnn(self) -> bool:
        return True

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        d_inner = self.expand * self.hidden_size
        return Composite(
            {
                f"_conv_state_mamba_{model_index}": Unbounded(shape=torch.Size([d_inner, self.d_conv])),
                f"_ssm_state_mamba_{model_index}": Unbounded(shape=torch.Size([d_inner, self.d_state])),
            }
        )