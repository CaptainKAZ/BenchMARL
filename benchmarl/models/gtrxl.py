#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
#  This code is adapted from RLlib and refactored for BenchMARL with fixes
#  for correctness, memory usage, and performance.

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Any, Optional, Type

import torch
from tensordict import TensorDictBase
from tensordict.utils import unravel_key_list, expand_as_right
from torch import nn
from torch.nn import functional as F
from torchrl.data import Composite, Unbounded
from torchrl.modules import MLP

from benchmarl.models.common import Model, ModelConfig
from benchmarl.utils import DEVICE_TYPING

# ######################################################################
# # Helper modules from original GTrXL implementation
# ######################################################################

class SlimFC(nn.Module):
    """A simple fully-connected layer with optional activation and initialization."""
    def __init__(
        self,
        in_size: int,
        out_size: int,
        initializer: Any = None,
        activation_fn: Any = None,
        use_bias: bool = True,
        bias_init: float = 0.0,
        device: DEVICE_TYPING = None,
    ):
        super().__init__()
        layers = []
        linear = nn.Linear(in_size, out_size, bias=use_bias, device=device)
        if initializer:
            initializer(linear.weight)
        else:
            nn.init.xavier_uniform_(linear.weight)
        if use_bias:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)
        if activation_fn is not None:
            if isinstance(activation_fn, type):
                layers.append(activation_fn())
            else:
                layers.append(activation_fn)
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

class GRUGate(nn.Module):
    """A GRU-style gated unit for use in GTrXL."""
    def __init__(self, dim: int, init_bias: float = 2.0, device: DEVICE_TYPING = None):
        super().__init__()
        self._w_r = nn.Linear(dim, dim, bias=True, device=device)
        self._u_r = nn.Linear(dim, dim, bias=False, device=device)
        self._w_z = nn.Linear(dim, dim, bias=True, device=device)
        self._u_z = nn.Linear(dim, dim, bias=False, device=device)
        self._w_g = nn.Linear(dim, dim, bias=True, device=device)
        self._u_g = nn.Linear(dim, dim, bias=False, device=device)
        self.init_bias = init_bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._w_r.weight)
        nn.init.xavier_uniform_(self._u_r.weight)
        nn.init.xavier_uniform_(self._w_z.weight)
        nn.init.xavier_uniform_(self._u_z.weight)
        nn.init.xavier_uniform_(self._w_g.weight)
        nn.init.xavier_uniform_(self._u_g.weight)
        if self.init_bias is not None:
            nn.init.constant_(self._w_r.bias, self.init_bias)
            nn.init.constant_(self._w_z.bias, -self.init_bias)
            nn.init.constant_(self._w_g.bias, 0.0)

    def forward(self, x, h):
        r = torch.sigmoid(self._w_r(x) + self._u_r(h))
        z = torch.sigmoid(self._w_z(x) + self._u_z(h))
        g = torch.tanh(self._w_g(x) + self._u_g(r * h))
        return (1 - z) * h + z * g

class SkipConnection(nn.Module):
    """Skip connection module."""
    def __init__(self, module: nn.Module, *, fan_in_layer: nn.Module = None):
        super().__init__()
        self.module = module
        self.fan_in_layer = fan_in_layer

    def forward(self, x, *args, **kwargs):
        y = self.module(x, *args, **kwargs)
        if self.fan_in_layer is not None:
            return self.fan_in_layer(y, x)
        else:
            return y + x

class RelativeMultiHeadAttention(nn.Module):
    """A relative multi-head attention module that accepts an attention mask."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        head_dim: int,
        input_layernorm: bool = False,
        output_activation: Optional[Type[nn.Module]] = None,
        device: DEVICE_TYPING = None,
    ):
        super().__init__()
        self._q = SlimFC(in_dim, num_heads * head_dim, use_bias=False, device=device)
        self._k = SlimFC(in_dim, num_heads * head_dim, use_bias=False, device=device)
        self._v = SlimFC(in_dim, num_heads * head_dim, use_bias=False, device=device)
        self._pos_proj = SlimFC(in_dim, num_heads * head_dim, use_bias=False, device=device)
        self._u = nn.Parameter(torch.zeros(num_heads, head_dim, device=device))
        self._v_2 = nn.Parameter(torch.zeros(num_heads, head_dim, device=device))
        self._out = SlimFC(num_heads * head_dim, out_dim, use_bias=False, device=device)
        self._num_heads = num_heads
        self._head_dim = head_dim
        self.input_layernorm = nn.LayerNorm(in_dim, device=device) if input_layernorm else None
        self.output_activation = output_activation() if output_activation else None

    def forward(self, x: torch.Tensor, memory: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if self.input_layernorm is not None:
            x = self.input_layernorm(x)

        full_input = torch.cat([memory, x], dim=1)
        q = self._q(x)
        k = self._k(full_input)
        v = self._v(full_input)

        q = q.view(q.shape[0], q.shape[1], self._num_heads, self._head_dim)
        k = k.view(k.shape[0], k.shape[1], self._num_heads, self._head_dim)
        v = v.view(v.shape[0], v.shape[1], self._num_heads, self._head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        T_q = q.shape[2]
        T_k = k.shape[2]

        content_score = torch.matmul(q + self._u.unsqueeze(1), k.transpose(-1, -2))
        pos_embed = self._pos_proj(x).view(x.shape[0], x.shape[1], self._num_heads, self._head_dim).permute(0, 2, 1, 3)
        pos_score = torch.matmul(q + self._v_2.unsqueeze(1), pos_embed.transpose(-1, -2))
        pos_score = self._rel_shift(pos_score, T_q)

        if T_k > T_q:
            pad_amount = T_k - T_q
            pos_score = F.pad(pos_score, (pad_amount, 0), "constant", 0)

        attn_score = (content_score + pos_score) / (self._head_dim ** 0.5)

        if attn_mask is not None:
            fill_value = torch.finfo(attn_score.dtype).min
            attn_score = attn_score.masked_fill_(~attn_mask.unsqueeze(1), fill_value)
        
        attn_score = F.softmax(attn_score, dim=-1)

        attn_out = torch.matmul(attn_score, v)
        permuted_attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        current_seq_len = permuted_attn_out.shape[1]
        attn_out = permuted_attn_out.view(x.shape[0], current_seq_len, self._num_heads * self._head_dim)
        
        out = self._out(attn_out)
        if self.output_activation:
            out = self.output_activation(out)
        return out

    def _rel_shift(self, x, T_q):
        B, H, T_q_shape, T_k = x.shape
        x_padded = F.pad(x, (0, 1))
        x_padded = x_padded.view(B, H, T_q_shape * (T_k + 1))
        x_padded = x_padded.view(B, H, T_k + 1, T_q_shape)
        x_shifted = x_padded[:, :, 1:]
        x_shifted = x_shifted.view(B, H, T_k * T_q_shape)
        x_shifted = x_shifted.view(B, H, T_q_shape, T_k)
        return x_shifted

# ######################################################################
# # BenchMARL Integration
# ######################################################################

@dataclass
class GTrXLConfig(ModelConfig):
    """Config for a Gated Transformer-XL."""
    num_transformer_units: int = MISSING
    attention_dim: int = MISSING
    num_heads: int = MISSING
    head_dim: int = MISSING
    memory_training: int = MISSING
    position_wise_mlp_dim: int = MISSING
    init_gru_gate_bias: float = 2.0
    dropout: float = 0.0
    
    mlp_activation_class: Type[nn.Module] = nn.ReLU
    mlp_layer_class: Type[nn.Module] = nn.Linear
    
    @staticmethod
    def associated_class():
        return GTrXL

    @property
    def is_rnn(self) -> bool:
        return True

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        spec = Composite({
            f"_memory_gtrxl_{model_index}": Unbounded(
                shape=(self.num_transformer_units, self.memory_training, self.attention_dim)
            )
        })
        return spec

class GTrXL(Model):
    """A Gated Transformer-XL network (GTrXL) adapted for BenchMARL."""

    def __init__(
        self,
        num_transformer_units: int,
        attention_dim: int,
        num_heads: int,
        head_dim: int,
        memory_training: int,
        position_wise_mlp_dim: int,
        init_gru_gate_bias: float,
        dropout: float,
        mlp_layer_class: Type[nn.Module],
        mlp_activation_class: Type[nn.Module],
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
        
        self.memory_key = (self.agent_group, f"_memory_gtrxl_{self.model_index}")
        self.rnn_keys = unravel_key_list(["is_init", self.memory_key])
        self.in_keys = list(self.input_spec.keys(True, True)) + self.rnn_keys

        self.input_features = sum(
            [spec.shape[-1] for spec in self.input_spec.values(True, True) if spec.shape]
        )
        self.output_features = self.output_leaf_spec.shape[-1]
        self.attention_dim = attention_dim
        self.num_transformer_units = num_transformer_units
        self.memory_len = memory_training
        
        self.linear_layer = SlimFC(
            in_size=self.input_features, out_size=self.attention_dim, device=self.device
        )

        attention_layers = []
        for _ in range(self.num_transformer_units):
            mha_layer = SkipConnection(
                RelativeMultiHeadAttention(
                    in_dim=self.attention_dim,
                    out_dim=self.attention_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    input_layernorm=True,
                    output_activation=mlp_activation_class,
                    device=self.device,
                ),
                fan_in_layer=GRUGate(self.attention_dim, init_gru_gate_bias, device=self.device),
            )
            
            ff_mlp = MLP(
                in_features=self.attention_dim,
                out_features=self.attention_dim,
                num_cells=[position_wise_mlp_dim],
                layer_class=mlp_layer_class,
                activation_class=mlp_activation_class,
                device=self.device,
            )
            ff_layer = nn.Sequential(nn.LayerNorm(self.attention_dim, device=self.device), ff_mlp, nn.Dropout(dropout))
            
            e_layer = SkipConnection(
                ff_layer,
                fan_in_layer=GRUGate(self.attention_dim, init_gru_gate_bias, device=self.device),
            )
            attention_layers.extend([mha_layer, e_layer])
        
        self.attention_layers = nn.ModuleList(attention_layers)
        
        self.logits_layer = SlimFC(
            in_size=self.attention_dim, out_size=self.output_features, device=self.device
        )

        if self.is_critic:
            self.value_head = SlimFC(
                in_size=self.attention_dim, out_size=1, device=self.device
            )

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # --- 1. Data Extraction and Dimension Handling ---
        input_keys = [key for key in self.in_keys if key not in self.rnn_keys]
        input_data = torch.cat([tensordict.get(key) for key in input_keys], dim=-1)
        is_init = tensordict.get("is_init")
        memory = tensordict.get(self.memory_key, None)

        training = memory is None
        missing_batch = not training and input_data.dim() < 3

        if missing_batch:
            input_data, memory, is_init = (
                t.unsqueeze(0) for t in (input_data, memory, is_init)
            )
        if not training:
            input_data, is_init = (t.unsqueeze(1) for t in (input_data, is_init))

        B, T, A, _ = input_data.shape

        # --- 2. Initialize Memory & Flatten Dimensions ---
        if training:
            mem_shape = (B, A, self.num_transformer_units, self.memory_len, self.attention_dim)
            memory = torch.zeros(mem_shape, device=self.device)

        flat_input = input_data.permute(0, 2, 1, 3).reshape(B * A, T, self.input_features)
        flat_memory = memory.permute(0, 1, 2, 3, 4).reshape(B * A, self.num_transformer_units, self.memory_len, self.attention_dim)
        flat_is_init = is_init.unsqueeze(2).expand(B, T, A, 1).permute(0, 2, 1, 3).reshape(B * A, T, 1)

        # # [FIX] Ensure all tensors for mask creation are on the correct device
        # flat_is_init = flat_is_init.to(self.device)
        # # Move other input-derived tensors to the correct device as well
        # flat_input = flat_input.to(self.device)
        # flat_memory = flat_memory.to(self.device)


        # --- 3. Create Attention Mask for Parallel Processing ---
        M = self.memory_len
        episode_ids = torch.cumsum(flat_is_init.squeeze(-1), dim=1)
        mem_ids = episode_ids[:, 0:1].expand(-1, M)
        full_episode_ids = torch.cat([mem_ids, episode_ids], dim=1)
        episode_mask = full_episode_ids.unsqueeze(1) == full_episode_ids.unsqueeze(2)
        causal_mask = torch.tril(torch.ones(M + T, M + T, device=episode_mask.device, dtype=torch.bool))
        final_attn_mask = (episode_mask & causal_mask)[..., -T:, :]

        # --- 4. GTrXL Core Logic (Fully Parallel) ---
        x = self.linear_layer(flat_input)
        
        new_memory_units = []
        for i in range(self.num_transformer_units):
            memory_unit_in = flat_memory[:, i, :, :]
            mha_layer = self.attention_layers[i * 2]
            ff_layer = self.attention_layers[i * 2 + 1]
            
            x = mha_layer(x, memory=memory_unit_in, attn_mask=final_attn_mask)
            x = ff_layer(x)
            
            new_memory_units.append(x)
        
        final_features = x
        logits = self.logits_layer(final_features)
        
        new_memory_stacked = torch.stack(new_memory_units, dim=1)
        updated_mem_history = torch.cat([flat_memory, new_memory_stacked], dim=2)
        next_memory_flat = updated_mem_history[:, :, -M:, :]

        # --- 5. Reshape and Store Outputs ---
        logits = logits.view(B, A, T, self.output_features).permute(0, 2, 1, 3)
        next_memory = next_memory_flat.view(B, A, self.num_transformer_units, M, self.attention_dim)

        if not self.output_has_agent_dim:
            logits = logits[:, :, 0, ...]
        
        if not training:
            logits = logits.squeeze(1)
        if missing_batch:
            logits = logits.squeeze(0)
            next_memory = next_memory.squeeze(0)

        tensordict.set(self.out_key, logits)
        
        if self.is_critic:
            value = self.value_head(final_features)
            value = value.view(B, A, T, 1).permute(0, 2, 1, 3)
            if self.centralised:
                value = value.mean(dim=2, keepdim=True)
            if not training:
                value = value.squeeze(1)
            if missing_batch:
                value = value.squeeze(0)
            tensordict.set("state_value", value.squeeze(-1))

        if not training:
            tensordict.set(("next", *self.memory_key), next_memory)

        return tensordict