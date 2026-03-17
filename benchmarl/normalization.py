#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Reward normalization utilities for MARL algorithms."""

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizerMode(Enum):
    """Enumeration of normalization modes."""

    PER_GROUP = "per_group"
    PER_AGENT = "per_agent"


class Normalizer(nn.Module, ABC):
    """Abstract base class for reward normalizers.

    Normalizers are responsible for computing running statistics of rewards
    and applying normalization transformations. They support checkpointing
    through state_dict/load_state_dict methods.
    """

    mode: NormalizerMode

    @abstractmethod
    def update(self, rewards: torch.Tensor, **kwargs) -> None:
        """Update normalization statistics with new reward data."""
        pass

    @abstractmethod
    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using computed statistics."""
        pass

    @abstractmethod
    def denormalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Denormalize rewards (inverse of normalize)."""
        pass

    @abstractmethod
    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Return the state dict for checkpointing."""
        pass

    @abstractmethod
    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load normalizer state from a state dict."""
        pass


class RunningMeanStd(Normalizer):
    """Running mean and standard deviation normalizer using EMA.

    This normalizer maintains exponential moving average statistics of rewards
    and normalizes using the running mean and standard deviation.

    Args:
        decay: EMA decay factor (default: 0.99). Higher values = slower adaptation.
        eps: Small constant for numerical stability (default: 1e-8).
        device: Device for tensors (default: 'cpu').
    """

    def __init__(
        self,
        decay: float = 0.99,
        eps: float = 1e-8,
        device: str = "cpu",
    ):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.device = device

        # Register as buffers (saved in state_dict but not trained via gradient descent)
        # mean: running mean of rewards
        # mean_sq: running mean of squared rewards (used to compute variance)
        # count: number of updates (for potential bias correction)
        self.register_buffer("mean", torch.zeros(1, device=device))
        self.register_buffer("mean_sq", torch.ones(1, device=device))
        self.register_buffer("count", torch.tensor(0.0, device=device))

    def update(self, rewards: torch.Tensor, **kwargs) -> None:
        """Update running statistics with new reward data using EMA.

        Args:
            rewards: Tensor of rewards to update statistics with.
            **kwargs: Additional arguments (unused, for API compatibility).
        """
        # Move to correct device if needed
        rewards = rewards.to(self.device)

        # Compute batch statistics
        batch_mean = rewards.mean()
        batch_mean_sq = (rewards**2).mean()

        # EMA update
        self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
        self.mean_sq = self.decay * self.mean_sq + (1 - self.decay) * batch_mean_sq
        self.count += 1

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using running mean and standard deviation.

        Args:
            rewards: Tensor of rewards to normalize.

        Returns:
            Normalized rewards tensor.
        """
        rewards = rewards.to(self.device)
        # Variance = E[x^2] - E[x]^2
        variance = self.mean_sq - self.mean**2
        std = torch.sqrt(variance + self.eps)
        return (rewards - self.mean) / std

    def denormalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Denormalize rewards (inverse of normalize).

        Args:
            rewards: Tensor of normalized rewards to denormalize.

        Returns:
            Denormalized rewards tensor.
        """
        rewards = rewards.to(self.device)
        variance = self.mean_sq - self.mean**2
        std = torch.sqrt(variance + self.eps)
        return rewards * std + self.mean

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Return the state dict for checkpointing.

        Returns:
            Dictionary containing mean, mean_sq, count, and config.
        """
        state = nn.Module.state_dict(self, *args, **kwargs)
        state["decay"] = self.decay
        state["eps"] = self.eps
        state["device"] = self.device
        return state

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load normalizer state from a state dict.

        Args:
            state_dict: Dictionary containing saved state.
            strict: If True, raise error on missing/unexpected keys (default: True).
            assign: If True, assign tensors directly instead of copying (default: False).
        """
        state_dict = dict(state_dict)
        self.decay = state_dict.pop("decay", self.decay)
        self.eps = state_dict.pop("eps", self.eps)
        self.device = state_dict.pop("device", self.device)

        return nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)


class PopArt(Normalizer):
    """PopArt normalizer for value function prediction with stable outputs.

    PopArt (Preserving Outputs Precisely while Adaptively Rescaling Targets) maintains
    running statistics of target values and automatically adjusts the linear layer weights
    to keep the output stable when statistics change.

    When the mean/variance of target values change, the weights are adapted so that:
    - The output distribution remains approximately stable
    - The network doesn't need to relearn the value scale

    This is particularly useful for value-based methods like MAPPO where value targets
    can shift dramatically during training.

    Args:
        input_shape: Size of the input features.
        output_shape: Size of the output (default: 1 for scalar value).
        norm_axes: Number of leading axes to normalize over (default: 1).
        beta: EMA decay factor for running statistics (default: 0.99999).
        epsilon: Small constant for numerical stability (default: 1e-5).
        device: Device for tensors (default: 'cpu').
    """

    def __init__(
        self,
        input_shape: int,
        output_shape: int = 1,
        norm_axes: int = 1,
        beta: float = 0.99999,
        epsilon: float = 1e-5,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.norm_axes = norm_axes
        self.beta = beta
        self.epsilon = epsilon
        self.device = device

        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape))
        self.bias = nn.Parameter(torch.Tensor(output_shape))

        self.register_buffer("mean", torch.zeros(output_shape))
        self.register_buffer("mean_sq", torch.zeros(output_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))
        self.register_buffer("stddev", torch.ones(output_shape))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters with Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()
        self.stddev.fill_(1.0)

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation to input.

        Args:
            input_vector: Input tensor of shape [*, input_shape].

        Returns:
            Output tensor of shape [*, output_shape].
        """
        input_vector = input_vector.to(self.device)
        return F.linear(input_vector, self.weight, self.bias)

    def _debiased_mean_var(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute debiased mean and variance from running statistics.

        Returns:
            Tuple of (debiased_mean, debiased_variance).
        """
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, rewards: torch.Tensor, **kwargs) -> None:
        """Update running statistics and adapt weights for output stability.

        This method:
        1. Computes batch statistics from target values
        2. Updates running mean/variance with EMA
        3. Adapts linear layer weights to maintain stable outputs

        The weight adaptation ensures that when mean/variance change,
        the output distribution remains approximately constant:
        - new_weight = old_weight * old_std / new_std
        - new_bias = (old_std * old_bias + old_mean - new_mean) / new_std

        Args:
            rewards: Tensor of target values to update statistics with.
            **kwargs: Additional arguments (unused, for API compatibility).
        """
        rewards = rewards.to(self.device)

        old_mean, old_var = self._debiased_mean_var()
        old_stddev = torch.sqrt(old_var)

        batch_mean = rewards.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (rewards**2).mean(dim=tuple(range(self.norm_axes)))

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        new_mean, new_var = self._debiased_mean_var()
        new_stddev = torch.sqrt(new_var)
        self.stddev.copy_(new_stddev)

        weight_scale = old_stddev / new_stddev
        self.weight.data.mul_(weight_scale.unsqueeze(1))
        self.bias.data.mul_(old_stddev).add_(old_mean - new_mean).div_(new_stddev)

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize tensor using running statistics.

        Args:
            rewards: Tensor to normalize.

        Returns:
            Normalized tensor: (tensor - mean) / sqrt(var + eps)
        """
        rewards = rewards.to(self.device)
        mean, var = self._debiased_mean_var()
        broadcast_shape = [1] * self.norm_axes + [self.output_shape]
        mean_b = mean.view(broadcast_shape)
        std_b = torch.sqrt(var).view(broadcast_shape)
        return (rewards - mean_b) / std_b

    def denormalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor using running statistics.

        Args:
            rewards: Normalized tensor to denormalize.

        Returns:
            Denormalized tensor: tensor * sqrt(var + eps) + mean
        """
        rewards = rewards.to(self.device)
        mean, var = self._debiased_mean_var()
        broadcast_shape = [1] * self.norm_axes + [self.output_shape]
        mean_b = mean.view(broadcast_shape)
        std_b = torch.sqrt(var).view(broadcast_shape)
        return rewards * std_b + mean_b

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Return the state dict for checkpointing.

        Returns:
            Dictionary containing all parameters, buffers, and configuration.
        """
        state = nn.Module.state_dict(self, *args, **kwargs)
        state["input_shape"] = self.input_shape
        state["output_shape"] = self.output_shape
        state["norm_axes"] = self.norm_axes
        state["beta"] = self.beta
        state["epsilon"] = self.epsilon
        state["device"] = self.device
        return state

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load normalizer state from a state dict.

        Args:
            state_dict: Dictionary containing saved state.
            strict: If True, raise error on missing/unexpected keys (default: True).
            assign: If True, assign tensors directly instead of copying (default: False).
        """
        state_dict = dict(state_dict)
        self.input_shape = state_dict.pop("input_shape", self.input_shape)
        self.output_shape = state_dict.pop("output_shape", self.output_shape)
        self.norm_axes = state_dict.pop("norm_axes", self.norm_axes)
        self.beta = state_dict.pop("beta", self.beta)
        self.epsilon = state_dict.pop("epsilon", self.epsilon)
        self.device = state_dict.pop("device", self.device)

        return nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)
