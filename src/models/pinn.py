"""Neural density model for one-population Kimura diffusion PINNs."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Small residual MLP block with smooth activations."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.activation = nn.Tanh()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply a residual nonlinear transformation."""
        return self.activation(inputs + self.net(inputs))


class KimuraPINN(nn.Module):
    """Positive neural approximation to allele-frequency density phi(x, t).

    The model operates on the unfolded one-population domain. Folded SFS logic is
    intentionally kept outside this class.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        fourier_features: int = 0,
        context_dim: int = 0,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if fourier_features < 0:
            raise ValueError("fourier_features must be non-negative")
        if context_dim < 0:
            raise ValueError("context_dim must be non-negative")

        self.fourier_features = int(fourier_features)
        self.context_dim = int(context_dim)
        self.activation = activation if activation is not None else nn.Tanh()

        input_dim = 2 + context_dim
        if self.fourier_features:
            input_dim += 4 * self.fourier_features
            frequencies = torch.arange(1, self.fourier_features + 1, dtype=torch.float32)
            self.register_buffer("fourier_frequencies", frequencies)
        else:
            self.register_buffer("fourier_frequencies", torch.empty(0))

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), self.activation]
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.output_transform = nn.Softplus()

    def _encode(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        """Build model features without clamping or modifying user-provided data."""
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError(f"x must have shape [batch, 1], got {tuple(x.shape)}")
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError(f"t must have shape [batch, 1], got {tuple(t.shape)}")
        if x.shape[0] != t.shape[0]:
            raise ValueError("x and t must have the same batch size")

        features = [x, t]
        if self.fourier_features:
            frequencies = self.fourier_frequencies.to(device=x.device, dtype=x.dtype).view(1, -1)
            x_angles = 2.0 * math.pi * x * frequencies
            t_angles = 2.0 * math.pi * t * frequencies
            features.extend(
                [
                    torch.sin(x_angles),
                    torch.cos(x_angles),
                    torch.sin(t_angles),
                    torch.cos(t_angles),
                ]
            )

        if self.context_dim:
            if context is None:
                raise ValueError("context is required when context_dim > 0")
            if context.ndim != 2 or context.shape != (x.shape[0], self.context_dim):
                raise ValueError(
                    "context must have shape "
                    f"[batch, {self.context_dim}], got {tuple(context.shape)}"
                )
            features.append(context)
        elif context is not None and context.shape[0] != x.shape[0]:
            raise ValueError("context batch size must match x and t")

        return torch.cat(features, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate non-negative density phi(x, t), shape [batch, 1]."""
        encoded = self._encode(x, t, context)
        return self.output_transform(self.net(encoded))

