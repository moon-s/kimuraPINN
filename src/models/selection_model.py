"""Selection coefficient models for Kimura diffusion."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn


class SelectionModel(nn.Module):
    """Time-dependent scaled selection coefficient gamma(t)."""

    def __init__(
        self,
        mode: str = "constant",
        gamma: float = 0.0,
        learnable: bool = True,
        breakpoints: Optional[Sequence[float]] = None,
        values: Optional[Sequence[float]] = None,
        hidden_dim: int = 32,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "constant":
            initial = torch.tensor(float(gamma), dtype=torch.float32)
            if learnable:
                self.gamma = nn.Parameter(initial)
            else:
                self.register_buffer("gamma", initial)
        elif mode == "piecewise_linear":
            if breakpoints is None or values is None:
                raise ValueError("piecewise_linear mode requires breakpoints and values")
            if len(breakpoints) != len(values):
                raise ValueError("breakpoints and values must have the same length")
            if len(breakpoints) < 2:
                raise ValueError("piecewise_linear mode requires at least two breakpoints")
            breakpoint_tensor = torch.tensor(list(breakpoints), dtype=torch.float32)
            if not torch.all(breakpoint_tensor[1:] > breakpoint_tensor[:-1]):
                raise ValueError("breakpoints must be strictly increasing")
            value_tensor = torch.tensor(list(values), dtype=torch.float32)
            self.register_buffer("breakpoints", breakpoint_tensor)
            if learnable:
                self.values = nn.Parameter(value_tensor)
            else:
                self.register_buffer("values", value_tensor)
        elif mode == "neural_network":
            if hidden_dim <= 0 or num_layers <= 0:
                raise ValueError("hidden_dim and num_layers must be positive")
            layers: list[nn.Module] = [nn.Linear(1, hidden_dim), nn.Tanh()]
            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
            layers.append(nn.Linear(hidden_dim, 1))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(
                "mode must be one of 'constant', 'piecewise_linear', or 'neural_network'"
            )

    def forward(
        self,
        t: torch.Tensor,
        population: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return gamma(t) with shape [batch, 1]."""
        del population, context
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError(f"t must have shape [batch, 1], got {tuple(t.shape)}")

        if self.mode == "constant":
            return t * 0.0 + self.gamma.to(device=t.device, dtype=t.dtype).view(1, 1)
        if self.mode == "piecewise_linear":
            return self._piecewise_linear(t)
        return self.net(t)

    def _piecewise_linear(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate continuous linear interpolation on configured breakpoints."""
        breakpoints = self.breakpoints.to(device=t.device, dtype=t.dtype)
        values = self.values.to(device=t.device, dtype=t.dtype)
        t_flat = t.reshape(-1)
        t_clamped = torch.clamp(t_flat, min=breakpoints[0], max=breakpoints[-1])
        right = torch.searchsorted(breakpoints, t_clamped, right=True)
        right = torch.clamp(right, min=1, max=breakpoints.numel() - 1)
        left = right - 1
        t0 = breakpoints[left]
        t1 = breakpoints[right]
        y0 = values[left]
        y1 = values[right]
        weight = (t_clamped - t0) / (t1 - t0)
        return (y0 + weight * (y1 - y0)).reshape_as(t)

    def smoothness_loss(self, t_grid: torch.Tensor) -> torch.Tensor:
        """Compute mean squared derivative dgamma/dt on a time grid."""
        if t_grid.ndim != 2 or t_grid.shape[1] != 1:
            raise ValueError(f"t_grid must have shape [batch, 1], got {tuple(t_grid.shape)}")
        t_req = t_grid.detach().clone().requires_grad_(True)
        gamma = self.forward(t_req)
        grad = torch.autograd.grad(
            gamma.sum(),
            t_req,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            return gamma.sum() * 0.0
        return torch.mean(grad.pow(2))

