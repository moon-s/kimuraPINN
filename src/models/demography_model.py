"""Relative effective population size models for one-population Kimura PINNs."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn


class DemographyModel(nn.Module):
    """Positive relative effective population size trajectory nu(t)."""

    def __init__(
        self,
        mode: str = "constant",
        nu: float = 1.0,
        breakpoints: Optional[Sequence[float]] = None,
        values: Optional[Sequence[float]] = None,
        learnable: bool = False,
        min_nu: float = 1e-6,
    ) -> None:
        super().__init__()
        if min_nu <= 0:
            raise ValueError("min_nu must be positive")
        self.mode = mode
        self.min_nu = float(min_nu)

        if mode == "constant":
            if nu <= 0:
                raise ValueError("nu must be positive")
            initial = torch.tensor(float(nu), dtype=torch.float32)
            if learnable:
                self.raw_nu = nn.Parameter(self._inverse_softplus(initial - self.min_nu))
            else:
                self.register_buffer("nu_value", initial)
        elif mode == "epoch":
            if breakpoints is None or values is None:
                raise ValueError("epoch mode requires breakpoints and values")
            if len(breakpoints) != len(values) + 1:
                raise ValueError("epoch mode expects len(breakpoints) == len(values) + 1")
            breakpoint_tensor = torch.tensor(list(breakpoints), dtype=torch.float32)
            if not torch.all(breakpoint_tensor[1:] > breakpoint_tensor[:-1]):
                raise ValueError("breakpoints must be strictly increasing")
            value_tensor = torch.tensor(list(values), dtype=torch.float32)
            if torch.any(value_tensor <= 0):
                raise ValueError("all epoch nu values must be positive")
            self.register_buffer("breakpoints", breakpoint_tensor)
            if learnable:
                self.raw_values = nn.Parameter(self._inverse_softplus(value_tensor - self.min_nu))
            else:
                self.register_buffer("epoch_values", value_tensor)
        else:
            raise ValueError("mode must be 'constant' or 'epoch'")

    @staticmethod
    def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp_min(1e-12)
        return torch.log(torch.expm1(value))

    def forward(self, t: torch.Tensor, population: Optional[str] = None) -> torch.Tensor:
        """Return positive nu(t), shape [batch, 1]."""
        del population
        if t.ndim != 2 or t.shape[1] != 1:
            raise ValueError(f"t must have shape [batch, 1], got {tuple(t.shape)}")

        if self.mode == "constant":
            if hasattr(self, "raw_nu"):
                nu_value = torch.nn.functional.softplus(self.raw_nu) + self.min_nu
            else:
                nu_value = self.nu_value.to(device=t.device, dtype=t.dtype)
            return torch.ones_like(t) * nu_value

        breakpoints = self.breakpoints.to(device=t.device, dtype=t.dtype)
        if hasattr(self, "raw_values"):
            values = torch.nn.functional.softplus(self.raw_values) + self.min_nu
            values = values.to(device=t.device, dtype=t.dtype)
        else:
            values = self.epoch_values.to(device=t.device, dtype=t.dtype)
        t_flat = t.reshape(-1)
        t_clamped = torch.clamp(t_flat, min=breakpoints[0], max=breakpoints[-1])
        right = torch.searchsorted(breakpoints, t_clamped, right=True)
        indices = torch.clamp(right - 1, min=0, max=values.numel() - 1)
        return values[indices].reshape_as(t)

