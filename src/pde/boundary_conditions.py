"""Boundary condition losses for one-population Kimura diffusion."""

from __future__ import annotations

from typing import Callable, Union

import torch
from torch import nn

from src.pde.fokker_planck import _as_positive_nu


NuLike = Union[float, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]


def boundary_loss_absorbing(model: nn.Module, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Penalize nonzero density near absorbing boundaries x=0 and x=1."""
    if t.ndim != 2 or t.shape[1] != 1:
        raise ValueError(f"t must have shape [batch, 1], got {tuple(t.shape)}")
    x_left = torch.full_like(t, eps)
    x_right = torch.full_like(t, 1.0 - eps)
    phi_left = model(x_left, t)
    phi_right = model(x_right, t)
    return torch.mean(phi_left.pow(2) + phi_right.pow(2))


def boundary_loss_no_flux_optional(
    model: nn.Module,
    selection_model: nn.Module,
    t: torch.Tensor,
    nu: NuLike = 1.0,
    h: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Optional no-flux penalty evaluated near x=0 and x=1.

    The flux is J = M phi - 0.5 d(V phi)/dx; this helper is available for
    experiments but is not required by the initial tests.
    """
    if t.ndim != 2 or t.shape[1] != 1:
        raise ValueError(f"t must have shape [batch, 1], got {tuple(t.shape)}")

    losses = []
    for x_value in (eps, 1.0 - eps):
        x = torch.full_like(t, x_value).detach().clone().requires_grad_(True)
        t_req = t.detach().clone().requires_grad_(True)
        phi = model(x, t_req)
        gamma = selection_model(t_req)
        nu_value = _as_positive_nu(nu, t_req)
        drift = 2.0 * gamma * float(h) * x * (1.0 - x)
        variance = x * (1.0 - x) / nu_value
        diff_grad = torch.autograd.grad(
            variance * phi,
            x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True,
        )[0]
        flux = drift * phi - 0.5 * diff_grad
        losses.append(torch.mean(flux.pow(2)))
    return sum(losses)

