"""Autograd residuals for one-population Kimura Fokker-Planck dynamics."""

from __future__ import annotations

from typing import Callable, Union

import torch
from torch import nn


NuLike = Union[float, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]


def _as_positive_nu(nu: NuLike, t: torch.Tensor) -> torch.Tensor:
    """Evaluate or broadcast relative effective size nu(t)."""
    if callable(nu):
        nu_value = nu(t)
    elif isinstance(nu, torch.Tensor):
        nu_value = nu.to(device=t.device, dtype=t.dtype)
    else:
        nu_value = torch.tensor(float(nu), device=t.device, dtype=t.dtype)
    if nu_value.ndim == 0:
        nu_value = torch.ones_like(t) * nu_value
    if nu_value.shape != t.shape:
        nu_value = torch.broadcast_to(nu_value, t.shape)
    if torch.any(nu_value <= 0):
        raise ValueError("nu must be positive")
    return nu_value


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Differentiate outputs with respect to inputs, preserving graph."""
    grad = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad


def compute_fokker_planck_residual(
    model: nn.Module,
    selection_model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    nu: NuLike = 1.0,
    h: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute residual phi_t + (M phi)_x - 0.5 (V phi)_xx.

    Endpoint clamping is local to this PDE helper to avoid singular behavior in
    x(1-x) while leaving model inputs semantically unfolded.
    """
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(f"x must have shape [batch, 1], got {tuple(x.shape)}")
    if t.ndim != 2 or t.shape[1] != 1:
        raise ValueError(f"t must have shape [batch, 1], got {tuple(t.shape)}")
    if x.shape[0] != t.shape[0]:
        raise ValueError("x and t must have the same batch size")
    if not (0.0 < eps < 0.5):
        raise ValueError("eps must be in (0, 0.5)")

    x_safe = x.detach().clone().clamp(eps, 1.0 - eps).requires_grad_(True)
    t_req = t.detach().clone().requires_grad_(True)

    phi = model(x_safe, t_req)
    gamma = selection_model(t_req)
    nu_value = _as_positive_nu(nu, t_req)

    drift = 2.0 * gamma * float(h) * x_safe * (1.0 - x_safe)
    variance = x_safe * (1.0 - x_safe) / nu_value

    phi_t = _grad(phi, t_req)
    drift_flux_x = _grad(drift * phi, x_safe)
    diffusion_flux_x = _grad(variance * phi, x_safe)
    diffusion_flux_xx = _grad(diffusion_flux_x, x_safe)
    return phi_t + drift_flux_x - 0.5 * diffusion_flux_xx

