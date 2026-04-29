"""Two-population Kimura Fokker-Planck residual with migration."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def _call_population_model(model_or_models, t: torch.Tensor, index: int) -> torch.Tensor:
    if isinstance(model_or_models, Sequence):
        return model_or_models[index](t)
    try:
        return model_or_models(t, population=index)
    except TypeError:
        return model_or_models(t)


def compute_two_pop_fokker_planck_residual(
    model: nn.Module,
    selection_models,
    demography_models,
    admixture_model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    h: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute two-population Kimura residual for x=[x1, x2]."""
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"x must have shape [batch, 2], got {tuple(x.shape)}")
    if t.ndim != 2 or t.shape[1] != 1:
        raise ValueError(f"t must have shape [batch, 1], got {tuple(t.shape)}")
    if x.shape[0] != t.shape[0]:
        raise ValueError("x and t must have the same batch size")
    if not (0.0 < eps < 0.5):
        raise ValueError("eps must be in (0, 0.5)")

    x_req = x.detach().clone().clamp(eps, 1.0 - eps).requires_grad_(True)
    t_req = t.detach().clone().requires_grad_(True)
    phi = model(x_req, t_req)
    residual = _grad(phi, t_req)

    for i in range(2):
        xi = x_req[:, i : i + 1]
        gamma_i = _call_population_model(selection_models, t_req, i)
        nu_i = _call_population_model(demography_models, t_req, i)
        if torch.any(nu_i <= 0):
            raise ValueError("demography models must return positive nu(t)")
        drift_i = 2.0 * gamma_i * float(h) * xi * (1.0 - xi)
        variance_i = xi * (1.0 - xi) / nu_i
        drift_grad = _grad(drift_i * phi, x_req)[:, i : i + 1]
        diffusion_grad = _grad(variance_i * phi, x_req)[:, i : i + 1]
        diffusion_grad2 = _grad(diffusion_grad, x_req)[:, i : i + 1]
        residual = residual + drift_grad - 0.5 * diffusion_grad2

    migration = admixture_model().to(device=x_req.device, dtype=x_req.dtype)
    for i in range(2):
        xi = x_req[:, i : i + 1]
        for j in range(2):
            if i == j:
                continue
            xj = x_req[:, j : j + 1]
            migration_flux = (xj - xi) * phi
            migration_grad = _grad(migration_flux, x_req)[:, i : i + 1]
            residual = residual + migration[i, j] * migration_grad
    return residual

