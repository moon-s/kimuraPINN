"""Loss functions for single-population KimuraPINN workflows."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from src.pde.boundary_conditions import boundary_loss_absorbing
from src.pde.fokker_planck import compute_fokker_planck_residual


def _to_float_tensor(values: torch.Tensor, reference: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)
    if reference is not None:
        values = values.to(device=reference.device, dtype=reference.dtype)
    elif not torch.is_floating_point(values):
        values = values.to(dtype=torch.float32)
    return values


def _normalize_counts(values: torch.Tensor) -> torch.Tensor:
    total = values.sum()
    return values / total.clamp_min(torch.finfo(values.dtype).tiny)


def mse_sfs_loss(predicted: torch.Tensor, observed: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Mean-squared error between predicted and observed folded SFS vectors."""
    pred = _to_float_tensor(predicted)
    obs = _to_float_tensor(observed, pred)
    if pred.shape != obs.shape:
        raise ValueError(f"predicted and observed must have same shape, got {pred.shape} and {obs.shape}")
    if normalize:
        pred = _normalize_counts(pred)
        obs = _normalize_counts(obs)
    return torch.mean((pred - obs).pow(2))


def poisson_nll_sfs_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Poisson negative log likelihood up to the observed-count constant."""
    pred = _to_float_tensor(predicted).clamp_min(eps)
    obs = _to_float_tensor(observed, pred)
    if pred.shape != obs.shape:
        raise ValueError(f"predicted and observed must have same shape, got {pred.shape} and {obs.shape}")
    return torch.mean(pred - obs * torch.log(pred))


def gamma_smoothness_loss(selection_model, t_grid: torch.Tensor) -> torch.Tensor:
    """Delegate gamma smoothness regularization to the selection model."""
    return selection_model.smoothness_loss(t_grid)


def total_single_pop_loss(
    model,
    selection_model,
    x_collocation: torch.Tensor,
    t_collocation: torch.Tensor,
    predicted_sfs: torch.Tensor,
    observed_sfs: torch.Tensor,
    t_boundary: Optional[torch.Tensor] = None,
    t_gamma_grid: Optional[torch.Tensor] = None,
    nu=1.0,
    h: float = 0.5,
    data_loss: str = "mse",
    lambda_pde: float = 1.0,
    lambda_data: float = 1.0,
    lambda_boundary: float = 0.0,
    lambda_gamma: float = 0.0,
    boundary_loss_fn: Callable = boundary_loss_absorbing,
) -> dict[str, torch.Tensor]:
    """Combine PDE, data, boundary, and gamma smoothness losses.

    This helper prepares the pieces needed by future training code without
    implementing an optimizer or training loop.
    """
    residual = compute_fokker_planck_residual(
        model,
        selection_model,
        x_collocation,
        t_collocation,
        nu=nu,
        h=h,
    )
    pde_loss = torch.mean(residual.pow(2))
    if data_loss == "mse":
        sfs_loss = mse_sfs_loss(predicted_sfs, observed_sfs, normalize=True)
    elif data_loss == "poisson":
        sfs_loss = poisson_nll_sfs_loss(predicted_sfs, observed_sfs)
    else:
        raise ValueError("data_loss must be 'mse' or 'poisson'")

    if t_boundary is None:
        boundary = pde_loss * 0.0
    else:
        boundary = boundary_loss_fn(model, t_boundary)

    if t_gamma_grid is None:
        gamma = pde_loss * 0.0
    else:
        gamma = gamma_smoothness_loss(selection_model, t_gamma_grid)

    total = (
        float(lambda_pde) * pde_loss
        + float(lambda_data) * sfs_loss
        + float(lambda_boundary) * boundary
        + float(lambda_gamma) * gamma
    )
    return {
        "total": total,
        "pde": pde_loss,
        "data": sfs_loss,
        "boundary": boundary,
        "gamma_smoothness": gamma,
    }

