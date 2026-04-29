"""Marginal folded SFS projection for two-population densities."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from src.sfs.projection import (
    MAX_FULL_PROJECTION_N,
    fold_sfs,
    make_quadrature_grid,
    project_density_to_observed_k,
    project_density_to_sfs,
)


def _project_marginal(
    marginal_phi: torch.Tensor,
    grid: torch.Tensor,
    weights: torch.Tensor,
    n_pop: int,
    observed_k: Optional[torch.Tensor],
) -> torch.Tensor:
    if observed_k is not None:
        return project_density_to_observed_k(
            marginal_phi,
            grid,
            weights,
            n=n_pop,
            k_values=observed_k.to(device=marginal_phi.device),
            normalize=True,
            folded=True,
        )
    if n_pop > MAX_FULL_PROJECTION_N:
        raise ValueError("observed_k is required for large n_pop marginal projection")
    return fold_sfs(project_density_to_sfs(marginal_phi, grid, weights, n=n_pop, normalize=True))


def project_two_pop_density_to_marginal_folded_sfs(
    model: nn.Module,
    t_eval: float | torch.Tensor,
    n_pop_1: int,
    n_pop_2: int,
    n_grid: int = 128,
    observed_k_1: Optional[torch.Tensor] = None,
    observed_k_2: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project phi(x1,x2,t) to marginal folded SFS for each population."""
    model_device = device
    if model_device is None:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
    x_grid, weights = make_quadrature_grid(n_points=n_grid, device=model_device)
    x1, x2 = torch.meshgrid(x_grid, x_grid, indexing="ij")
    points = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=1)
    if isinstance(t_eval, torch.Tensor):
        t_scalar = t_eval.to(device=model_device, dtype=x_grid.dtype).reshape(-1)[0]
    else:
        t_scalar = torch.tensor(float(t_eval), device=model_device, dtype=x_grid.dtype)
    t = torch.full((points.shape[0], 1), t_scalar, device=model_device, dtype=x_grid.dtype)
    phi = model(points, t).reshape(n_grid, n_grid)

    marginal_1 = torch.sum(phi * weights.view(1, -1), dim=1)
    marginal_2 = torch.sum(phi * weights.view(-1, 1), dim=0)
    sfs_1 = _project_marginal(marginal_1, x_grid, weights, n_pop_1, observed_k_1)
    sfs_2 = _project_marginal(marginal_2, x_grid, weights, n_pop_2, observed_k_2)
    return sfs_1, sfs_2

