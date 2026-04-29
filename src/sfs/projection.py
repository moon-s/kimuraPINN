"""Project continuous allele-frequency densities to count-based SFS objects."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import torch
from torch import nn


MAX_FULL_PROJECTION_N = 20_000


def make_quadrature_grid(
    n_points: int = 512,
    eps: float = 1e-6,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return trapezoid quadrature nodes and weights over (eps, 1 - eps)."""
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if not (0.0 <= eps < 0.5):
        raise ValueError("eps must be in [0, 0.5)")
    dtype = dtype if dtype is not None else torch.float32
    width = 1.0 - 2.0 * eps
    if n_points == 1:
        x_grid = torch.tensor([0.5], device=device, dtype=dtype)
        weights = torch.tensor([width], device=device, dtype=dtype)
        return x_grid, weights
    x_grid = torch.linspace(eps, 1.0 - eps, n_points, device=device, dtype=dtype)
    step = width / float(n_points - 1)
    weights = torch.full((n_points,), step, device=device, dtype=dtype)
    weights[0] = step / 2.0
    weights[-1] = step / 2.0
    return x_grid, weights


def _as_column_free_vector(values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape [n_grid] or [n_grid, 1]")
    return values


def binomial_log_prob(k: Union[int, torch.Tensor], n: int, x: torch.Tensor) -> torch.Tensor:
    """Compute stable log Binomial(k; n, x) with broadcasting support."""
    if n < 0:
        raise ValueError("n must be non-negative")
    k_tensor = torch.as_tensor(k, device=x.device, dtype=x.dtype)
    n_tensor = torch.as_tensor(float(n), device=x.device, dtype=x.dtype)
    x_safe = x.clamp(1e-12, 1.0 - 1e-12)
    log_comb = (
        torch.lgamma(n_tensor + 1.0)
        - torch.lgamma(k_tensor + 1.0)
        - torch.lgamma(n_tensor - k_tensor + 1.0)
    )
    return log_comb + k_tensor * torch.log(x_safe) + (n_tensor - k_tensor) * torch.log1p(-x_safe)


def _normalize_if_requested(values: torch.Tensor, normalize: bool) -> torch.Tensor:
    if not normalize:
        return values
    total = values.sum()
    return values / total.clamp_min(torch.finfo(values.dtype).tiny)


def project_density_to_sfs(
    phi_values: torch.Tensor,
    x_grid: torch.Tensor,
    weights: torch.Tensor,
    n: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Project density values to unfolded expected SFS for k=0,...,n."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if n > MAX_FULL_PROJECTION_N:
        raise ValueError(
            f"Full projection for n={n} would allocate n+1 classes; use "
            "project_density_to_observed_k for sparse projection."
        )
    phi = _as_column_free_vector(phi_values, "phi_values")
    x = _as_column_free_vector(x_grid, "x_grid").to(device=phi.device, dtype=phi.dtype)
    w = _as_column_free_vector(weights, "weights").to(device=phi.device, dtype=phi.dtype)
    if phi.shape != x.shape or phi.shape != w.shape:
        raise ValueError("phi_values, x_grid, and weights must have the same length")

    k_values = torch.arange(n + 1, device=phi.device, dtype=phi.dtype).view(-1, 1)
    log_kernel = binomial_log_prob(k_values, n, x.view(1, -1))
    log_weighted_phi = torch.log((phi * w).clamp_min(torch.finfo(phi.dtype).tiny)).view(1, -1)
    unfolded = torch.exp(torch.logsumexp(log_kernel + log_weighted_phi, dim=1))
    return _normalize_if_requested(unfolded, normalize)


def project_density_to_observed_k(
    phi_values: torch.Tensor,
    x_grid: torch.Tensor,
    weights: torch.Tensor,
    n: int,
    k_values: torch.Tensor,
    normalize: bool = True,
    folded: bool = True,
) -> torch.Tensor:
    """Project density only to requested count classes.

    When folded=True, k_values are interpreted as folded classes
    1..floor(n/2), and the complementary n-k contribution is added except for
    the even-n midpoint.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    phi = _as_column_free_vector(phi_values, "phi_values")
    x = _as_column_free_vector(x_grid, "x_grid").to(device=phi.device, dtype=phi.dtype)
    w = _as_column_free_vector(weights, "weights").to(device=phi.device, dtype=phi.dtype)
    if phi.shape != x.shape or phi.shape != w.shape:
        raise ValueError("phi_values, x_grid, and weights must have the same length")

    k = _as_column_free_vector(k_values.to(device=phi.device), "k_values").to(dtype=phi.dtype)
    if torch.any(k < 0) or torch.any(k > n):
        raise ValueError("k_values must be between 0 and n")
    log_weighted_phi = torch.log((phi * w).clamp_min(torch.finfo(phi.dtype).tiny)).view(1, -1)
    log_kernel = binomial_log_prob(k.view(-1, 1), n, x.view(1, -1))
    projected = torch.exp(torch.logsumexp(log_kernel + log_weighted_phi, dim=1))

    if folded:
        if torch.any(k < 1) or torch.any(k > n // 2):
            raise ValueError("folded k_values must be in 1..floor(n/2)")
        complement = n - k
        add_complement = complement != k
        if torch.any(add_complement):
            comp_log_kernel = binomial_log_prob(complement.view(-1, 1), n, x.view(1, -1))
            comp_projected = torch.exp(torch.logsumexp(comp_log_kernel + log_weighted_phi, dim=1))
            projected = projected + torch.where(add_complement, comp_projected, torch.zeros_like(projected))

    return _normalize_if_requested(projected, normalize)


def fold_sfs(unfolded_sfs: Union[torch.Tensor, object]) -> torch.Tensor:
    """Fold an unfolded SFS, excluding monomorphic classes k=0 and k=n."""
    sfs = torch.as_tensor(unfolded_sfs)
    if sfs.ndim != 1:
        raise ValueError("unfolded_sfs must be one-dimensional")
    if sfs.numel() < 2:
        raise ValueError("unfolded_sfs must have length n + 1")
    n = sfs.numel() - 1
    rows = []
    for k in range(1, n // 2 + 1):
        if n % 2 == 0 and k == n // 2:
            rows.append(sfs[k])
        else:
            rows.append(sfs[k] + sfs[n - k])
    if not rows:
        return torch.empty(0, device=sfs.device, dtype=sfs.dtype)
    return torch.stack(rows)


def project_model_to_folded_sfs(
    model: nn.Module,
    t_eval: Union[float, torch.Tensor],
    n: int,
    n_grid: int = 512,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Evaluate phi(x,t_eval), project to unfolded SFS, and fold."""
    if n > MAX_FULL_PROJECTION_N:
        raise ValueError(
            f"n={n} exceeds full projection limit; use sparse observed-k projection."
        )
    model_device = device
    if model_device is None:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
    x_grid, weights = make_quadrature_grid(n_points=n_grid, device=model_device)
    if isinstance(t_eval, torch.Tensor):
        t_scalar = t_eval.to(device=model_device, dtype=x_grid.dtype).reshape(-1)[0]
    else:
        t_scalar = torch.tensor(float(t_eval), device=model_device, dtype=x_grid.dtype)
    t_grid = torch.full_like(x_grid, t_scalar).view(-1, 1)
    phi_values = model(x_grid.view(-1, 1), t_grid).reshape(-1)
    unfolded = project_density_to_sfs(phi_values, x_grid, weights, n, normalize=True)
    return fold_sfs(unfolded)


def load_folded_sfs_tsv(path: Union[str, Path]) -> Tuple[torch.Tensor, int, torch.Tensor, str]:
    """Load count-indexed folded SFS TSV output from the VCF-to-SFS pipeline."""
    frame = pd.read_csv(path, sep="\t")
    required = {"population", "n_pop", "k_folded", "count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Folded SFS TSV is missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Folded SFS TSV is empty")
    n_values = sorted(frame["n_pop"].dropna().astype(int).unique())
    populations = sorted(frame["population"].dropna().astype(str).unique())
    if len(n_values) != 1:
        raise ValueError(f"Expected one n_pop value, found {n_values}")
    if len(populations) != 1:
        raise ValueError(f"Expected one population value, found {populations}")
    ordered = frame.sort_values("k_folded")
    observed_counts = torch.tensor(ordered["count"].to_numpy(), dtype=torch.float32)
    k_values = torch.tensor(ordered["k_folded"].to_numpy(), dtype=torch.long)
    return observed_counts, int(n_values[0]), k_values, populations[0]
