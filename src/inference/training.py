"""Single-population KimuraPINN training utilities."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict, Optional

import pandas as pd
import torch
import yaml

from src.inference.losses import (
    gamma_smoothness_loss,
    mse_sfs_loss,
    poisson_nll_sfs_loss,
)
from src.models.demography_model import DemographyModel
from src.models.pinn import KimuraPINN
from src.models.selection_model import SelectionModel
from src.pde.boundary_conditions import boundary_loss_absorbing
from src.pde.fokker_planck import compute_fokker_planck_residual
from src.sfs.projection import (
    load_folded_sfs_tsv,
    make_quadrature_grid,
    project_density_to_observed_k,
)


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config must contain a YAML mapping")
    return data


def set_random_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a configuration dictionary."""
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def build_model(config: Dict[str, Any]) -> KimuraPINN:
    model_cfg = config.get("model", {})
    return KimuraPINN(
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", model_cfg.get("n_layers", 4))),
        fourier_features=int(model_cfg.get("fourier_features", 0)),
    )


def build_selection_model(config: Dict[str, Any]) -> SelectionModel:
    cfg = config.get("selection", {})
    mode = cfg.get("mode", "piecewise_linear")
    return SelectionModel(
        mode=mode,
        gamma=float(cfg.get("gamma", 0.0)),
        learnable=bool(cfg.get("learnable", True)),
        breakpoints=cfg.get("breakpoints"),
        values=cfg.get("values"),
        hidden_dim=int(cfg.get("hidden_dim", 32)),
        num_layers=int(cfg.get("num_layers", 2)),
    )


def build_demography_model(config: Dict[str, Any]) -> DemographyModel:
    cfg = config.get("demography", {})
    return DemographyModel(
        mode=cfg.get("mode", "constant"),
        nu=float(cfg.get("nu", 1.0)),
        breakpoints=cfg.get("breakpoints"),
        values=cfg.get("values"),
        learnable=bool(cfg.get("learnable", False)),
    )


def choose_training_k_values(
    observed_counts: torch.Tensor,
    k_values: torch.Tensor,
    max_zero_bins: int,
) -> torch.Tensor:
    """Use all nonzero bins plus a deterministic subset of zero-count bins."""
    nonzero_mask = observed_counts > 0
    nonzero_indices = torch.where(nonzero_mask)[0]
    zero_indices = torch.where(~nonzero_mask)[0]
    if max_zero_bins > 0 and zero_indices.numel() > max_zero_bins:
        positions = torch.linspace(0, zero_indices.numel() - 1, steps=max_zero_bins).round().long()
        zero_indices = zero_indices[positions]
    selected = torch.cat([nonzero_indices, zero_indices])
    selected = torch.unique(selected, sorted=True)
    return selected


def sample_collocation(batch_size: int, device: torch.device, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample interior collocation points in x and t."""
    x = eps + (1.0 - 2.0 * eps) * torch.rand(batch_size, 1, device=device)
    t = torch.rand(batch_size, 1, device=device)
    return x, t


def project_current_model(
    model: KimuraPINN,
    n_pop: int,
    k_values: torch.Tensor,
    observed_total: torch.Tensor,
    n_grid: int,
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    """Project model density at t=1 to selected folded count classes."""
    x_grid, weights = make_quadrature_grid(n_points=n_grid, device=device)
    t_grid = torch.ones_like(x_grid).view(-1, 1)
    phi_values = model(x_grid.view(-1, 1), t_grid).reshape(-1)
    projected = project_density_to_observed_k(
        phi_values,
        x_grid,
        weights,
        n=n_pop,
        k_values=k_values.to(device=device),
        normalize=normalize,
        folded=True,
    )
    return projected * observed_total.to(device=device, dtype=projected.dtype)


def project_current_model_chunked(
    model: KimuraPINN,
    n_pop: int,
    k_values: torch.Tensor,
    observed_total: torch.Tensor,
    n_grid: int,
    device: torch.device,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Memory-safe projection for output tables across many folded classes."""
    x_grid, weights = make_quadrature_grid(n_points=n_grid, device=device)
    t_grid = torch.ones_like(x_grid).view(-1, 1)
    with torch.no_grad():
        phi_values = model(x_grid.view(-1, 1), t_grid).reshape(-1)
        chunks = []
        for start in range(0, k_values.numel(), chunk_size):
            chunk_k = k_values[start : start + chunk_size].to(device=device)
            projected = project_density_to_observed_k(
                phi_values,
                x_grid,
                weights,
                n=n_pop,
                k_values=chunk_k,
                normalize=False,
                folded=True,
            )
            chunks.append(projected.detach().cpu())
        raw = torch.cat(chunks)
        total = raw.sum().clamp_min(torch.finfo(raw.dtype).tiny)
        return raw / total * observed_total.detach().cpu()


def train_single_population(
    sfs_path: str | Path,
    output_dir: str | Path,
    config: Dict[str, Any],
    epochs_override: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train one-population KimuraPINN against a folded count SFS."""
    seed = int(config.get("seed", 123))
    set_random_seed(seed)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    requested_device = device or config.get("device", "cpu")
    torch_device = torch.device(requested_device if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

    observed_counts, n_pop, k_values, population = load_folded_sfs_tsv(sfs_path)
    observed_counts = observed_counts.to(torch_device)
    k_values = k_values.to(torch_device)
    observed_total = observed_counts.sum().clamp_min(1.0)

    model = build_model(config).to(torch_device)
    selection_model = build_selection_model(config).to(torch_device)
    demography_model = build_demography_model(config).to(torch_device)

    training_cfg = config.get("training", {})
    loss_cfg = config.get("loss_weights", {})
    projection_cfg = config.get("projection", {})
    epochs = int(epochs_override if epochs_override is not None else training_cfg.get("epochs", 100))
    lr = float(training_cfg.get("lr", 1e-3))
    batch_collocation = int(training_cfg.get("batch_collocation", 128))
    boundary_batch = int(training_cfg.get("boundary_batch", min(batch_collocation, 128)))
    gamma_grid_size = int(training_cfg.get("gamma_grid_size", 64))
    n_grid = int(projection_cfg.get("n_grid", 256))
    max_zero_bins = int(projection_cfg.get("max_zero_bins_for_loss", 256))
    data_loss_name = str(training_cfg.get("data_loss", "mse"))

    selected_indices = choose_training_k_values(observed_counts.detach().cpu(), k_values.detach().cpu(), max_zero_bins)
    selected_indices = selected_indices.to(torch_device)
    train_k = k_values[selected_indices]
    train_observed = observed_counts[selected_indices]

    parameters = list(model.parameters()) + list(selection_model.parameters()) + list(demography_model.parameters())
    optimizer = torch.optim.Adam([param for param in parameters if param.requires_grad], lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        x_collocation, t_collocation = sample_collocation(batch_collocation, torch_device)
        residual = compute_fokker_planck_residual(
            model,
            selection_model,
            x_collocation,
            t_collocation,
            nu=demography_model,
            h=float(config.get("h", 0.5)),
        )
        pde_loss = torch.mean(residual.pow(2))

        predicted = project_current_model(
            model,
            n_pop,
            train_k,
            observed_total,
            n_grid,
            torch_device,
            normalize=True,
        )
        if data_loss_name == "poisson":
            data_loss = poisson_nll_sfs_loss(predicted, train_observed)
        else:
            data_loss = mse_sfs_loss(predicted, train_observed, normalize=True)

        t_boundary = torch.rand(boundary_batch, 1, device=torch_device)
        boundary_loss = boundary_loss_absorbing(model, t_boundary)
        t_gamma = torch.linspace(0.0, 1.0, gamma_grid_size, device=torch_device).view(-1, 1)
        gamma_loss = gamma_smoothness_loss(selection_model, t_gamma)

        total_loss = (
            float(loss_cfg.get("pde", 1.0)) * pde_loss
            + float(loss_cfg.get("data", 1.0)) * data_loss
            + float(loss_cfg.get("boundary", 0.01)) * boundary_loss
            + float(loss_cfg.get("gamma_smoothness", 0.01)) * gamma_loss
        )
        total_loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": epoch,
                "total_loss": float(total_loss.detach().cpu()),
                "pde_loss": float(pde_loss.detach().cpu()),
                "data_loss": float(data_loss.detach().cpu()),
                "boundary_loss": float(boundary_loss.detach().cpu()),
                "gamma_smoothness_loss": float(gamma_loss.detach().cpu()),
            }
        )

    save_training_outputs(
        run_dir=run_dir,
        model=model,
        selection_model=selection_model,
        demography_model=demography_model,
        config=config,
        history=history,
        observed_counts=observed_counts.detach().cpu(),
        n_pop=n_pop,
        k_values=k_values.detach().cpu(),
        population=population,
        observed_total=observed_total.detach().cpu(),
        n_grid=n_grid,
        device=torch_device,
    )
    return {
        "run_dir": str(run_dir),
        "population": population,
        "n_pop": n_pop,
        "epochs": epochs,
        "final_total_loss": history[-1]["total_loss"] if history else None,
    }


def save_training_outputs(
    run_dir: Path,
    model: KimuraPINN,
    selection_model: SelectionModel,
    demography_model: DemographyModel,
    config: Dict[str, Any],
    history: list[Dict[str, float]],
    observed_counts: torch.Tensor,
    n_pop: int,
    k_values: torch.Tensor,
    population: str,
    observed_total: torch.Tensor,
    n_grid: int,
    device: torch.device,
) -> None:
    """Persist model state, config, metrics, and tabular predictions."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "selection_state_dict": selection_model.state_dict(),
            "demography_state_dict": demography_model.state_dict(),
            "config": config,
            "n_pop": n_pop,
            "population": population,
        },
        run_dir / "model.pt",
    )
    with (run_dir / "run_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    pd.DataFrame(history).to_csv(run_dir / "loss_history.tsv", sep="\t", index=False)

    final_metrics = history[-1] if history else {}
    with (run_dir / "metrics.json").open("w") as handle:
        json.dump(final_metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    t_gamma = torch.linspace(0.0, 1.0, int(config.get("output", {}).get("gamma_points", 101)), device=device).view(-1, 1)
    with torch.no_grad():
        gamma = selection_model(t_gamma).detach().cpu().reshape(-1)
    pd.DataFrame(
        {
            "time": t_gamma.detach().cpu().reshape(-1).tolist(),
            "gamma": gamma.tolist(),
            "population": population,
        }
    ).to_csv(run_dir / "gamma_trajectory.tsv", sep="\t", index=False)

    chunk_size = int(config.get("projection", {}).get("output_chunk_size", 4096))
    predicted = project_current_model_chunked(
        model,
        n_pop,
        k_values,
        observed_total,
        n_grid,
        device,
        chunk_size=chunk_size,
    )
    pd.DataFrame(
        {
            "population": population,
            "n_pop": n_pop,
            "k_folded": k_values.tolist(),
            "observed_count": observed_counts.tolist(),
            "predicted_count": predicted.tolist(),
        }
    ).to_csv(run_dir / "predicted_sfs.tsv", sep="\t", index=False)
