"""Two-population marginal-SFS training for KimuraPINN admixture skeletons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import yaml

from src.inference.losses import gamma_smoothness_loss, mse_sfs_loss, poisson_nll_sfs_loss
from src.inference.training import choose_training_k_values, load_yaml_config, set_random_seed
from src.models.admixture_model import AdmixtureModel
from src.models.demography_model import DemographyModel
from src.models.pinn import MultiPopKimuraPINN
from src.models.selection_model import SelectionModel
from src.pde.kimura_multipop import compute_two_pop_fokker_planck_residual
from src.sfs.projection import load_folded_sfs_tsv, make_quadrature_grid, project_density_to_observed_k


def build_two_pop_model(config: Dict[str, Any]) -> MultiPopKimuraPINN:
    """Build the two-population density model."""
    cfg = config.get("model", {})
    return MultiPopKimuraPINN(
        n_populations=2,
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        num_layers=int(cfg.get("num_layers", cfg.get("n_layers", 4))),
        fourier_features=int(cfg.get("fourier_features", 0)),
    )


def _selection_cfg(config: Dict[str, Any], index: int) -> Dict[str, Any]:
    cfg = config.get("selection", {})
    populations = cfg.get("populations")
    if isinstance(populations, list) and index < len(populations):
        merged = dict(cfg)
        merged.pop("populations", None)
        merged.update(populations[index])
        return merged
    return cfg


def build_two_pop_selection_models(config: Dict[str, Any]) -> list[SelectionModel]:
    """Build population-specific selection models."""
    models = []
    for index in range(2):
        cfg = _selection_cfg(config, index)
        models.append(
            SelectionModel(
                mode=cfg.get("mode", "piecewise_linear"),
                gamma=float(cfg.get("gamma", 0.0)),
                learnable=bool(cfg.get("learnable", True)),
                breakpoints=cfg.get("breakpoints"),
                values=cfg.get("values"),
                hidden_dim=int(cfg.get("hidden_dim", 32)),
                num_layers=int(cfg.get("num_layers", 2)),
            )
        )
    return models


def _demography_cfg(config: Dict[str, Any], index: int) -> Dict[str, Any]:
    cfg = config.get("demography", {})
    populations = cfg.get("populations")
    if isinstance(populations, list) and index < len(populations):
        merged = dict(cfg)
        merged.pop("populations", None)
        merged.update(populations[index])
        return merged
    return cfg


def build_two_pop_demography_models(config: Dict[str, Any]) -> list[DemographyModel]:
    """Build population-specific demography models."""
    models = []
    for index in range(2):
        cfg = _demography_cfg(config, index)
        models.append(
            DemographyModel(
                mode=cfg.get("mode", "constant"),
                nu=float(cfg.get("nu", 1.0)),
                breakpoints=cfg.get("breakpoints"),
                values=cfg.get("values"),
                learnable=bool(cfg.get("learnable", False)),
            )
        )
    return models


def build_admixture_model(config: Dict[str, Any]) -> AdmixtureModel:
    """Build a two-population admixture model from config."""
    cfg = config.get("admixture", {})
    return AdmixtureModel(
        initial_matrix=cfg.get("initial_matrix", [[0.0, 0.001], [0.001, 0.0]]),
        learnable=bool(cfg.get("learnable", True)),
    )


def sample_two_pop_collocation(batch_size: int, device: torch.device, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample two-dimensional frequency and time collocation points."""
    x = eps + (1.0 - 2.0 * eps) * torch.rand(batch_size, 2, device=device)
    t = torch.rand(batch_size, 1, device=device)
    return x, t


def two_pop_boundary_loss(model: MultiPopKimuraPINN, batch_size: int, device: torch.device, eps: float = 1e-6) -> torch.Tensor:
    """Simple absorbing boundary loss on all four faces of the square domain."""
    t = torch.rand(batch_size, 1, device=device)
    losses = []
    for dim in range(2):
        for value in (eps, 1.0 - eps):
            x = torch.rand(batch_size, 2, device=device)
            x[:, dim] = value
            losses.append(torch.mean(model(x, t).pow(2)))
    return sum(losses) / len(losses)


def _marginals_from_model(
    model: MultiPopKimuraPINN,
    n_grid: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate model at t=1 and integrate to marginal density values."""
    x_grid, weights = make_quadrature_grid(n_points=n_grid, device=device)
    x1, x2 = torch.meshgrid(x_grid, x_grid, indexing="ij")
    points = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=1)
    t = torch.ones(points.shape[0], 1, device=device, dtype=x_grid.dtype)
    phi = model(points, t).reshape(n_grid, n_grid)
    marginal_a = torch.sum(phi * weights.view(1, -1), dim=1)
    marginal_b = torch.sum(phi * weights.view(-1, 1), dim=0)
    return marginal_a, marginal_b, x_grid, weights


def project_two_pop_selected_counts(
    model: MultiPopKimuraPINN,
    n_pop_a: int,
    n_pop_b: int,
    k_a: torch.Tensor,
    k_b: torch.Tensor,
    total_a: torch.Tensor,
    total_b: torch.Tensor,
    n_grid: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project current joint density to selected marginal folded count classes."""
    marginal_a, marginal_b, x_grid, weights = _marginals_from_model(model, n_grid, device)
    pred_a = project_density_to_observed_k(marginal_a, x_grid, weights, n_pop_a, k_a.to(device), normalize=True, folded=True)
    pred_b = project_density_to_observed_k(marginal_b, x_grid, weights, n_pop_b, k_b.to(device), normalize=True, folded=True)
    return pred_a * total_a.to(device), pred_b * total_b.to(device)


def project_two_pop_full_counts_chunked(
    model: MultiPopKimuraPINN,
    n_pop: int,
    k_values: torch.Tensor,
    observed_total: torch.Tensor,
    population_index: int,
    n_grid: int,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    """Project all observed folded count classes for one marginal population."""
    marginal_a, marginal_b, x_grid, weights = _marginals_from_model(model, n_grid, device)
    marginal = marginal_a if population_index == 0 else marginal_b
    chunks = []
    with torch.no_grad():
        for start in range(0, k_values.numel(), chunk_size):
            chunk_k = k_values[start : start + chunk_size].to(device)
            raw = project_density_to_observed_k(marginal, x_grid, weights, n_pop, chunk_k, normalize=False, folded=True)
            chunks.append(raw.detach().cpu())
    raw_all = torch.cat(chunks)
    return raw_all / raw_all.sum().clamp_min(torch.finfo(raw_all.dtype).tiny) * observed_total.detach().cpu()


def train_two_population_marginal_sfs(
    sfs_a: str | Path,
    sfs_b: str | Path,
    population_a: str,
    population_b: str,
    output_dir: str | Path,
    config: Dict[str, Any],
    epochs_override: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a two-population KimuraPINN with marginal folded SFS losses."""
    seed = int(config.get("seed", 123))
    set_random_seed(seed)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    requested_device = device or config.get("device", "cpu")
    torch_device = torch.device(requested_device if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

    observed_a, n_pop_a, k_values_a, _pop_a_file = load_folded_sfs_tsv(sfs_a)
    observed_b, n_pop_b, k_values_b, _pop_b_file = load_folded_sfs_tsv(sfs_b)
    observed_a = observed_a.to(torch_device)
    observed_b = observed_b.to(torch_device)
    k_values_a = k_values_a.to(torch_device)
    k_values_b = k_values_b.to(torch_device)
    total_a = observed_a.sum().clamp_min(1.0)
    total_b = observed_b.sum().clamp_min(1.0)

    model = build_two_pop_model(config).to(torch_device)
    selection_models = [m.to(torch_device) for m in build_two_pop_selection_models(config)]
    demography_models = [m.to(torch_device) for m in build_two_pop_demography_models(config)]
    admixture_model = build_admixture_model(config).to(torch_device)

    training_cfg = config.get("training", {})
    projection_cfg = config.get("projection", {})
    loss_cfg = config.get("loss_weights", {})
    epochs = int(epochs_override if epochs_override is not None else training_cfg.get("epochs", 100))
    lr = float(training_cfg.get("lr", 1e-3))
    batch_collocation = int(training_cfg.get("batch_collocation", 64))
    boundary_batch = int(training_cfg.get("boundary_batch", min(batch_collocation, 64)))
    gamma_grid_size = int(training_cfg.get("gamma_grid_size", 64))
    data_loss_name = str(training_cfg.get("data_loss", "mse"))
    n_grid = int(projection_cfg.get("n_grid", 64))
    max_zero_bins = int(projection_cfg.get("max_zero_bins_for_loss", 64))

    idx_a = choose_training_k_values(observed_a.detach().cpu(), k_values_a.detach().cpu(), max_zero_bins).to(torch_device)
    idx_b = choose_training_k_values(observed_b.detach().cpu(), k_values_b.detach().cpu(), max_zero_bins).to(torch_device)
    train_k_a = k_values_a[idx_a]
    train_k_b = k_values_b[idx_b]
    train_obs_a = observed_a[idx_a]
    train_obs_b = observed_b[idx_b]

    parameters = (
        list(model.parameters())
        + [p for m in selection_models for p in m.parameters()]
        + [p for m in demography_models for p in m.parameters()]
        + list(admixture_model.parameters())
    )
    optimizer = torch.optim.Adam([p for p in parameters if p.requires_grad], lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        x_collocation, t_collocation = sample_two_pop_collocation(batch_collocation, torch_device)
        residual = compute_two_pop_fokker_planck_residual(
            model,
            selection_models,
            demography_models,
            admixture_model,
            x_collocation,
            t_collocation,
            h=float(config.get("h", 0.5)),
        )
        pde_loss = torch.mean(residual.pow(2))
        pred_a, pred_b = project_two_pop_selected_counts(
            model,
            n_pop_a,
            n_pop_b,
            train_k_a,
            train_k_b,
            total_a,
            total_b,
            n_grid,
            torch_device,
        )
        if data_loss_name == "poisson":
            data_loss_a = poisson_nll_sfs_loss(pred_a, train_obs_a)
            data_loss_b = poisson_nll_sfs_loss(pred_b, train_obs_b)
        else:
            data_loss_a = mse_sfs_loss(pred_a, train_obs_a, normalize=True)
            data_loss_b = mse_sfs_loss(pred_b, train_obs_b, normalize=True)
        boundary_loss = two_pop_boundary_loss(model, boundary_batch, torch_device)
        t_gamma = torch.linspace(0.0, 1.0, gamma_grid_size, device=torch_device).view(-1, 1)
        gamma_loss = sum(gamma_smoothness_loss(sel, t_gamma) for sel in selection_models) / len(selection_models)
        migration_matrix = admixture_model()
        migration_reg = torch.mean(migration_matrix.pow(2))
        total_loss = (
            float(loss_cfg.get("pde", 1.0)) * pde_loss
            + float(loss_cfg.get("data_a", loss_cfg.get("data", 1.0))) * data_loss_a
            + float(loss_cfg.get("data_b", loss_cfg.get("data", 1.0))) * data_loss_b
            + float(loss_cfg.get("boundary", 0.01)) * boundary_loss
            + float(loss_cfg.get("gamma_smoothness", 0.01)) * gamma_loss
            + float(loss_cfg.get("migration", 0.0)) * migration_reg
        )
        total_loss.backward()
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "total_loss": float(total_loss.detach().cpu()),
                "pde_loss": float(pde_loss.detach().cpu()),
                "data_loss_a": float(data_loss_a.detach().cpu()),
                "data_loss_b": float(data_loss_b.detach().cpu()),
                "boundary_loss": float(boundary_loss.detach().cpu()),
                "gamma_smoothness_loss": float(gamma_loss.detach().cpu()),
                "migration_reg_loss": float(migration_reg.detach().cpu()),
            }
        )

    save_two_pop_outputs(
        run_dir,
        model,
        selection_models,
        demography_models,
        admixture_model,
        config,
        history,
        population_a,
        population_b,
        observed_a.detach().cpu(),
        observed_b.detach().cpu(),
        k_values_a.detach().cpu(),
        k_values_b.detach().cpu(),
        n_pop_a,
        n_pop_b,
        total_a.detach().cpu(),
        total_b.detach().cpu(),
        n_grid,
        int(projection_cfg.get("output_chunk_size", 2048)),
        torch_device,
    )
    return {
        "run_dir": str(run_dir),
        "population_a": population_a,
        "population_b": population_b,
        "epochs": epochs,
        "final_total_loss": history[-1]["total_loss"] if history else None,
    }


def save_two_pop_outputs(
    run_dir: Path,
    model: MultiPopKimuraPINN,
    selection_models: list[SelectionModel],
    demography_models: list[DemographyModel],
    admixture_model: AdmixtureModel,
    config: Dict[str, Any],
    history: list[Dict[str, float]],
    population_a: str,
    population_b: str,
    observed_a: torch.Tensor,
    observed_b: torch.Tensor,
    k_values_a: torch.Tensor,
    k_values_b: torch.Tensor,
    n_pop_a: int,
    n_pop_b: int,
    total_a: torch.Tensor,
    total_b: torch.Tensor,
    n_grid: int,
    chunk_size: int,
    device: torch.device,
) -> None:
    """Save two-population training artifacts."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "selection_state_dicts": [m.state_dict() for m in selection_models],
            "demography_state_dicts": [m.state_dict() for m in demography_models],
            "admixture_state_dict": admixture_model.state_dict(),
            "config": config,
            "populations": [population_a, population_b],
            "n_pop": [n_pop_a, n_pop_b],
        },
        run_dir / "model.pt",
    )
    with (run_dir / "run_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    pd.DataFrame(history).to_csv(run_dir / "loss_history.tsv", sep="\t", index=False)
    final_metrics = history[-1] if history else {}
    final_metrics["limitation"] = (
        "Marginal folded SFS only; migration estimates are effective coupling parameters, "
        "not definitive migration rates."
    )
    with (run_dir / "metrics.json").open("w") as handle:
        json.dump(final_metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    gamma_points = int(config.get("output", {}).get("gamma_points", 101))
    t_gamma = torch.linspace(0.0, 1.0, gamma_points, device=device).view(-1, 1)
    gamma_rows = []
    with torch.no_grad():
        for population, selection_model in zip([population_a, population_b], selection_models):
            gamma = selection_model(t_gamma).detach().cpu().reshape(-1).tolist()
            for time_value, gamma_value in zip(t_gamma.detach().cpu().reshape(-1).tolist(), gamma):
                gamma_rows.append({"time": time_value, "gamma": gamma_value, "population": population})
    pd.DataFrame(gamma_rows).to_csv(run_dir / "gamma_trajectory.tsv", sep="\t", index=False)

    matrix = admixture_model().detach().cpu()
    migration_rows = []
    populations = [population_a, population_b]
    for target_index, target in enumerate(populations):
        for source_index, source in enumerate(populations):
            migration_rows.append(
                {
                    "source_population": source,
                    "target_population": target,
                    "migration_rate": float(matrix[target_index, source_index]),
                }
            )
    pd.DataFrame(migration_rows).to_csv(run_dir / "migration_matrix.tsv", sep="\t", index=False)

    pred_a = project_two_pop_full_counts_chunked(
        model,
        n_pop_a,
        k_values_a,
        total_a,
        0,
        n_grid,
        device,
        chunk_size,
    )
    pred_b = project_two_pop_full_counts_chunked(
        model,
        n_pop_b,
        k_values_b,
        total_b,
        1,
        n_grid,
        device,
        chunk_size,
    )
    _write_predicted_sfs(run_dir / f"predicted_sfs_{population_a}.tsv", population_a, n_pop_a, k_values_a, observed_a, pred_a)
    _write_predicted_sfs(run_dir / f"predicted_sfs_{population_b}.tsv", population_b, n_pop_b, k_values_b, observed_b, pred_b)


def _write_predicted_sfs(
    path: Path,
    population: str,
    n_pop: int,
    k_values: torch.Tensor,
    observed: torch.Tensor,
    predicted: torch.Tensor,
) -> None:
    pd.DataFrame(
        {
            "population": population,
            "n_pop": n_pop,
            "k_folded": k_values.tolist(),
            "observed_count": observed.tolist(),
            "predicted_count": predicted.tolist(),
        }
    ).to_csv(path, sep="\t", index=False)


def train_two_population_from_config(
    sfs_a: str | Path,
    sfs_b: str | Path,
    population_a: str,
    population_b: str,
    config_path: str | Path,
    output_dir: str | Path,
    epochs: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Load config and run two-population marginal-SFS training."""
    config = load_yaml_config(config_path)
    return train_two_population_marginal_sfs(
        sfs_a=sfs_a,
        sfs_b=sfs_b,
        population_a=population_a,
        population_b=population_b,
        output_dir=output_dir,
        config=config,
        epochs_override=epochs,
        device=device,
    )

