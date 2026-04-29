"""Plot PINN density heatmaps when a saved model can be reconstructed."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kimurapinn_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from src.inference.training import build_model
from src.visualization.plot_sfs import save_figure


def load_model_for_density(run_dir: str | Path):
    """Load a KimuraPINN model from model.pt and run_config.yaml if possible."""
    run_path = Path(run_dir)
    model_path = run_path / "model.pt"
    config_path = run_path / "run_config.yaml"
    if not model_path.exists() or not config_path.exists():
        return None
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle) or {}
    checkpoint = torch.load(model_path, map_location="cpu")
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def plot_phi_density_heatmap(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    n_x: int = 120,
    n_t: int = 80,
) -> Optional[list[Path]]:
    """Plot phi(x,t) as a heatmap if the model loads successfully."""
    try:
        model = load_model_for_density(run_dir)
    except Exception:
        return None
    if model is None:
        return None

    x_values = torch.linspace(1e-4, 1.0 - 1e-4, n_x)
    t_values = torch.linspace(0.0, 1.0, n_t)
    rows: list[list[float]] = []
    with torch.no_grad():
        for t_value in t_values:
            x = x_values.view(-1, 1)
            t = torch.full_like(x, t_value)
            phi = model(x, t).reshape(-1).detach().cpu().tolist()
            rows.append([float(value) for value in phi])

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    mesh = ax.imshow(
        rows,
        origin="lower",
        aspect="auto",
        extent=[float(x_values[0]), float(x_values[-1]), float(t_values[0]), float(t_values[-1])],
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax, label="phi(x,t)")
    ax.set_xlabel("Allele frequency x")
    ax.set_ylabel("Time t")
    ax.set_title("PINN density")
    return save_figure(fig, Path(output_dir), "phi_density_heatmap")
