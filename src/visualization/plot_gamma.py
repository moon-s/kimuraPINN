"""Plot inferred selection trajectories."""

from __future__ import annotations

from pathlib import Path

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kimurapinn_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.plot_sfs import save_figure


def plot_gamma_trajectory(gamma_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Plot gamma(t) from a gamma_trajectory.tsv file."""
    frame = pd.read_csv(gamma_path, sep="\t")
    required = {"time", "gamma"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"gamma_trajectory.tsv is missing columns: {', '.join(sorted(missing))}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if "population" in frame.columns:
        for population, group in frame.groupby("population"):
            ax.plot(group["time"], group["gamma"], label=str(population), linewidth=1.8)
        ax.legend(frameon=False)
    else:
        ax.plot(frame["time"], frame["gamma"], linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("gamma(t)")
    ax.set_title("Selection trajectory")
    ax.grid(True, alpha=0.25)
    return save_figure(fig, Path(output_dir), "gamma_trajectory")
