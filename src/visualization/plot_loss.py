"""Plot training loss histories."""

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


LOSS_COLUMNS = [
    "total_loss",
    "pde_loss",
    "data_loss",
    "boundary_loss",
    "gamma_smoothness_loss",
]


def plot_loss_history(loss_history_path: str | Path, output_dir: str | Path, *, log_y: bool = True) -> list[Path]:
    """Plot available loss columns over epochs."""
    frame = pd.read_csv(loss_history_path, sep="\t")
    if "epoch" not in frame.columns:
        raise ValueError("loss_history.tsv is missing column: epoch")
    available = [column for column in LOSS_COLUMNS if column in frame.columns]
    if not available:
        raise ValueError("loss_history.tsv has no recognized loss columns")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for column in available:
        values = frame[column].clip(lower=1e-12) if log_y else frame[column]
        ax.plot(frame["epoch"], values, label=column, linewidth=1.4)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss history")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    return save_figure(fig, Path(output_dir), "loss_history")
