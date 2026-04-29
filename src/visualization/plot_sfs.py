"""Plot observed and predicted count-indexed folded SFS outputs."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kimurapinn_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_SFS_COLUMNS = {"k_folded", "observed_count", "predicted_count"}


def _validate_sfs_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_SFS_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"predicted_sfs.tsv is missing columns: {', '.join(sorted(missing))}")


def _display_frame(frame: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    """Downsample very large SFS tables for plotting without changing source data."""
    if len(frame) <= max_points:
        return frame
    nonzero = frame[(frame["observed_count"] > 0) | (frame["predicted_count"] > 0)]
    remaining = max(max_points - len(nonzero), 0)
    if remaining > 0:
        sampled = frame.iloc[
            [round(i) for i in pd.Series(range(remaining)) * (len(frame) - 1) / max(remaining - 1, 1)]
        ]
        display = pd.concat([nonzero, sampled], axis=0)
    else:
        display = nonzero
    return display.drop_duplicates(subset=["k_folded"]).sort_values("k_folded")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    """Save a figure as PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return paths


def plot_sfs_observed_vs_predicted(
    predicted_sfs_path: str | Path,
    output_dir: str | Path,
    *,
    log_y: bool = False,
    max_points: int = 5000,
) -> list[Path]:
    """Plot observed and predicted folded SFS counts."""
    frame = pd.read_csv(predicted_sfs_path, sep="\t")
    _validate_sfs_frame(frame)
    display = _display_frame(frame, max_points=max_points)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(display["k_folded"], display["observed_count"], label="Observed", linewidth=1.5)
    ax.plot(display["k_folded"], display["predicted_count"], label="Predicted", linewidth=1.5)
    ax.set_xlabel("Folded allele count k")
    ax.set_ylabel("Count")
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(1e-6, min(display[["observed_count", "predicted_count"]].min().min(), 1.0)))
    population = frame["population"].iloc[0] if "population" in frame.columns and not frame.empty else ""
    ax.set_title(f"Folded SFS {population}".strip())
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    return save_figure(fig, Path(output_dir), "sfs_observed_vs_predicted")


def plot_sfs_residuals(
    predicted_sfs_path: str | Path,
    output_dir: str | Path,
    *,
    max_points: int = 5000,
) -> list[Path]:
    """Plot predicted minus observed folded SFS counts."""
    frame = pd.read_csv(predicted_sfs_path, sep="\t")
    _validate_sfs_frame(frame)
    frame = frame.copy()
    frame["residual"] = frame["predicted_count"] - frame["observed_count"]
    display = _display_frame(frame, max_points=max_points)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(display["k_folded"], display["residual"], color="#8c2d04", linewidth=1.2)
    ax.set_xlabel("Folded allele count k")
    ax.set_ylabel("Predicted - observed")
    ax.set_title("Folded SFS residuals")
    ax.grid(True, alpha=0.25)
    return save_figure(fig, Path(output_dir), "sfs_residuals")
