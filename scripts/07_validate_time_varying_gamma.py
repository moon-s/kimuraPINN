#!/usr/bin/env python3
"""
Script: 07_validate_time_varying_gamma.py

Purpose:
    Validate single-population KimuraPINN training by fitting synthetic folded
    SFS data generated under a known time-varying selection trajectory.

Biological Question:
    Does the inferred gamma(t) qualitatively track the known selection
    trajectory used to simulate a folded SFS?

Assumptions:
    - Folded SFS is used because ancestral allele is unavailable.
    - Simulated inputs represent SNVs.
    - Folded SFS alone may not uniquely identify gamma(t); metrics are
      descriptive validation diagnostics, not proof of exact recovery.

Input:
    - Simulation YAML config, or a precomputed simulated folded SFS plus true
      gamma trajectory.
    - Training YAML config for the existing single-population trainer.

Output:
    - model.pt, loss_history.tsv, predicted_sfs.tsv
    - inferred_gamma_trajectory.tsv
    - true_gamma_trajectory.tsv
    - gamma_recovery_metrics.json
    - figures for true-vs-inferred gamma, SFS fit, and gamma error

Process Schematic:
    Simulation config or simulated SFS
        -> create/load folded SFS and true gamma
        -> train one-population KimuraPINN
        -> compare true and inferred gamma(t)
        -> save metrics and figures

Example:
    python scripts/07_validate_time_varying_gamma.py \
        --simulation-config configs/simulate_time_varying_selection.yaml \
        --training-config configs/validate_time_varying_selection.yaml \
        --output-dir results/validation_step_gamma_test \
        --epochs 100
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kimurapinn_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.inference.losses import mse_sfs_loss, poisson_nll_sfs_loss
from src.inference.training import load_yaml_config, train_single_population
from src.simulation.simulate_sfs import simulate_single_population_sfs
from src.visualization.plot_sfs import save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate time-varying gamma recovery.")
    parser.add_argument("--simulation-config", help="Simulation YAML config.")
    parser.add_argument("--training-config", required=True, help="Training YAML config.")
    parser.add_argument("--output-dir", required=True, help="Validation output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--simulated-sfs", default=None, help="Precomputed folded SFS TSV.")
    parser.add_argument("--true-gamma", default=None, help="Precomputed true gamma trajectory TSV.")
    return parser.parse_args()


def _pearson(x, y):
    if len(x) < 2 or pd.Series(x).var() <= 0 or pd.Series(y).var() <= 0:
        return None
    return float(pd.Series(x).corr(pd.Series(y)))


def plot_true_vs_inferred(true_frame: pd.DataFrame, inferred_frame: pd.DataFrame, output_dir: Path) -> None:
    merged = align_gamma(true_frame, inferred_frame)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(merged["time"], merged["true_gamma"], label="True", linewidth=1.8)
    ax.plot(merged["time"], merged["inferred_gamma"], label="Inferred", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("gamma(t)")
    ax.set_title("True vs inferred gamma")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    save_figure(fig, output_dir, "true_vs_inferred_gamma")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(merged["time"], merged["inferred_gamma"] - merged["true_gamma"], linewidth=1.6, color="#8c2d04")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Inferred - true")
    ax.set_title("Gamma recovery error")
    ax.grid(True, alpha=0.25)
    save_figure(fig, output_dir, "gamma_error")


def plot_simulated_sfs_fit(predicted_path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(predicted_path, sep="\t")
    display = frame[(frame["observed_count"] > 0) | (frame["predicted_count"] > 0)]
    if display.empty:
        display = frame
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(display["k_folded"], display["observed_count"], label="Simulated", linewidth=1.5)
    ax.plot(display["k_folded"], display["predicted_count"], label="Predicted", linewidth=1.5)
    ax.set_xlabel("Folded allele count k")
    ax.set_ylabel("Count")
    ax.set_title("Simulated SFS fit")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    save_figure(fig, output_dir, "simulated_sfs_fit")


def align_gamma(true_frame: pd.DataFrame, inferred_frame: pd.DataFrame) -> pd.DataFrame:
    times = inferred_frame["time"].astype(float)
    true_interp = pd.Series(index=times.index, dtype=float)
    true_sorted = true_frame.sort_values("time")
    for idx, time in times.items():
        lower = true_sorted[true_sorted["time"] <= time].tail(1)
        upper = true_sorted[true_sorted["time"] >= time].head(1)
        if lower.empty:
            true_interp.loc[idx] = float(upper["gamma"].iloc[0])
        elif upper.empty:
            true_interp.loc[idx] = float(lower["gamma"].iloc[0])
        elif float(lower["time"].iloc[0]) == float(upper["time"].iloc[0]):
            true_interp.loc[idx] = float(lower["gamma"].iloc[0])
        else:
            t0 = float(lower["time"].iloc[0])
            t1 = float(upper["time"].iloc[0])
            g0 = float(lower["gamma"].iloc[0])
            g1 = float(upper["gamma"].iloc[0])
            true_interp.loc[idx] = g0 + (float(time) - t0) * (g1 - g0) / (t1 - t0)
    return pd.DataFrame(
        {
            "time": times,
            "true_gamma": true_interp,
            "inferred_gamma": inferred_frame["gamma"].astype(float).reset_index(drop=True),
        }
    )


def compute_metrics(output_dir: Path) -> dict:
    true_frame = pd.read_csv(output_dir / "true_gamma_trajectory.tsv", sep="\t")
    inferred_frame = pd.read_csv(output_dir / "inferred_gamma_trajectory.tsv", sep="\t")
    merged = align_gamma(true_frame, inferred_frame)
    gamma_error = merged["inferred_gamma"] - merged["true_gamma"]
    predicted = pd.read_csv(output_dir / "predicted_sfs.tsv", sep="\t")
    sfs_mse = float(((predicted["predicted_count"] - predicted["observed_count"]) ** 2).mean())
    import torch

    poisson = float(
        poisson_nll_sfs_loss(
            torch.tensor(predicted["predicted_count"].tolist()),
            torch.tensor(predicted["observed_count"].tolist()),
        )
    )
    loss_history = pd.read_csv(output_dir / "loss_history.tsv", sep="\t")
    return {
        "gamma_mse": float((gamma_error**2).mean()),
        "gamma_pearson_correlation": _pearson(merged["true_gamma"], merged["inferred_gamma"]),
        "final_sfs_mse": sfs_mse,
        "final_sfs_poisson_nll": poisson,
        "final_total_loss": float(loss_history["total_loss"].iloc[-1]) if not loss_history.empty else None,
        "interpretation": "Folded SFS validation is qualitative; exact gamma(t) recovery is not guaranteed.",
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.simulated_sfs:
        if not args.true_gamma:
            raise ValueError("--true-gamma is required when --simulated-sfs is provided")
        sfs_path = Path(args.simulated_sfs)
        shutil.copy2(args.true_gamma, output_dir / "true_gamma_trajectory.tsv")
    else:
        if not args.simulation_config:
            raise ValueError("--simulation-config is required unless --simulated-sfs is provided")
        sim_config = load_yaml_config(args.simulation_config)
        sim_dir = output_dir / "simulation_inputs"
        sim_summary = simulate_single_population_sfs(sim_config, sim_dir)
        sfs_path = Path(sim_summary["folded_sfs"])
        shutil.copy2(sim_summary["true_gamma"], output_dir / "true_gamma_trajectory.tsv")

    train_config = load_yaml_config(args.training_config)
    train_single_population(
        sfs_path=sfs_path,
        output_dir=output_dir,
        config=train_config,
        epochs_override=args.epochs,
    )
    shutil.copy2(output_dir / "gamma_trajectory.tsv", output_dir / "inferred_gamma_trajectory.tsv")

    true_frame = pd.read_csv(output_dir / "true_gamma_trajectory.tsv", sep="\t")
    inferred_frame = pd.read_csv(output_dir / "inferred_gamma_trajectory.tsv", sep="\t")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_true_vs_inferred(true_frame, inferred_frame, figures_dir)
    plot_simulated_sfs_fit(output_dir / "predicted_sfs.tsv", figures_dir)
    metrics = compute_metrics(output_dir)
    with (output_dir / "gamma_recovery_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Validation complete: {output_dir}")


if __name__ == "__main__":
    main()

