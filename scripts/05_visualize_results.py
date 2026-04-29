#!/usr/bin/env python3
"""
Script: 05_visualize_results.py

Purpose:
    Generate publication-oriented diagnostic figures from a completed
    single-population KimuraPINN run directory.

Biological Question:
    How well does the inferred KimuraPINN density reproduce the observed folded
    SFS, and what time-varying selection trajectory gamma(t) was inferred?

Assumptions:
    - Folded SFS is used because ancestral allele is unavailable.
    - Only SNVs are retained upstream by the VCF-to-SFS pipeline.
    - Allele count fields are taken from gnomAD-style INFO columns upstream.
    - The SFS is allele-count indexed, not frequency-bin indexed.

Input:
    - Run directory containing loss_history.tsv, gamma_trajectory.tsv,
      predicted_sfs.tsv, and optionally model.pt plus run_config.yaml.

Output:
    - PNG and PDF figures written to <run-dir>/figures/.
    - Observed-vs-predicted SFS, SFS residuals, gamma trajectory, loss history,
      and optionally phi density heatmap if the model loads cleanly.

Process Schematic:
    Run directory
        -> read tabular outputs
        -> plot SFS fit and residuals
        -> plot gamma trajectory
        -> plot loss history
        -> optionally load model and plot phi(x,t)
        -> save figures

Example:
    python scripts/05_visualize_results.py \
        --run-dir results/time_varying_nfe_test \
        --log-sfs
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kimurapinn_matplotlib"))

from src.visualization.plot_gamma import plot_gamma_trajectory
from src.visualization.plot_loss import plot_loss_history
from src.visualization.plot_sfs import plot_sfs_observed_vs_predicted, plot_sfs_residuals


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize a KimuraPINN run directory.")
    parser.add_argument("--run-dir", required=True, help="Run directory with TSV outputs.")
    parser.add_argument("--log-sfs", action="store_true", help="Use log y-scale for SFS counts.")
    parser.add_argument("--max-sfs-points", type=int, default=5000, help="Maximum SFS points to draw.")
    parser.add_argument("--skip-density", action="store_true", help="Do not attempt model heatmap plotting.")
    return parser.parse_args()


def main() -> None:
    """Create figures for a run directory."""
    args = parse_args()
    run_dir = Path(args.run_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    created = []
    created.extend(
        plot_sfs_observed_vs_predicted(
            run_dir / "predicted_sfs.tsv",
            figures_dir,
            log_y=args.log_sfs,
            max_points=args.max_sfs_points,
        )
    )
    created.extend(
        plot_sfs_residuals(
            run_dir / "predicted_sfs.tsv",
            figures_dir,
            max_points=args.max_sfs_points,
        )
    )
    created.extend(plot_gamma_trajectory(run_dir / "gamma_trajectory.tsv", figures_dir))
    created.extend(plot_loss_history(run_dir / "loss_history.tsv", figures_dir))

    if not args.skip_density:
        density_paths = None
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                from src.visualization.plot_density import plot_phi_density_heatmap

                density_paths = plot_phi_density_heatmap(run_dir, figures_dir)
            except Exception:
                density_paths = None
        if density_paths:
            created.extend(density_paths)

    print(f"Wrote {len(created)} figure files to {figures_dir}")


if __name__ == "__main__":
    main()
