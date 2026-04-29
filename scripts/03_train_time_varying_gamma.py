#!/usr/bin/env python3
"""
Script: 03_train_time_varying_gamma.py

Purpose:
    Train a one-population KimuraPINN against an observed count-indexed folded
    SFS and infer a time-varying selection trajectory gamma(t).

Biological Question:
    Given a folded allele-count spectrum for one population, what smooth
    time-varying selection coefficient trajectory is compatible with the
    Kimura diffusion model and the observed spectrum?

Assumptions:
    - Folded SFS is used because ancestral allele is unavailable.
    - Only SNVs are retained upstream by the VCF-to-SFS pipeline.
    - Allele count fields are taken from gnomAD-style INFO columns upstream.
    - Observed SFS rows are indexed by folded allele count k, not frequency bins.
    - The current implementation fits one population and does not model admixture.

Input:
    - Count-indexed folded SFS TSV with population, n_pop, k_folded, count.
    - YAML configuration for model, selection, demography, training, and losses.

Output:
    - model.pt with model, selection, and demography state dictionaries.
    - run_config.yaml with the effective run configuration.
    - metrics.json with final loss metrics.
    - loss_history.tsv with epoch-level losses.
    - gamma_trajectory.tsv with time, gamma, population.
    - predicted_sfs.tsv with observed and predicted folded count classes.

Process Schematic:
    Folded SFS TSV
        -> load observed count classes and n_pop
        -> initialize PINN, SelectionModel, DemographyModel
        -> sample collocation points
        -> compute PDE, boundary, projected SFS, and smoothness losses
        -> optimize with Adam
        -> save model and tabular outputs

Example:
    python scripts/03_train_time_varying_gamma.py \
        --sfs data/processed/folded_sfs_nfe.tsv \
        --config configs/time_varying_selection.yaml \
        --output-dir results/time_varying_nfe_test \
        --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inverse_solver import SinglePopulationInverseSolver
from src.inference.training import load_yaml_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train one-population time-varying gamma KimuraPINN.")
    parser.add_argument("--sfs", required=True, help="Input folded SFS TSV.")
    parser.add_argument("--config", required=True, help="YAML configuration file.")
    parser.add_argument("--output-dir", required=True, help="Run output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    """Run single-population inverse fitting."""
    args = parse_args()
    config = load_yaml_config(args.config)
    solver = SinglePopulationInverseSolver(config)
    summary = solver.fit(args.sfs, args.output_dir, epochs=args.epochs, device=args.device)
    print(
        "Completed training: "
        f"population={summary['population']} n_pop={summary['n_pop']} "
        f"epochs={summary['epochs']} output={summary['run_dir']}"
    )


if __name__ == "__main__":
    main()

