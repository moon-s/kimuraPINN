#!/usr/bin/env python3
"""
Script: 04_train_admixture_pinn.py

Purpose:
    Train a two-population KimuraPINN with migration/admixture coupling against
    two marginal count-indexed folded SFS files.

Biological Question:
    Can a two-population Kimura diffusion model fit marginal folded SFS from two
    populations while estimating population-specific gamma(t) trajectories and
    effective coupling rates between populations?

Assumptions:
    - Folded SFS is used because ancestral allele is unavailable.
    - Only SNVs are retained upstream by the VCF-to-SFS pipeline.
    - Allele count fields are taken from gnomAD-style INFO columns upstream.
    - This milestone uses marginal SFS only, not joint 2D SFS.
    - Migration estimates are effective coupling parameters, not definitive
      migration rates.

Input:
    - Two folded SFS TSV files with population, n_pop, k_folded, count.
    - A YAML config for two-population model, selection, demography, migration,
      projection, and loss weights.

Output:
    - model.pt
    - run_config.yaml
    - metrics.json
    - loss_history.tsv
    - gamma_trajectory.tsv
    - migration_matrix.tsv
    - predicted_sfs_<population>.tsv for both populations
    - figures/ directory placeholder

Process Schematic:
    Marginal folded SFS A and B
        -> initialize two-pop PINN, selection, demography, migration matrix
        -> sample two-dimensional collocation points
        -> compute two-pop PDE residual
        -> project joint density to marginal SFS for each population
        -> optimize combined loss with Adam
        -> save fitted trajectories, matrix, and SFS predictions

Example:
    python scripts/04_train_admixture_pinn.py \
        --sfs-a data/processed/folded_sfs_afr.tsv \
        --sfs-b data/processed/folded_sfs_nfe.tsv \
        --population-a afr \
        --population-b nfe \
        --config configs/two_pop_admixture.yaml \
        --output-dir results/admixture_afr_nfe_test \
        --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.two_pop_solver import train_two_population_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a two-population marginal-SFS admixture PINN.")
    parser.add_argument("--sfs-a", required=True, help="Folded SFS TSV for population A.")
    parser.add_argument("--sfs-b", required=True, help="Folded SFS TSV for population B.")
    parser.add_argument("--population-a", required=True, help="Population A label.")
    parser.add_argument("--population-b", required=True, help="Population B label.")
    parser.add_argument("--config", required=True, help="Two-population training config.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--device", default=None, help="Override device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_two_population_from_config(
        sfs_a=args.sfs_a,
        sfs_b=args.sfs_b,
        population_a=args.population_a,
        population_b=args.population_b,
        config_path=args.config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        device=args.device,
    )
    print(
        "Completed two-pop training: "
        f"{summary['population_a']}-{summary['population_b']} "
        f"epochs={summary['epochs']} output={summary['run_dir']}"
    )


if __name__ == "__main__":
    main()

