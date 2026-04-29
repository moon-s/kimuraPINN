#!/usr/bin/env python3
"""
Script: 01_vcf_to_folded_sfs.py

Purpose:
    Parse a gnomAD-style VCF, retain SNVs, project population-specific allele
    counts to a common cohort allele number, and write count-indexed folded SFS
    tables for downstream KimuraPINN analysis.

Biological Question:
    What is the observed folded allele-count spectrum for each population after
    harmonizing variable per-site allele numbers to a population-specific cohort
    sample size?

Assumptions:
    - Folded SFS is used because ancestral allele is unavailable.
    - Only SNVs are retained.
    - Allele count fields are taken from gnomAD-style INFO columns.
    - The main SFS is indexed by projected allele count, not frequency bins.

Input:
    - Path to a VCF or VCF.gz input file.
    - Required INFO fields: AC_<population>, AN_<population>, variant_type.

Output:
    - allele_counts.tsv with raw AC/AN, projected allele counts, folded counts,
      block IDs, and variant type.
    - folded_sfs_<population>.tsv with columns population, n_pop, k_folded,
      count.
    - folded_sfs_summary.json with variant and per-population summary counts.

Process Schematic:
    Input VCF
        -> filter SNVs
        -> extract AC/AN
        -> set n_pop = max AN per population
        -> project AC/AN to allele counts
        -> keep segregating projected variants
        -> construct folded count-indexed SFS
        -> save processed data

Example:
    python scripts/01_vcf_to_folded_sfs.py \
        --input data/raw/sample.vcf \
        --output-dir data/processed \
        --populations afr eas nfe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.sfs_io import write_tsv
from src.io.vcf_parser import build_population_allele_table, retained_snv_count, total_variant_count
from src.sfs.folded import make_all_folded_sfs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert VCF allele counts to folded SFS tables.")
    parser.add_argument("--input", required=True, help="Input VCF or VCF.gz file.")
    parser.add_argument("--output-dir", required=True, help="Directory for processed outputs.")
    parser.add_argument(
        "--populations",
        nargs="+",
        default=["afr", "eas", "nfe"],
        help="Population suffixes to parse from AC_<pop>/AN_<pop> INFO fields.",
    )
    return parser.parse_args()


def main() -> None:
    """Run VCF parsing and folded SFS construction."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    populations = [population.lower() for population in args.populations]

    allele_counts, n_by_population, valid_raw_by_population = build_population_allele_table(
        args.input,
        populations,
    )
    allele_counts_path = output_dir / "allele_counts.tsv"
    write_tsv(allele_counts, allele_counts_path)

    sfs_tables = make_all_folded_sfs(allele_counts, populations)
    population_summary: dict[str, dict[str, int]] = {}
    for population, sfs in sfs_tables.items():
        write_tsv(sfs, output_dir / f"folded_sfs_{population}.tsv")
        population_summary[population] = {
            "n_pop": int(n_by_population.get(population, 0)),
            "n_raw_snv_rows_with_an": int(valid_raw_by_population.get(population, 0)),
            "n_projected_segregating": int(
                (allele_counts["population"] == population).sum()
                if not allele_counts.empty
                else 0
            ),
            "sfs_total_count": int(sfs["count"].sum()) if not sfs.empty else 0,
            "n_count_bins": int(len(sfs)),
        }

    summary = {
        "n_variants_total": int(total_variant_count(args.input)),
        "n_snvs_retained": int(retained_snv_count(args.input)),
        "n_snv_sites_retained": int(allele_counts[["chrom", "pos", "ref", "alt"]].drop_duplicates().shape[0])
        if not allele_counts.empty
        else 0,
        "populations": population_summary,
        "folded": True,
        "sfs_index": "allele_count",
    }
    with (output_dir / "folded_sfs_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
