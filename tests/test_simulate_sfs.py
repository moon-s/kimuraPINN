from pathlib import Path

import pandas as pd

from src.simulation.simulate_sfs import build_folded_sfs_from_counts, simulate_single_population_sfs


def test_folded_sfs_excludes_monomorphic_classes() -> None:
    sfs = build_folded_sfs_from_counts([0, 1, 2, 9, 10], n_pop=10, population="toy")

    assert list(sfs["k_folded"]) == [1, 2, 3, 4, 5]
    assert int(sfs["count"].sum()) == 3
    assert int(sfs.loc[sfs["k_folded"] == 1, "count"].item()) == 2


def test_simulate_single_population_sfs_writes_outputs(tmp_path: Path) -> None:
    config = {
        "seed": 3,
        "simulation": {
            "population": "toy",
            "n_pop": 20,
            "n_variants": 30,
            "n_generations": 4,
        },
        "initial_frequencies": {"mode": "fixed", "value": 0.2},
        "gamma": {"mode": "step", "gamma_before": 0.0, "gamma_after": 1.0, "t_change": 0.5},
        "demography": {"mode": "constant", "nu": 1.0},
        "output": {"gamma_points": 5},
    }

    summary = simulate_single_population_sfs(config, tmp_path)

    assert Path(summary["folded_sfs"]).exists()
    assert Path(summary["true_gamma"]).exists()
    assert (tmp_path / "simulated_variants.tsv").exists()
    assert (tmp_path / "simulation_config.yaml").exists()
    assert (tmp_path / "metrics.json").exists()
    variants = pd.read_csv(tmp_path / "simulated_variants.tsv", sep="\t")
    assert variants["ac"].between(0, 20).all()
    sfs = pd.read_csv(tmp_path / "folded_sfs_simulated.tsv", sep="\t")
    assert int(sfs["count"].sum()) == int(((variants["ac"] > 0) & (variants["ac"] < 20)).sum())

