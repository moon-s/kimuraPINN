"""Generate synthetic count-indexed folded SFS data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from src.simulation.simulate_wf import (
    make_gamma_schedule,
    make_gamma_trajectory,
    make_nu_schedule,
    simulate_wright_fisher,
)


def build_folded_sfs_from_counts(ac_values, n_pop: int, population: str) -> pd.DataFrame:
    """Build a count-indexed folded SFS from unfolded allele counts."""
    max_k = n_pop // 2
    counts = {k: 0 for k in range(1, max_k + 1)}
    for ac in ac_values:
        ac_int = int(ac)
        if not (1 <= ac_int <= n_pop - 1):
            continue
        k_folded = min(ac_int, n_pop - ac_int)
        counts[k_folded] += 1
    return pd.DataFrame(
        [
            {
                "population": population,
                "n_pop": n_pop,
                "k_folded": k,
                "count": counts[k],
            }
            for k in range(1, max_k + 1)
        ]
    )


def simulate_single_population_sfs(config: Dict[str, Any], output_dir: str | Path) -> Dict[str, Any]:
    """Simulate variants and save folded SFS validation inputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    simulation_cfg = config.get("simulation", {})
    population = str(simulation_cfg.get("population", "sim"))
    n_pop = int(simulation_cfg.get("n_pop", 100))
    n_variants = int(simulation_cfg.get("n_variants", 1000))
    n_generations = int(simulation_cfg.get("n_generations", 50))
    seed = int(config.get("seed", 123))

    gamma_cfg = config.get("gamma", {"mode": "constant", "gamma0": 0.0})
    demography_cfg = config.get("demography", {"mode": "constant", "nu": 1.0})
    gamma_schedule = make_gamma_schedule(gamma_cfg)
    nu_schedule = make_nu_schedule(demography_cfg)
    simulated = simulate_wright_fisher(
        n_variants=n_variants,
        n_pop=n_pop,
        n_generations=n_generations,
        gamma_schedule=gamma_schedule,
        nu_schedule=nu_schedule,
        initial_frequency_config=config.get("initial_frequencies", {}),
        seed=seed,
    )

    folded = [min(int(ac), n_pop - int(ac)) if 1 <= int(ac) <= n_pop - 1 else 0 for ac in simulated["ac"]]
    variants = pd.DataFrame(
        {
            "variant_id": [f"var_{i + 1}" for i in range(n_variants)],
            "ac": simulated["ac"],
            "an": simulated["an"],
            "af": simulated["af"],
            "k_folded": folded,
            "final_frequency": simulated["final_frequency"],
            "gamma_mode": gamma_schedule.mode,
        }
    )
    sfs = build_folded_sfs_from_counts(simulated["ac"], n_pop=n_pop, population=population)
    times, gamma = make_gamma_trajectory(gamma_schedule, int(config.get("output", {}).get("gamma_points", 101)))
    gamma_frame = pd.DataFrame({"time": times, "gamma": gamma, "population": population})

    variants.to_csv(output_path / "simulated_variants.tsv", sep="\t", index=False)
    sfs.to_csv(output_path / "folded_sfs_simulated.tsv", sep="\t", index=False)
    gamma_frame.to_csv(output_path / "true_gamma_trajectory.tsv", sep="\t", index=False)
    with (output_path / "simulation_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    segregating = int(sfs["count"].sum())
    metrics = {
        "n_variants": n_variants,
        "n_segregating": segregating,
        "n_monomorphic": n_variants - segregating,
        "n_pop": n_pop,
        "gamma_mode": gamma_schedule.mode,
    }
    with (output_path / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "output_dir": str(output_path),
        "folded_sfs": str(output_path / "folded_sfs_simulated.tsv"),
        "true_gamma": str(output_path / "true_gamma_trajectory.tsv"),
        "metrics": metrics,
    }

