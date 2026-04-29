from pathlib import Path

import pandas as pd

from src.inference.two_pop_solver import train_two_population_marginal_sfs


def write_sfs(path: Path, population: str, counts: list[int]) -> None:
    rows = ["population\tn_pop\tk_folded\tcount"]
    for index, count in enumerate(counts, start=1):
        rows.append(f"{population}\t10\t{index}\t{count}")
    path.write_text("\n".join(rows) + "\n")


def tiny_two_pop_config() -> dict:
    return {
        "seed": 11,
        "device": "cpu",
        "h": 0.5,
        "model": {
            "hidden_dim": 8,
            "num_layers": 2,
            "fourier_features": 1,
        },
        "selection": {
            "mode": "piecewise_linear",
            "learnable": True,
            "breakpoints": [0.0, 0.5, 1.0],
            "values": [0.0, 0.0, 0.0],
        },
        "demography": {
            "mode": "constant",
            "nu": 1.0,
            "learnable": False,
        },
        "admixture": {
            "learnable": True,
            "initial_matrix": [[0.0, 0.001], [0.001, 0.0]],
        },
        "training": {
            "epochs": 2,
            "lr": 0.001,
            "batch_collocation": 4,
            "boundary_batch": 4,
            "gamma_grid_size": 5,
            "data_loss": "mse",
        },
        "loss_weights": {
            "pde": 1.0,
            "data_a": 1.0,
            "data_b": 1.0,
            "boundary": 0.01,
            "gamma_smoothness": 0.01,
            "migration": 0.001,
        },
        "projection": {
            "n_grid": 8,
            "max_zero_bins_for_loss": 2,
            "output_chunk_size": 2,
        },
        "output": {
            "gamma_points": 5,
        },
    }


def test_two_pop_training_smoke_creates_outputs(tmp_path: Path) -> None:
    sfs_a = tmp_path / "folded_sfs_afr.tsv"
    sfs_b = tmp_path / "folded_sfs_nfe.tsv"
    output_dir = tmp_path / "admixture_run"
    write_sfs(sfs_a, "afr", [4, 2, 1, 0, 1])
    write_sfs(sfs_b, "nfe", [3, 1, 2, 1, 0])

    summary = train_two_population_marginal_sfs(
        sfs_a=sfs_a,
        sfs_b=sfs_b,
        population_a="afr",
        population_b="nfe",
        output_dir=output_dir,
        config=tiny_two_pop_config(),
    )

    assert summary["population_a"] == "afr"
    assert summary["population_b"] == "nfe"
    for filename in [
        "model.pt",
        "run_config.yaml",
        "metrics.json",
        "loss_history.tsv",
        "gamma_trajectory.tsv",
        "migration_matrix.tsv",
        "predicted_sfs_afr.tsv",
        "predicted_sfs_nfe.tsv",
    ]:
        assert (output_dir / filename).exists()

    loss_history = pd.read_csv(output_dir / "loss_history.tsv", sep="\t")
    assert list(loss_history.columns) == [
        "epoch",
        "total_loss",
        "pde_loss",
        "data_loss_a",
        "data_loss_b",
        "boundary_loss",
        "gamma_smoothness_loss",
        "migration_reg_loss",
    ]
    assert len(loss_history) == 2
    numeric_losses = loss_history.drop(columns=["epoch"])
    assert numeric_losses.notna().all().all()
    assert (numeric_losses < float("inf")).all().all()

    gamma = pd.read_csv(output_dir / "gamma_trajectory.tsv", sep="\t")
    assert set(gamma["population"]) == {"afr", "nfe"}

    migration = pd.read_csv(output_dir / "migration_matrix.tsv", sep="\t")
    assert set(migration.columns) == {"source_population", "target_population", "migration_rate"}
    assert len(migration) == 4

    pred_a = pd.read_csv(output_dir / "predicted_sfs_afr.tsv", sep="\t")
    pred_b = pd.read_csv(output_dir / "predicted_sfs_nfe.tsv", sep="\t")
    assert {"observed_count", "predicted_count"}.issubset(pred_a.columns)
    assert {"observed_count", "predicted_count"}.issubset(pred_b.columns)
