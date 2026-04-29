from pathlib import Path

import pandas as pd

from src.inference.training import train_single_population


def write_tiny_sfs(path: Path) -> None:
    path.write_text(
        "population\tn_pop\tk_folded\tcount\n"
        "toy\t10\t1\t4\n"
        "toy\t10\t2\t2\n"
        "toy\t10\t3\t1\n"
        "toy\t10\t4\t0\n"
        "toy\t10\t5\t1\n"
    )


def tiny_config() -> dict:
    return {
        "seed": 7,
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
            "data": 1.0,
            "boundary": 0.01,
            "gamma_smoothness": 0.01,
        },
        "projection": {
            "n_grid": 16,
            "max_zero_bins_for_loss": 2,
            "output_chunk_size": 3,
        },
        "output": {
            "gamma_points": 5,
        },
    }


def test_short_training_run_creates_outputs(tmp_path: Path) -> None:
    sfs_path = tmp_path / "folded_sfs_toy.tsv"
    output_dir = tmp_path / "run"
    write_tiny_sfs(sfs_path)

    summary = train_single_population(sfs_path, output_dir, tiny_config())

    assert summary["population"] == "toy"
    assert (output_dir / "model.pt").exists()
    assert (output_dir / "run_config.yaml").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "loss_history.tsv").exists()
    assert (output_dir / "gamma_trajectory.tsv").exists()
    assert (output_dir / "predicted_sfs.tsv").exists()

    loss_history = pd.read_csv(output_dir / "loss_history.tsv", sep="\t")
    assert list(loss_history.columns) == [
        "epoch",
        "total_loss",
        "pde_loss",
        "data_loss",
        "boundary_loss",
        "gamma_smoothness_loss",
    ]
    assert len(loss_history) == 2

    predicted = pd.read_csv(output_dir / "predicted_sfs.tsv", sep="\t")
    assert {"observed_count", "predicted_count"}.issubset(predicted.columns)
    assert len(predicted) == 5

