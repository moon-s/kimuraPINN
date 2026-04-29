from pathlib import Path
import subprocess
import sys

import pandas as pd


def write_yaml(path: Path, data: str) -> Path:
    path.write_text(data)
    return path


def test_validation_pipeline_runs_tiny_settings(tmp_path: Path) -> None:
    simulation_config = write_yaml(
        tmp_path / "sim.yaml",
        """
seed: 5
simulation:
  population: toy
  n_pop: 12
  n_variants: 20
  n_generations: 3
initial_frequencies:
  mode: fixed
  value: 0.25
gamma:
  mode: constant
  gamma0: 0.5
demography:
  mode: constant
  nu: 1.0
output:
  gamma_points: 5
""",
    )
    training_config = write_yaml(
        tmp_path / "train.yaml",
        """
seed: 5
device: cpu
h: 0.5
model:
  hidden_dim: 8
  num_layers: 2
  fourier_features: 1
selection:
  mode: piecewise_linear
  learnable: true
  breakpoints: [0.0, 0.5, 1.0]
  values: [0.0, 0.0, 0.0]
demography:
  mode: constant
  nu: 1.0
  learnable: false
training:
  epochs: 2
  lr: 0.001
  batch_collocation: 4
  boundary_batch: 4
  gamma_grid_size: 5
  data_loss: mse
loss_weights:
  pde: 1.0
  data: 1.0
  boundary: 0.01
  gamma_smoothness: 0.01
projection:
  n_grid: 16
  max_zero_bins_for_loss: 2
  output_chunk_size: 4
output:
  gamma_points: 5
""",
    )
    output_dir = tmp_path / "validation"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/07_validate_time_varying_gamma.py",
            "--simulation-config",
            str(simulation_config),
            "--training-config",
            str(training_config),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Validation complete" in result.stdout

    expected = [
        "model.pt",
        "loss_history.tsv",
        "predicted_sfs.tsv",
        "inferred_gamma_trajectory.tsv",
        "true_gamma_trajectory.tsv",
        "gamma_recovery_metrics.json",
        "figures/true_vs_inferred_gamma.png",
        "figures/true_vs_inferred_gamma.pdf",
        "figures/simulated_sfs_fit.png",
        "figures/simulated_sfs_fit.pdf",
        "figures/gamma_error.png",
        "figures/gamma_error.pdf",
    ]
    for relative in expected:
        assert (output_dir / relative).exists()
    loss_history = pd.read_csv(output_dir / "loss_history.tsv", sep="\t")
    assert len(loss_history) == 2
