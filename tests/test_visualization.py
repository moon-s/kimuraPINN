from pathlib import Path

from src.visualization.plot_gamma import plot_gamma_trajectory
from src.visualization.plot_loss import plot_loss_history
from src.visualization.plot_sfs import plot_sfs_observed_vs_predicted, plot_sfs_residuals


def write_fake_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predicted_sfs.tsv").write_text(
        "population\tn_pop\tk_folded\tobserved_count\tpredicted_count\n"
        "toy\t10\t1\t4\t3.5\n"
        "toy\t10\t2\t2\t2.2\n"
        "toy\t10\t3\t1\t1.1\n"
        "toy\t10\t4\t0\t0.2\n"
        "toy\t10\t5\t1\t0.8\n"
    )
    (run_dir / "gamma_trajectory.tsv").write_text(
        "time\tgamma\tpopulation\n"
        "0.0\t0.0\ttoy\n"
        "0.5\t0.2\ttoy\n"
        "1.0\t0.1\ttoy\n"
    )
    (run_dir / "loss_history.tsv").write_text(
        "epoch\ttotal_loss\tpde_loss\tdata_loss\tboundary_loss\tgamma_smoothness_loss\n"
        "1\t10\t8\t1\t0.5\t0.1\n"
        "2\t5\t4\t0.5\t0.3\t0.05\n"
    )


def assert_png_pdf(figures_dir: Path, stem: str) -> None:
    assert (figures_dir / f"{stem}.png").exists()
    assert (figures_dir / f"{stem}.pdf").exists()


def test_visualization_functions_create_required_figures(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    figures_dir = run_dir / "figures"
    write_fake_run(run_dir)

    plot_sfs_observed_vs_predicted(run_dir / "predicted_sfs.tsv", figures_dir, log_y=True)
    plot_sfs_residuals(run_dir / "predicted_sfs.tsv", figures_dir)
    plot_gamma_trajectory(run_dir / "gamma_trajectory.tsv", figures_dir)
    plot_loss_history(run_dir / "loss_history.tsv", figures_dir)

    assert_png_pdf(figures_dir, "sfs_observed_vs_predicted")
    assert_png_pdf(figures_dir, "sfs_residuals")
    assert_png_pdf(figures_dir, "gamma_trajectory")
    assert_png_pdf(figures_dir, "loss_history")

