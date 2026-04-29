from pathlib import Path

import torch
from torch import nn

from src.sfs.projection import (
    binomial_log_prob,
    fold_sfs,
    load_folded_sfs_tsv,
    make_quadrature_grid,
    project_density_to_observed_k,
    project_density_to_sfs,
    project_model_to_folded_sfs,
)


class ConstantDensity(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del t, context
        return torch.ones_like(x)


def test_binomial_projection_output_shape_and_finite_values() -> None:
    x_grid, weights = make_quadrature_grid(n_points=64)
    phi = torch.ones_like(x_grid)

    unfolded = project_density_to_sfs(phi, x_grid, weights, n=10)

    assert unfolded.shape == (11,)
    assert torch.all(torch.isfinite(unfolded))
    assert torch.isclose(unfolded.sum(), torch.tensor(1.0), atol=1e-5)


def test_binomial_log_prob_broadcasts() -> None:
    x = torch.tensor([[0.2, 0.8]])
    k = torch.tensor([[1], [2]])

    values = binomial_log_prob(k, 4, x)

    assert values.shape == (2, 2)
    assert torch.all(torch.isfinite(values))


def test_fold_sfs_even_n_excludes_monomorphic_and_keeps_midpoint_once() -> None:
    unfolded = torch.arange(11, dtype=torch.float32)

    folded = fold_sfs(unfolded)

    assert folded.shape == (5,)
    assert torch.allclose(folded, torch.tensor([10.0, 10.0, 10.0, 10.0, 5.0]))


def test_fold_sfs_odd_n_excludes_monomorphic() -> None:
    unfolded = torch.arange(12, dtype=torch.float32)

    folded = fold_sfs(unfolded)

    assert folded.shape == (5,)
    assert torch.allclose(folded, torch.tensor([11.0, 11.0, 11.0, 11.0, 11.0]))


def test_sparse_observed_k_projection_works_for_large_n() -> None:
    x_grid, weights = make_quadrature_grid(n_points=128)
    phi = torch.ones_like(x_grid)
    k_values = torch.tensor([1, 10, 100, 1000], dtype=torch.long)

    sparse = project_density_to_observed_k(
        phi,
        x_grid,
        weights,
        n=50_000,
        k_values=k_values,
        folded=True,
    )

    assert sparse.shape == (4,)
    assert torch.all(torch.isfinite(sparse))
    assert torch.isclose(sparse.sum(), torch.tensor(1.0), atol=1e-5)


def test_project_model_to_folded_sfs_returns_count_indexed_vector() -> None:
    model = ConstantDensity()

    folded = project_model_to_folded_sfs(model, t_eval=0.0, n=10, n_grid=64)

    assert folded.shape == (5,)
    assert torch.all(torch.isfinite(folded))


def test_load_folded_sfs_tsv_returns_counts_metadata(tmp_path: Path) -> None:
    path = tmp_path / "folded_sfs_afr.tsv"
    path.write_text(
        "population\tn_pop\tk_folded\tcount\n"
        "afr\t10\t1\t3\n"
        "afr\t10\t2\t0\n"
        "afr\t10\t3\t5\n"
        "afr\t10\t4\t1\n"
        "afr\t10\t5\t2\n"
    )

    counts, n_pop, k_values, population = load_folded_sfs_tsv(path)

    assert n_pop == 10
    assert population == "afr"
    assert torch.equal(k_values, torch.tensor([1, 2, 3, 4, 5]))
    assert torch.equal(counts, torch.tensor([3.0, 0.0, 5.0, 1.0, 2.0]))
