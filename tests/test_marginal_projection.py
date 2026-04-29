import torch

from src.models.pinn import MultiPopKimuraPINN
from src.sfs.marginal_projection import project_two_pop_density_to_marginal_folded_sfs


class ConstantTwoPopDensity(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del t, context
        return torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)


def test_marginal_projection_returns_two_finite_folded_sfs_arrays() -> None:
    model = ConstantTwoPopDensity()

    sfs_1, sfs_2 = project_two_pop_density_to_marginal_folded_sfs(
        model,
        t_eval=0.0,
        n_pop_1=10,
        n_pop_2=11,
        n_grid=16,
    )

    assert sfs_1.shape == (5,)
    assert sfs_2.shape == (5,)
    assert torch.all(torch.isfinite(sfs_1))
    assert torch.all(torch.isfinite(sfs_2))


def test_marginal_projection_sparse_observed_k_for_large_n() -> None:
    model = MultiPopKimuraPINN(n_populations=2, hidden_dim=8, num_layers=2)
    observed_k_1 = torch.tensor([1, 2, 10])
    observed_k_2 = torch.tensor([1, 5])

    sfs_1, sfs_2 = project_two_pop_density_to_marginal_folded_sfs(
        model,
        t_eval=1.0,
        n_pop_1=50_000,
        n_pop_2=60_000,
        n_grid=12,
        observed_k_1=observed_k_1,
        observed_k_2=observed_k_2,
    )

    assert sfs_1.shape == (3,)
    assert sfs_2.shape == (2,)
    assert torch.all(torch.isfinite(sfs_1))
    assert torch.all(torch.isfinite(sfs_2))

