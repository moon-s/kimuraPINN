import torch

from src.models.admixture_model import AdmixtureModel
from src.models.demography_model import DemographyModel
from src.models.pinn import MultiPopKimuraPINN
from src.models.selection_model import SelectionModel
from src.pde.kimura_multipop import compute_two_pop_fokker_planck_residual


def test_two_pop_pinn_forward_pass_shape_and_positive() -> None:
    model = MultiPopKimuraPINN(n_populations=2, hidden_dim=12, num_layers=2, fourier_features=1)
    x = torch.tensor([[0.2, 0.3], [0.4, 0.7], [0.8, 0.1]], dtype=torch.float32)
    t = torch.linspace(0.0, 1.0, 3).reshape(-1, 1)

    phi = model(x, t)

    assert phi.shape == (3, 1)
    assert torch.all(phi >= 0.0)


def test_two_pop_pde_residual_is_finite_with_migration() -> None:
    model = MultiPopKimuraPINN(n_populations=2, hidden_dim=12, num_layers=2)
    selection = [
        SelectionModel(mode="constant", gamma=0.1, learnable=True),
        SelectionModel(mode="constant", gamma=-0.1, learnable=True),
    ]
    demography = [
        DemographyModel(mode="constant", nu=1.0),
        DemographyModel(mode="constant", nu=1.5),
    ]
    admixture = AdmixtureModel(initial_matrix=[[0.0, 0.001], [0.002, 0.0]], learnable=True)
    x = torch.tensor([[0.2, 0.3], [0.4, 0.7], [0.8, 0.1]], dtype=torch.float32)
    t = torch.linspace(0.0, 1.0, 3).reshape(-1, 1)

    residual = compute_two_pop_fokker_planck_residual(model, selection, demography, admixture, x, t)
    loss = residual.pow(2).mean()
    loss.backward()

    assert residual.shape == (3, 1)
    assert torch.all(torch.isfinite(residual))
    assert admixture.raw_offdiag.grad is not None


def test_two_pop_pde_residual_zero_migration_is_finite() -> None:
    model = MultiPopKimuraPINN(n_populations=2, hidden_dim=10, num_layers=2)
    selection = [SelectionModel(mode="constant", gamma=0.0), SelectionModel(mode="constant", gamma=0.0)]
    demography = [DemographyModel(mode="constant", nu=1.0), DemographyModel(mode="constant", nu=1.0)]
    admixture = AdmixtureModel(initial_matrix=[[0.0, 0.0], [0.0, 0.0]], learnable=False)
    x = torch.tensor([[0.25, 0.25], [0.5, 0.5]], dtype=torch.float32)
    t = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    residual = compute_two_pop_fokker_planck_residual(model, selection, demography, admixture, x, t)

    assert residual.shape == (2, 1)
    assert torch.all(torch.isfinite(residual))

