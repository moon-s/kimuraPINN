import torch

from src.models.demography_model import DemographyModel


def test_constant_demography_returns_positive_nu() -> None:
    model = DemographyModel(mode="constant", nu=1.5)
    t = torch.linspace(0.0, 1.0, 5).reshape(-1, 1)

    nu = model(t)

    assert nu.shape == (5, 1)
    assert torch.all(nu > 0)
    assert torch.allclose(nu, torch.full_like(t, 1.5))


def test_epoch_demography_returns_positive_piecewise_values() -> None:
    model = DemographyModel(
        mode="epoch",
        breakpoints=[0.0, 0.5, 1.0],
        values=[1.0, 2.0],
    )
    t = torch.tensor([[0.1], [0.49], [0.5], [0.9]])

    nu = model(t)

    assert torch.all(nu > 0)
    assert torch.allclose(nu.reshape(-1), torch.tensor([1.0, 1.0, 2.0, 2.0]))

