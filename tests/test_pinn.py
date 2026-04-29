import torch

from src.models.pinn import KimuraPINN
from src.pde.boundary_conditions import boundary_loss_absorbing


def test_kimura_pinn_forward_shape_and_nonnegative_output() -> None:
    model = KimuraPINN(hidden_dim=16, num_layers=2, fourier_features=2)
    x = torch.linspace(0.1, 0.9, 8).reshape(-1, 1)
    t = torch.linspace(0.0, 1.0, 8).reshape(-1, 1)

    phi = model(x, t)

    assert phi.shape == (8, 1)
    assert torch.all(phi >= 0.0)


def test_absorbing_boundary_loss_is_finite() -> None:
    model = KimuraPINN(hidden_dim=12, num_layers=2)
    t = torch.linspace(0.0, 1.0, 5).reshape(-1, 1)

    loss = boundary_loss_absorbing(model, t)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

