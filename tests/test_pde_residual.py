import torch

from src.models.pinn import KimuraPINN
from src.models.selection_model import SelectionModel
from src.pde.fokker_planck import compute_fokker_planck_residual


def test_fokker_planck_residual_is_finite_and_has_expected_shape() -> None:
    model = KimuraPINN(hidden_dim=16, num_layers=2, fourier_features=1)
    selection = SelectionModel(mode="constant", gamma=0.1, learnable=True)
    x = torch.linspace(0.05, 0.95, 10).reshape(-1, 1)
    t = torch.linspace(0.0, 1.0, 10).reshape(-1, 1)

    residual = compute_fokker_planck_residual(model, selection, x, t, nu=2.0)

    assert residual.shape == (10, 1)
    assert torch.all(torch.isfinite(residual))


def test_fokker_planck_residual_backpropagates_to_model_parameters() -> None:
    model = KimuraPINN(hidden_dim=16, num_layers=2)
    selection = SelectionModel(mode="constant", gamma=0.0, learnable=True)
    x = torch.linspace(0.1, 0.9, 6).reshape(-1, 1)
    t = torch.linspace(0.0, 0.5, 6).reshape(-1, 1)

    residual = compute_fokker_planck_residual(model, selection, x, t)
    loss = residual.pow(2).mean()
    loss.backward()

    model_grads = [param.grad for param in model.parameters() if param.requires_grad]
    selection_grads = [param.grad for param in selection.parameters() if param.requires_grad]
    assert any(grad is not None and torch.any(torch.isfinite(grad)) for grad in model_grads)
    assert any(grad is not None and torch.any(torch.isfinite(grad)) for grad in selection_grads)

