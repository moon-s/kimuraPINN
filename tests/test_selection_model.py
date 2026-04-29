import torch

from src.models.selection_model import SelectionModel


def test_constant_selection_model_shape_and_value() -> None:
    model = SelectionModel(mode="constant", gamma=0.25, learnable=False)
    t = torch.linspace(0.0, 1.0, 4).reshape(-1, 1)

    gamma = model(t)

    assert gamma.shape == (4, 1)
    assert torch.allclose(gamma, torch.full_like(t, 0.25))


def test_piecewise_linear_selection_is_continuous_at_breakpoint() -> None:
    model = SelectionModel(
        mode="piecewise_linear",
        breakpoints=[0.0, 1.0, 2.0],
        values=[0.0, 2.0, 0.0],
        learnable=False,
    )
    t = torch.tensor([[1.0 - 1e-5], [1.0], [1.0 + 1e-5]])

    gamma = model(t)

    assert gamma.shape == (3, 1)
    assert torch.all(torch.isfinite(gamma))
    assert torch.max(torch.abs(gamma - 2.0)) < 3e-5


def test_neural_network_selection_model_shape() -> None:
    model = SelectionModel(mode="neural_network", hidden_dim=8, num_layers=2)
    t = torch.linspace(0.0, 1.0, 7).reshape(-1, 1)

    gamma = model(t)

    assert gamma.shape == (7, 1)
    assert torch.all(torch.isfinite(gamma))


def test_smoothness_loss_is_finite_for_all_modes() -> None:
    t_grid = torch.linspace(0.0, 1.0, 11).reshape(-1, 1)
    models = [
        SelectionModel(mode="constant", gamma=0.1, learnable=False),
        SelectionModel(
            mode="piecewise_linear",
            breakpoints=[0.0, 0.5, 1.0],
            values=[0.0, 1.0, 0.0],
        ),
        SelectionModel(mode="neural_network", hidden_dim=8, num_layers=1),
    ]

    for model in models:
        loss = model.smoothness_loss(t_grid)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

