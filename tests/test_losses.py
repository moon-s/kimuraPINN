import torch

from src.inference.losses import (
    gamma_smoothness_loss,
    mse_sfs_loss,
    poisson_nll_sfs_loss,
)
from src.models.selection_model import SelectionModel


def test_mse_sfs_loss_near_zero_when_predictions_match_observed() -> None:
    observed = torch.tensor([1.0, 2.0, 3.0])
    predicted = observed.clone()

    loss = mse_sfs_loss(predicted, observed)

    assert loss < 1e-12


def test_poisson_nll_sfs_loss_is_finite() -> None:
    predicted = torch.tensor([0.5, 2.0, 4.0])
    observed = torch.tensor([0.0, 3.0, 5.0])

    loss = poisson_nll_sfs_loss(predicted, observed)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gamma_smoothness_loss_wrapper_is_finite() -> None:
    selection = SelectionModel(
        mode="piecewise_linear",
        breakpoints=[0.0, 0.5, 1.0],
        values=[0.0, 1.0, 0.0],
    )
    t_grid = torch.linspace(0.0, 1.0, 9).reshape(-1, 1)

    loss = gamma_smoothness_loss(selection, t_grid)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

