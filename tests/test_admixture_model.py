import torch

from src.models.admixture_model import AdmixtureModel


def test_admixture_model_matrix_shape_zero_diagonal_nonnegative() -> None:
    model = AdmixtureModel(initial_matrix=[[0.0, 0.002], [0.003, 0.0]], learnable=True)

    matrix = model()

    assert matrix.shape == (2, 2)
    assert torch.allclose(torch.diag(matrix), torch.zeros(2))
    assert torch.all(matrix[~torch.eye(2, dtype=torch.bool)] >= 0)


def test_admixture_model_gradients_exist_for_learnable_rates() -> None:
    model = AdmixtureModel(initial_matrix=[[0.0, 0.002], [0.003, 0.0]], learnable=True)

    loss = model().sum()
    loss.backward()

    assert model.raw_offdiag.grad is not None
    assert torch.all(torch.isfinite(model.raw_offdiag.grad))


def test_fixed_admixture_model_has_no_trainable_parameters() -> None:
    model = AdmixtureModel(initial_matrix=[[0.0, 0.0], [0.001, 0.0]], learnable=False)

    matrix = model.migration_matrix()

    assert matrix[0, 1] == 0.0
    assert matrix[1, 0] >= 0.0
    assert list(model.parameters()) == []

