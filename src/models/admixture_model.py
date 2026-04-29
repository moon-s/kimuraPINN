"""Migration/admixture parameterization for two-population Kimura models."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class AdmixtureModel(nn.Module):
    """Two-population non-negative migration matrix with zero diagonal."""

    def __init__(
        self,
        initial_matrix: Sequence[Sequence[float]] | None = None,
        learnable: bool = True,
        min_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.k = 2
        self.learnable = bool(learnable)
        self.min_rate = float(min_rate)
        matrix = torch.tensor(
            initial_matrix if initial_matrix is not None else [[0.0, 0.001], [0.001, 0.0]],
            dtype=torch.float32,
        )
        if matrix.shape != (2, 2):
            raise ValueError("Milestone 6A supports a 2x2 initial_matrix only")
        if torch.any(matrix < 0):
            raise ValueError("migration rates must be non-negative")
        matrix = matrix.clone()
        matrix.fill_diagonal_(0.0)
        offdiag = torch.tensor([matrix[0, 1], matrix[1, 0]], dtype=torch.float32)
        if learnable:
            self.raw_offdiag = nn.Parameter(self._inverse_softplus((offdiag - self.min_rate).clamp_min(0.0)))
        else:
            self.register_buffer("offdiag", offdiag)

    @staticmethod
    def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.expm1(value.clamp_min(1e-12)))

    def migration_matrix(self) -> torch.Tensor:
        """Return a [2, 2] matrix where entry [i, j] is migration j -> i."""
        if hasattr(self, "raw_offdiag"):
            offdiag = torch.nn.functional.softplus(self.raw_offdiag) + self.min_rate
        else:
            offdiag = self.offdiag
        matrix = torch.zeros(2, 2, device=offdiag.device, dtype=offdiag.dtype)
        matrix[0, 1] = offdiag[0]
        matrix[1, 0] = offdiag[1]
        return matrix

    def forward(self) -> torch.Tensor:
        """Alias for migration_matrix."""
        return self.migration_matrix()

