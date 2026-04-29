"""Thin inverse-solver interface for single-population KimuraPINN fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.inference.training import train_single_population


class SinglePopulationInverseSolver:
    """Coordinate one-population inverse fitting without owning training loops."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def fit(
        self,
        sfs_path: str | Path,
        output_dir: str | Path,
        epochs: Optional[int] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fit the configured model to one folded SFS file."""
        return train_single_population(
            sfs_path=sfs_path,
            output_dir=output_dir,
            config=self.config,
            epochs_override=epochs,
            device=device,
        )

