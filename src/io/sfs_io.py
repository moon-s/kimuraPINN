"""Read and write SFS tables."""

from pathlib import Path
from typing import Union

import pandas as pd


def write_tsv(frame: pd.DataFrame, path: Union[str, Path]) -> None:
    """Write a dataframe as tab-separated text with parent directories created."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, sep="\t", index=False)
