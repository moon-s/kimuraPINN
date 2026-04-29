"""Summary helpers for count-indexed folded SFS outputs."""

from __future__ import annotations

import pandas as pd


def total_variant_count(sfs: pd.DataFrame) -> int:
    """Return the total number of variants represented by an SFS table."""
    if sfs.empty:
        return 0
    return int(sfs["count"].sum())


def singleton_count(sfs: pd.DataFrame) -> int:
    """Return the count in the folded singleton class."""
    if sfs.empty:
        return 0
    singletons = sfs.loc[sfs["k_folded"] == 1, "count"]
    return int(singletons.sum()) if not singletons.empty else 0

