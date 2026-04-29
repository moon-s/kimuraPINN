"""Count-indexed folded site frequency spectrum construction."""

from __future__ import annotations

from typing import Tuple, Union

import pandas as pd


SFS_COLUMNS = ["population", "n_pop", "k_folded", "count"]


def make_count_folded_sfs(allele_counts: pd.DataFrame, population: str) -> pd.DataFrame:
    """Construct a folded SFS indexed by allele count for one population.

    The output includes every folded count class k=1..floor(n_pop/2), even when
    the observed count is zero.
    """
    population = population.lower()
    pop_rows = allele_counts.loc[allele_counts["population"] == population]
    if pop_rows.empty:
        return pd.DataFrame(columns=SFS_COLUMNS)

    n_values = sorted(pop_rows["n_pop"].dropna().astype(int).unique())
    if len(n_values) != 1:
        raise ValueError(f"Population {population} has inconsistent n_pop values: {n_values}")
    n_pop = n_values[0]
    max_k = n_pop // 2
    counts = pop_rows.groupby("k_folded").size().to_dict()
    rows = [
        {
            "population": population,
            "n_pop": n_pop,
            "k_folded": k,
            "count": int(counts.get(k, 0)),
        }
        for k in range(1, max_k + 1)
    ]
    return pd.DataFrame(rows, columns=SFS_COLUMNS)


def make_all_folded_sfs(
    allele_counts: pd.DataFrame, populations: Union[list[str], Tuple[str, ...]]
) -> dict[str, pd.DataFrame]:
    """Construct folded SFS tables for all requested populations."""
    return {
        population.lower(): make_count_folded_sfs(allele_counts, population.lower())
        for population in populations
    }
