from __future__ import annotations

import pandas as pd


def iceberg_cube_buc(
    df: pd.DataFrame,
    dims: list[str],
    measure_col: str,
    min_support: int,
) -> pd.DataFrame:
    """Compute an iceberg cube using a top-down BUC recursion with pruning.

    This implementation assumes ``measure_col`` is a binary target (0/1) or
    numeric where mean is meaningful. It returns:
    - one column per dimension (values or 'ALL')
    - support: number of rows
    - positive_count: sum(measure_col) when binary
    - diabetes_rate: mean(measure_col)

    Pruning: stop exploring a branch when support < ``min_support``.
    """

    if min_support <= 0:
        raise ValueError("min_support must be > 0")

    if not dims:
        support = int(len(df))
        if support < min_support:
            return pd.DataFrame(columns=["support", "positive_count", "diabetes_rate"])
        m = pd.to_numeric(df[measure_col], errors="coerce")
        out = pd.DataFrame(
            {
                "support": [support],
                "positive_count": [float(m.sum(skipna=True))],
                "diabetes_rate": [float(m.mean(skipna=True))],
            }
        )
        return out

    for c in dims + [measure_col]:
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    dims = list(dims)
    results: list[dict] = []
    all_dims = dims.copy()

    def _record(current_df: pd.DataFrame, combo: dict) -> None:
        support = int(len(current_df))
        if support < min_support:
            return

        m = pd.to_numeric(current_df[measure_col], errors="coerce")
        row = {dim: combo.get(dim, "ALL") for dim in all_dims}
        row["support"] = support
        row["positive_count"] = float(m.sum(skipna=True))
        row["diabetes_rate"] = float(m.mean(skipna=True))
        results.append(row)

    def _buc(current_df: pd.DataFrame, remaining_dims: list[str], combo: dict) -> None:
        support = int(len(current_df))
        if support < min_support:
            return

        _record(current_df, combo)

        # Top-down recursion: choose next dimension in order
        for i, dim in enumerate(remaining_dims):
            # groupby over the chosen dimension, then recurse on the suffix
            for val, group_df in current_df.groupby(dim, dropna=False):
                new_combo = combo.copy()
                # Keep NaN explicit (use pandas NA/None) to avoid losing info
                new_combo[dim] = val
                _buc(group_df, remaining_dims[i + 1 :], new_combo)

    _buc(df, dims, {})

    out = pd.DataFrame(results)
    if out.empty:
        return out

    # Highest diabetes_rate first, then largest support
    return out.sort_values(["diabetes_rate", "support"], ascending=False).reset_index(drop=True)
