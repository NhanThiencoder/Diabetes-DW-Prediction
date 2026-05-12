from __future__ import annotations

import pandas as pd


def iceberg_cube_buc(
    df: pd.DataFrame,
    dims: list[str],
    measure_col: str,
    min_support: int,
) -> pd.DataFrame:
    """Compute an iceberg cube (placeholder).

    Later: implement top-down BUC with pruning.
    For now: a safe, simple baseline using groupby over all dims.
    """
    if not dims:
        out = df.agg(support=(measure_col, "size"), diabetes_rate=(measure_col, "mean")).to_frame().T
        return out[out["support"] >= min_support].reset_index(drop=True)

    g = (
        df.groupby(dims, dropna=False)[measure_col]
        .agg(support="size", diabetes_rate="mean")
        .reset_index()
    )
    return g[g["support"] >= min_support].sort_values(["diabetes_rate", "support"], ascending=False)
