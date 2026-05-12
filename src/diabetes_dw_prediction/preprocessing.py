from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureGroups:
    """Define which columns are dimensions vs measures vs target."""

    target: str
    dims: list[str]
    measures: list[str]


def default_feature_groups(target: str = "Diabetes_binary") -> FeatureGroups:
    """Default grouping for the BRFSS diabetes indicators dataset.

    You can adjust this list depending on what you want to show in your seminar/cube.
    """

    dims = [
        "Age",
        "Sex",
        "Income",
        "Education",
        "GenHlth",
        "HighBP",
        "HighChol",
        "Smoker",
        "PhysActivity",
    ]
    measures = ["BMI", "PhysHlth", "MentHlth"]
    return FeatureGroups(target=target, dims=dims, measures=measures)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning (safe default)."""

    return df.copy()


def coerce_dimension_measure_dtypes(
    df: pd.DataFrame,
    dims: list[str],
    measures: list[str],
    target: str | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Coerce dtypes for dimensions (Int64) and measures (float64).

    - Dimensions in this dataset are mostly binary/ordinal.
    - Measures (BMI/PhysHlth/MentHlth) are numeric.
    """

    out = df.copy() if copy else df

    for col in dims:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    for col in measures:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    if target is not None and target in out.columns:
        out[target] = pd.to_numeric(out[target], errors="coerce").astype("Int64")

    return out


def quantile_bounds(series: pd.Series, low_q: float = 0.01, high_q: float = 0.99) -> tuple[float, float]:
    """Compute (low, high) quantile bounds for winsorization/capping."""

    s = series.dropna()
    if s.empty:
        return (float("nan"), float("nan"))
    return (float(s.quantile(low_q)), float(s.quantile(high_q)))


def compute_bounds_for_columns(
    df: pd.DataFrame,
    cols: list[str],
    low_q: float = 0.01,
    high_q: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """Compute quantile bounds for a set of numeric columns."""

    bounds: dict[str, tuple[float, float]] = {}
    for col in cols:
        if col in df.columns:
            bounds[col] = quantile_bounds(df[col], low_q=low_q, high_q=high_q)
    return bounds


def winsorize_clip(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    copy: bool = True,
) -> pd.DataFrame:
    """Clip values to provided bounds (winsorization via clipping)."""

    out = df.copy() if copy else df
    for col, (lo, hi) in bounds.items():
        if col in out.columns and not (np.isnan(lo) or np.isnan(hi)):
            out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Wrapper around sklearn.train_test_split with stratify=y."""

    from sklearn.model_selection import train_test_split

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor(
    feature_cols: list[str],
    numeric_cols: list[str],
    scaler: Literal["standard", "none"] = "standard",
):
    """Build ColumnTransformer: scale numeric columns, passthrough others."""

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    numeric_cols = [c for c in numeric_cols if c in feature_cols]
    passthrough_cols = [c for c in feature_cols if c not in numeric_cols]

    if scaler == "standard" and numeric_cols:
        num_transformer = StandardScaler()
    else:
        num_transformer = "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_cols),
            ("cat", "passthrough", passthrough_cols),
        ],
        remainder="drop",
    )


def prepare_for_classification(
    df: pd.DataFrame,
    feature_groups: FeatureGroups | None = None,
    apply_winsorize: bool = True,
    winsor_low_q: float = 0.01,
    winsor_high_q: float = 0.99,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """End-to-end preprocessing:

    - dtype coercion
    - (optional) winsorize measures
    - split train/test stratified
    - scale measures (StandardScaler)

    Returns:
        df_pp,
        (X_train, X_test, y_train, y_test),
        preprocessor,
        (X_train_ready, X_test_ready)
    """

    fg = feature_groups or default_feature_groups()
    df_pp = coerce_dimension_measure_dtypes(df, dims=fg.dims, measures=fg.measures, target=fg.target)

    if apply_winsorize:
        bounds = compute_bounds_for_columns(df_pp, cols=fg.measures, low_q=winsor_low_q, high_q=winsor_high_q)
        df_pp = winsorize_clip(df_pp, bounds=bounds)

    feature_cols = [c for c in (fg.dims + fg.measures) if c in df_pp.columns]
    X = df_pp[feature_cols].copy()
    y = df_pp[fg.target].astype("int64")

    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(feature_cols=feature_cols, numeric_cols=fg.measures, scaler="standard")
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    return df_pp, (X_train, X_test, y_train, y_test), preprocessor, (X_train_ready, X_test_ready)
