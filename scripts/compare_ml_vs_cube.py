from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DIMS = ["Age", "HighBP", "HighChol", "Smoker"]
TARGET_COL = "Diabetes_binary"
NUMERIC_COLS = ["BMI", "PhysHlth", "MentHlth"]


@dataclass(frozen=True)
class Paths:
    dw_path: Path
    splits_dir: Path
    models_dir: Path
    results_dir: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare Iceberg Cube group insights vs ML predictions on test set."
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=PROJECT_ROOT / "dw" / "data_warehouse.db",
        help="Path to SQLite DW file.",
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "splits",
        help="Folder containing train/test split CSVs.",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Folder containing trained models/artifacts.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Folder to save report-ready CSV outputs.",
    )
    p.add_argument(
        "--dims",
        type=str,
        default=",".join(DEFAULT_DIMS),
        help="Comma-separated dimensions for cube and grouping (default: Age,HighBP,HighChol,Smoker).",
    )
    p.add_argument(
        "--min-support",
        type=int,
        default=2000,
        help="Iceberg min_support threshold.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many top groups to export.",
    )
    p.add_argument(
        "--model",
        choices=["rf", "lr"],
        default="rf",
        help="Which model to use for prediction comparison (rf=main, lr=baseline).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for predicted label rate.",
    )
    return p.parse_args()


def load_flattened_from_dw(db_path: Path) -> pd.DataFrame:
    """Flatten star schema into a single dataframe via joins."""

    query = """
        SELECT
            d.Age,
            d.Sex,
            d.Education,
            d.Income,
            l.Smoker,
            l.PhysActivity,
            l.Fruits,
            l.Veggies,
            l.HvyAlcoholConsump,
            h.GenHlth,
            h.MentHlth,
            h.PhysHlth,
            h.DiffWalk,
            m.HighBP,
            m.HighChol,
            m.CholCheck,
            m.Stroke,
            m.HeartDiseaseorAttack,
            ha.AnyHealthcare,
            ha.NoDocbcCost,
            f.BMI,
            f.Diabetes_binary
        FROM Fact_PatientHealth f
        JOIN Dim_Demographic d ON f.DemographicKey = d.DemographicKey
        JOIN Dim_Lifestyle l ON f.LifestyleKey = l.LifestyleKey
        JOIN Dim_HealthStatus h ON f.HealthStatusKey = h.HealthStatusKey
        JOIN Dim_MedicalHistory m ON f.MedicalHistoryKey = m.MedicalHistoryKey
        JOIN Dim_HealthcareAccess ha ON f.HealthcareAccessKey = ha.HealthcareAccessKey
    """

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def compute_iceberg_groups(
    df: pd.DataFrame,
    dims: list[str],
    target: str,
    min_support: int,
) -> pd.DataFrame:
    """Compute iceberg groups as (dims -> support, diabetes_rate)."""

    for c in [*dims, target]:
        if c not in df.columns:
            raise ValueError(f"Missing column in DW extract: {c}")

    out = (
        df.groupby(dims, dropna=False)[target]
        .agg(support="size", diabetes_rate="mean")
        .reset_index()
    )
    out = out[out["support"] >= int(min_support)].copy()
    out = out.sort_values(["diabetes_rate", "support"], ascending=[False, False])
    return out


def _apply_winsor_bounds(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in out.columns and not (np.isnan(lo) or np.isnan(hi)):
            out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def predict_with_rf(paths: Paths, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model_path = paths.models_dir / "random_forest.joblib"
    arts_path = paths.models_dir / "rf_preproc.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not arts_path.exists():
        raise FileNotFoundError(f"Preproc artifacts not found: {arts_path}")

    model = load(model_path)
    arts = load(arts_path)
    imputer = arts["imputer"]
    bounds = arts["winsor_bounds"]

    X = test_df.drop(columns=[TARGET_COL])
    y = test_df[TARGET_COL].astype("int64").to_numpy()

    # impute numeric cols
    numeric = [c for c in NUMERIC_COLS if c in X.columns]
    if numeric:
        X_num = pd.DataFrame(imputer.transform(X[numeric]), columns=numeric, index=X.index)
        X = X.copy()
        X[numeric] = X_num

    # winsorize using train-fitted bounds
    X = _apply_winsor_bounds(X, bounds)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        # fallback: decision function (not expected for RF)
        scores = model.decision_function(X)
        proba = 1.0 / (1.0 + np.exp(-scores))

    pred = model.predict(X)
    return proba, pred.astype(int)


def predict_with_lr(paths: Paths, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model_path = paths.models_dir / "logistic_regression.joblib"
    scaler_path = paths.models_dir / "lr_scaler.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = load(model_path)
    scaler = load(scaler_path)

    X = test_df.drop(columns=[TARGET_COL]).copy()
    y = test_df[TARGET_COL].astype("int64").to_numpy()

    numeric = [c for c in NUMERIC_COLS if c in X.columns]
    if numeric:
        X[numeric] = scaler.transform(X[numeric])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
        proba = 1.0 / (1.0 + np.exp(-scores))

    pred = (proba >= 0.5).astype(int)
    return proba, pred


def group_prediction_stats(
    test_df: pd.DataFrame,
    dims: list[str],
    proba: np.ndarray,
    pred: np.ndarray,
) -> pd.DataFrame:
    df = test_df.copy()
    for d in dims:
        if d not in df.columns:
            raise ValueError(f"Dim not found in test split: {d}")

    df["pred_proba"] = proba
    df["pred_label"] = pred

    g = (
        df.groupby(dims, dropna=False)
        .agg(
            n=(TARGET_COL, "size"),
            actual_rate=(TARGET_COL, "mean"),
            pred_proba_mean=("pred_proba", "mean"),
            pred_label_rate=("pred_label", "mean"),
        )
        .reset_index()
    )

    return g.sort_values(["pred_proba_mean", "n"], ascending=[False, False])


def main() -> None:
    args = parse_args()
    dims = [d.strip() for d in args.dims.split(",") if d.strip()]

    paths = Paths(
        dw_path=args.db_path,
        splits_dir=args.splits_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
    )

    if not paths.dw_path.exists() or paths.dw_path.stat().st_size == 0:
        raise FileNotFoundError(f"DW not found or empty: {paths.dw_path}")

    # Load DW extract
    dw_df = load_flattened_from_dw(paths.dw_path)

    # Compute iceberg groups
    cube = compute_iceberg_groups(dw_df, dims=dims, target=TARGET_COL, min_support=args.min_support)
    top_cube = cube.head(int(args.top_n)).copy()

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    cube_out = paths.results_dir / f"cube_top_groups_{args.model}.csv"
    top_cube.to_csv(cube_out, index=False)

    # Load test split for chosen model
    test_path = paths.splits_dir / f"test_{args.model}.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing test split: {test_path}. Run: python scripts/split_from_dw.py"
        )

    test_df = pd.read_csv(test_path)

    # Predict
    if args.model == "rf":
        proba, pred = predict_with_rf(paths, test_df)
    else:
        proba, pred = predict_with_lr(paths, test_df)

    # Group prediction stats
    stats = group_prediction_stats(test_df, dims=dims, proba=proba, pred=(proba >= args.threshold).astype(int))

    # Join top cube groups with prediction stats (left join keeps cube ordering)
    compare = top_cube.merge(stats, on=dims, how="left")

    compare_out = paths.results_dir / f"compare_cube_vs_{args.model}.csv"
    compare.to_csv(compare_out, index=False)

    print("\n[OK] Saved cube top groups:")
    print(cube_out)
    print("\n[OK] Saved comparison table:")
    print(compare_out)

    print("\nPreview (comparison):")
    with pd.option_context("display.max_columns", 50, "display.width", 140):
        print(compare.head(min(10, len(compare))).to_string(index=False))


if __name__ == "__main__":
    main()
