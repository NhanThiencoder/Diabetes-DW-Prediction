from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import get_paths


LR_FEATURES = [
    "Age",
    "Sex",
    "Education",
    "Income",
    "GenHlth",
    "HighBP",
    "HighChol",
    "Smoker",
    "PhysActivity",
    "BMI",
    "PhysHlth",
    "MentHlth",
]

RF_FEATURES = [
    "Age",
    "Sex",
    "Education",
    "Income",
    "Smoker",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "HighBP",
    "HighChol",
    "CholCheck",
    "Stroke",
    "HeartDiseaseorAttack",
    "AnyHealthcare",
    "NoDocbcCost",
    "BMI",
]


def load_from_dw(db_path: Path) -> pd.DataFrame:
    """Load a flattened dataset from the DW star schema via joins."""

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
        df = pd.read_sql_query(query, conn)

    return df


def split_and_save(
    df: pd.DataFrame,
    target: str,
    out_dir: Path,
    test_size: float,
    random_state: int,
) -> tuple[Path, Path]:
    """Stratified train/test split and save to CSV (and Parquet when available)."""

    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    X = df.drop(columns=[target])
    y = df[target].astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[target] = y_train.values
    test_df = X_test.copy()
    test_df[target] = y_test.values

    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = out_dir / "train.csv"
    test_csv = out_dir / "test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    try:
        train_df.to_parquet(out_dir / "train.parquet", index=False)
        test_df.to_parquet(out_dir / "test.parquet", index=False)
    except Exception:
        pass

    return train_csv, test_csv


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DW extract: {missing}")
    return columns


def save_model_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    out_dir: Path,
    model_name: str,
    feature_cols: list[str],
) -> tuple[Path, Path]:
    """Save model-specific train/test CSVs with a selected feature set."""

    feature_cols = _ensure_columns(train_df, feature_cols)
    cols = feature_cols + [target]

    model_train = out_dir / f"train_{model_name}.csv"
    model_test = out_dir / f"test_{model_name}.csv"

    train_df[cols].to_csv(model_train, index=False)
    test_df[cols].to_csv(model_test, index=False)

    return model_train, model_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split DW data into train/test.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PROJECT_ROOT / "dw" / "data_warehouse.db",
        help="Path to SQLite DW file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Diabetes_binary",
        help="Target column name.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder (default: data/processed/splits).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = get_paths(PROJECT_ROOT)
    out_dir = args.out_dir or (paths.data_processed_dir / "splits")

    if not args.db_path.exists() or args.db_path.stat().st_size == 0:
        raise FileNotFoundError(f"DW not found or empty: {args.db_path}")

    df = load_from_dw(args.db_path)
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns.")

    train_csv, test_csv = split_and_save(
        df,
        target=args.target,
        out_dir=out_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    lr_train, lr_test = save_model_splits(
        train_df,
        test_df,
        target=args.target,
        out_dir=out_dir,
        model_name="lr",
        feature_cols=LR_FEATURES,
    )
    rf_train, rf_test = save_model_splits(
        train_df,
        test_df,
        target=args.target,
        out_dir=out_dir,
        model_name="rf",
        feature_cols=RF_FEATURES,
    )

    print("Saved:", train_csv)
    print("Saved:", test_csv)
    print("Saved:", lr_train)
    print("Saved:", lr_test)
    print("Saved:", rf_train)
    print("Saved:", rf_test)


if __name__ == "__main__":
    main()
