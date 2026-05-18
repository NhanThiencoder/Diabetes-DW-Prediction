from __future__ import annotations

from pathlib import Path

import pandas as pd


__all__ = [
    "load_brfss_diabetes_csv",
    "load_and_prepare_brfss",
]

def load_brfss_diabetes_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load the BRFSS diabetes indicators dataset from a CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def _infer_project_root() -> Path:
    # scripts/data_loading.py -> <project_root>/scripts/data_loading.py
    return Path(__file__).resolve().parents[1]


def load_and_prepare_brfss(
    csv_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    """
    Tải dữ liệu BRFSS từ Google Drive nếu chưa có, giải nén, đọc CSV vào DataFrame.
    Trả về: DataFrame
    """
    default_filename = "diabetes_binary_health_indicators_BRFSS2015.csv"

    if csv_path is not None:
        resolved_csv = Path(csv_path)
    else:
        root = Path(project_root) if project_root is not None else _infer_project_root()
        candidates = [
            root / "data" / "raw" / default_filename,
            root / "dataset" / default_filename,
            # notebook copy (if user runs directly inside notebooks/)
            root / "notebooks" / "dataset" / default_filename,
        ]
        resolved_csv = next((p for p in candidates if p.exists()), candidates[0])

    print(f"[*] Đang đọc file vào Pandas từ: {resolved_csv}")
    df = load_brfss_diabetes_csv(resolved_csv)
    print(f"-> Thành công! Kích thước dữ liệu: {df.shape[0]:,} dòng.")
    return df
