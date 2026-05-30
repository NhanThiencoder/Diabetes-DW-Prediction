import argparse
import sys
import os
import sqlite3
import pandas as pd
from pathlib import Path

# Thêm thư mục gốc vào đường dẫn hệ thống để Python hiểu được module 'scripts'
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from scripts.data_loading import load_and_prepare_brfss
from scripts import preprocessing as pp


DW_DIM_COLS = [
    # demographic
    "Age",
    "Sex",
    "Education",
    "Income",
    # lifestyle
    "Smoker",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    # health status
    "GenHlth",
    "DiffWalk",
    # medical history
    "HighBP",
    "HighChol",
    "CholCheck",
    "Stroke",
    "HeartDiseaseorAttack",
    # healthcare access
    "AnyHealthcare",
    "NoDocbcCost",
]

DW_MEASURE_COLS = [
    "BMI",
    "PhysHlth",
    "MentHlth",
]

DW_TARGET_COL = "Diabetes_binary"


def preprocess_before_dw(
    df: pd.DataFrame,
    apply_winsorize: bool = True,
    winsor_low_q: float = 0.01,
    winsor_high_q: float = 0.99,
    drop_missing_required: bool = True,
) -> pd.DataFrame:
    """Preprocess data before building/loading the DW.

    Purpose: keep DW data quality consistent for cube/BI while avoiding ML-only
    transformations (e.g., scaling) that would be trained on a split.

    Steps:
    - Coerce dtypes: dimensions -> Int64, measures -> float64, target -> Int64
    - Optional winsorize (clip) numeric measures
    - Optional drop rows that miss required fields
    """

    required_cols = [*DW_DIM_COLS, *DW_MEASURE_COLS, DW_TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for DW: {missing}")

    df_pp = pp.coerce_dimension_measure_dtypes(
        df,
        dims=DW_DIM_COLS,
        measures=DW_MEASURE_COLS,
        target=DW_TARGET_COL,
        copy=True,
    )

    if apply_winsorize:
        bounds = pp.compute_bounds_for_columns(
            df_pp,
            cols=DW_MEASURE_COLS,
            low_q=winsor_low_q,
            high_q=winsor_high_q,
        )
        df_pp = pp.winsorize_clip(df_pp, bounds=bounds, copy=False)

    if drop_missing_required:
        before = len(df_pp)
        df_pp = df_pp.dropna(subset=required_cols)
        dropped = before - len(df_pp)
        if dropped:
            print(f"[*] Dropped {dropped:,} rows with missing required values.")

    return df_pp

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETL: preprocess -> star schema -> load into SQLite DW")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to SQLite DW file (default: dw/data_warehouse.db)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable preprocessing before building the DW.",
    )
    parser.add_argument(
        "--no-winsorize",
        action="store_true",
        help="Disable winsorize/capping for numeric measures before DW load.",
    )
    parser.add_argument(
        "--winsor-low-q",
        type=float,
        default=0.01,
        help="Low quantile for winsorize clipping.",
    )
    parser.add_argument(
        "--winsor-high-q",
        type=float,
        default=0.99,
        help="High quantile for winsorize clipping.",
    )
    return parser.parse_args()


def run_etl_pipeline(
    db_path=None,
    preprocess: bool = True,
    apply_winsorize: bool = True,
    winsor_low_q: float = 0.01,
    winsor_high_q: float = 0.99,
):
    """
    Thực hiện ETL cho bộ dữ liệu 22 cột thực tế của nhóm:
    1. Lấy dữ liệu từ Staging (file CSV).
    2. Transform thành Star Schema chuẩn (5 Dim, 1 Fact).
    3. Load vào SQLite Database.
    """
    print("[1/3] Extracting data from Staging Area...")
    df = load_and_prepare_brfss()

    if preprocess:
        print("[1.5/3] Preprocessing before building Data Warehouse...")
        df = preprocess_before_dw(
            df,
            apply_winsorize=apply_winsorize,
            winsor_low_q=winsor_low_q,
            winsor_high_q=winsor_high_q,
            drop_missing_required=True,
        )
    
    # Tạo PatientID giả lập vì dataset gốc không có
    df = df.reset_index(names='PatientID')
    
    print("[2/3] Transforming data to Star Schema (Full 22 Columns)...")
    
    # 1. Dim_Demographic
    dim_demographic = df[['Age', 'Sex', 'Education', 'Income']].drop_duplicates().reset_index(drop=True)
    dim_demographic = dim_demographic.reset_index(names='DemographicKey')

    # 2. Dim_Lifestyle
    dim_lifestyle = df[['Smoker', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump']].drop_duplicates().reset_index(drop=True)
    dim_lifestyle = dim_lifestyle.reset_index(names='LifestyleKey')

    # 3. Dim_HealthStatus
    dim_healthstatus = df[['GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk']].drop_duplicates().reset_index(drop=True)
    dim_healthstatus = dim_healthstatus.reset_index(names='HealthStatusKey')

    # 4. Dim_MedicalHistory
    dim_medical = df[['HighBP', 'HighChol', 'CholCheck', 'Stroke', 'HeartDiseaseorAttack']].drop_duplicates().reset_index(drop=True)
    dim_medical = dim_medical.reset_index(names='MedicalHistoryKey')

    # 5. Dim_HealthcareAccess
    dim_healthcare = df[['AnyHealthcare', 'NoDocbcCost']].drop_duplicates().reset_index(drop=True)
    dim_healthcare = dim_healthcare.reset_index(names='HealthcareAccessKey')

    # 6. Fact_PatientHealth
    fact_df = df.copy()

    # Map các khóa (Keys)
    fact_df = fact_df.merge(dim_demographic, on=['Age', 'Sex', 'Education', 'Income'], how='left')
    fact_df = fact_df.merge(dim_lifestyle, on=['Smoker', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump'], how='left')
    fact_df = fact_df.merge(dim_healthstatus, on=['GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk'], how='left')
    fact_df = fact_df.merge(dim_medical, on=['HighBP', 'HighChol', 'CholCheck', 'Stroke', 'HeartDiseaseorAttack'], how='left')
    fact_df = fact_df.merge(dim_healthcare, on=['AnyHealthcare', 'NoDocbcCost'], how='left')

    # Giữ lại đúng các cột của Fact Table
    fact_patienthealth = fact_df[['PatientID', 'DemographicKey', 'LifestyleKey', 'HealthStatusKey', 
                                  'MedicalHistoryKey', 'HealthcareAccessKey', 'BMI', 'Diabetes_binary']]

    print("[3/3] Loading data into SQLite (Data Warehouse)...")
    if db_path is None:
        db_path = project_root / "dw" / "data_warehouse.db"
    db_path = Path(db_path)

    os.makedirs(db_path.parent, exist_ok=True)
    
    # Xóa file db cũ nếu có để khởi tạo lại từ đầu
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    
    # Đọc và thực thi file star_schema.sql để TẠO CẤU TRÚC BẢNG có sẵn Primary Key và Foreign Key
    schema_path = project_root / "dw" / "schema" / "star_schema.sql"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    conn.commit()
    
    # Ghi dữ liệu vào các bảng ĐÃ TẠO (sử dụng append để giữ nguyên Primary Key)
    dim_demographic.to_sql('Dim_Demographic', conn, if_exists='append', index=False)
    dim_lifestyle.to_sql('Dim_Lifestyle', conn, if_exists='append', index=False)
    dim_healthstatus.to_sql('Dim_HealthStatus', conn, if_exists='append', index=False)
    dim_medical.to_sql('Dim_MedicalHistory', conn, if_exists='append', index=False)
    dim_healthcare.to_sql('Dim_HealthcareAccess', conn, if_exists='append', index=False)
    fact_patienthealth.to_sql('Fact_PatientHealth', conn, if_exists='append', index=False)
    
    conn.close()
    print(f"[*] ETL hoàn tất! Database đã bao gồm toàn bộ Data của nhóm tại: {db_path}")

if __name__ == "__main__":
    args = parse_args()
    run_etl_pipeline(
        db_path=args.db_path,
        preprocess=(not args.no_preprocess),
        apply_winsorize=(not args.no_winsorize),
        winsor_low_q=args.winsor_low_q,
        winsor_high_q=args.winsor_high_q,
    )
