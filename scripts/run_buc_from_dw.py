from __future__ import annotations

import argparse
import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Thêm thư mục gốc vào đường dẫn hệ thống để Python hiểu được module 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from scripts.cube_buc import iceberg_cube_buc


DEFAULT_DIMS = ["Age", "Sex", "Smoker", "HighBP"]

def prepare_data_for_buc(db_path: Path | None = None) -> pd.DataFrame:
    """
    Kết nối tới Data Warehouse (SQLite), truy vấn toàn bộ Data từ Star Schema 
    và chuẩn bị DataFrame để đưa vào thuật toán BUC.
    """
    if db_path is None:
        db_path = project_root / "dw" / "data_warehouse.db"
        
    conn = sqlite3.connect(db_path)
    
    # Truy vấn SQL để JOIN TẤT CẢ các bảng lại (Mô phỏng truy vấn OLAP)
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
            f.Diabetes_binary as Target
        FROM Fact_PatientHealth f
        JOIN Dim_Demographic d ON f.DemographicKey = d.DemographicKey
        JOIN Dim_Lifestyle l ON f.LifestyleKey = l.LifestyleKey
        JOIN Dim_HealthStatus h ON f.HealthStatusKey = h.HealthStatusKey
        JOIN Dim_MedicalHistory m ON f.MedicalHistoryKey = m.MedicalHistoryKey
        JOIN Dim_HealthcareAccess ha ON f.HealthcareAccessKey = ha.HealthcareAccessKey
    """
    
    print("[*] Đang đọc dữ liệu từ Data Warehouse...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"[*] Đã tải thành công {df.shape[0]} dòng dữ liệu với {df.shape[1]} cột!")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Iceberg Cube (BUC) from the SQLite DW")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to SQLite DW file (default: dw/data_warehouse.db)",
    )
    parser.add_argument(
        "--dims",
        type=str,
        default=",".join(DEFAULT_DIMS),
        help="Comma-separated dimensions to build the cube (e.g. Age,Sex,Smoker,HighBP)",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=1000,
        help="Minimum support threshold for the iceberg cube.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top rows to print.",
    )
    parser.add_argument(
        "--save-to-dw",
        action="store_true",
        help="Save cube result to SQLite table IcebergCube_BUC.",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="IcebergCube_BUC",
        help="Table name when --save-to-dw is enabled.",
    )
    return parser.parse_args()


def save_cube_to_dw(db_path: Path, table: str, cube_df: pd.DataFrame) -> None:
    with sqlite3.connect(db_path) as conn:
        cube_df.to_sql(table, conn, if_exists="replace", index=False)


def main() -> None:
    args = parse_args()

    db_path = args.db_path or (project_root / "dw" / "data_warehouse.db")
    dims = [d.strip() for d in args.dims.split(",") if d.strip()]
    if not dims:
        raise SystemExit("No dims provided. Use --dims Age,Sex,...")

    df_buc = prepare_data_for_buc(db_path=db_path)

    print(f"\n[*] Bắt đầu chạy thuật toán BUC với các dimensions: {dims}")
    cube_result = iceberg_cube_buc(
        df=df_buc,
        dims=dims,
        measure_col="Target",
        min_support=args.min_support,
    )

    print(f"\n[*] Kết quả Iceberg Cube (Top {args.top}):")
    if cube_result.empty:
        print("(empty) Không có cuboid nào thỏa min_support")
    else:
        print(cube_result.head(args.top).to_string(index=False))

    if args.save_to_dw:
        save_cube_to_dw(db_path=db_path, table=args.table, cube_df=cube_result)
        print(f"\n[*] Đã lưu kết quả vào DW: table={args.table}")

if __name__ == "__main__":
    main()
