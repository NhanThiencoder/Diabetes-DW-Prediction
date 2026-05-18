import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Thêm thư mục gốc vào đường dẫn hệ thống để Python hiểu được module 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from scripts.cube_buc import iceberg_cube_buc

def prepare_data_for_buc(db_path=None):
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
        l.Smoker,
        l.PhysActivity,
        l.Fruits,
        l.Veggies,
        l.HvyAlcoholConsump,
        m.HighBP,
        m.HighChol,
        m.Stroke,
        m.HeartDiseaseorAttack,
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

if __name__ == "__main__":
    df_buc = prepare_data_for_buc()
    
    # Bạn có thể chọn bất kỳ Dimensions nào từ 22 cột ban đầu để phân tích
    dimensions = ["Age", "Sex", "Smoker", "HighBP"]
    
    print(f"\n[*] Bắt đầu chạy thuật toán BUC với các dimensions: {dimensions}")
    cube_result = iceberg_cube_buc(
        df=df_buc,
        dims=dimensions,
        measure_col="Target",
        min_support=1000  # Ngưỡng min_support
    )
    
    print("\n[*] Kết quả Iceberg Cube (Top 10):")
    # Sử dụng to_string thay vì to_markdown để tránh lỗi thiếu tabulate
    print(cube_result.head(10).to_string())
