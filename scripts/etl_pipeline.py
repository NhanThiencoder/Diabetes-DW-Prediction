import sys
import os
import sqlite3
import pandas as pd
from pathlib import Path

# Thêm thư mục gốc vào đường dẫn hệ thống để Python hiểu được module 'src'
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from src.diabetes_dw_prediction.data_loading import load_and_prepare_brfss

def run_etl_pipeline(db_path="dw/data_warehouse.db"):
    """
    Thực hiện ETL:
    1. Lấy dữ liệu từ Staging (file CSV).
    2. Transform thành Star Schema (Dim_Demographic, Dim_Lifestyle, Dim_HealthStatus, Fact_PatientHealth).
    3. Load vào SQLite Database.
    """
    print("[1/3] Extracting data from Staging Area...")
    # Lấy dữ liệu (hàm này có sẵn trong project của bạn)
    df = load_and_prepare_brfss()
    
    # Tạo ID giả lập cho mỗi dòng (PatientID) vì dataset gốc không có
    df = df.reset_index(names='PatientID')
    
    print("[2/3] Transforming data to Star Schema...")
    # ---------------------------------------------------------
    # 1. Dim_Demographic
    # ---------------------------------------------------------
    # Các cột: DemographicKey, AgeGroup, Sex, Education, Income
    # Lưu ý: Trong file gốc các cột tên là Age, Sex, Education, Income
    dim_demographic = df[['Age', 'Sex', 'Education', 'Income']].drop_duplicates().reset_index(drop=True)
    dim_demographic = dim_demographic.reset_index(names='DemographicKey')
    # Đổi tên cột cho đúng chuẩn
    dim_demographic.rename(columns={'Age': 'AgeGroup'}, inplace=True)

    # ---------------------------------------------------------
    # 2. Dim_Lifestyle
    # ---------------------------------------------------------
    # Các cột: LifestyleKey, Smoker, PhysActivity
    dim_lifestyle = df[['Smoker', 'PhysActivity']].drop_duplicates().reset_index(drop=True)
    dim_lifestyle = dim_lifestyle.reset_index(names='LifestyleKey')

    # ---------------------------------------------------------
    # 3. Dim_HealthStatus
    # ---------------------------------------------------------
    # Các cột: HealthStatusKey, GenHlth, HighBP, HighChol
    dim_healthstatus = df[['GenHlth', 'HighBP', 'HighChol']].drop_duplicates().reset_index(drop=True)
    dim_healthstatus = dim_healthstatus.reset_index(names='HealthStatusKey')

    # ---------------------------------------------------------
    # 4. Fact_PatientHealth
    # ---------------------------------------------------------
    # Merge để lấy Key từ các Dimension bảng
    fact_df = df[['PatientID', 'Age', 'Sex', 'Education', 'Income', 
                  'Smoker', 'PhysActivity', 
                  'GenHlth', 'HighBP', 'HighChol', 
                  'BMI', 'Diabetes_binary']].copy()

    # Map DemographicKey
    fact_df = fact_df.merge(dim_demographic, left_on=['Age', 'Sex', 'Education', 'Income'], right_on=['AgeGroup', 'Sex', 'Education', 'Income'], how='left')
    
    # Map LifestyleKey
    fact_df = fact_df.merge(dim_lifestyle, on=['Smoker', 'PhysActivity'], how='left')
    
    # Map HealthStatusKey
    fact_df = fact_df.merge(dim_healthstatus, on=['GenHlth', 'HighBP', 'HighChol'], how='left')

    # Chỉ giữ lại các cột đúng như yêu cầu
    fact_patienthealth = fact_df[['PatientID', 'DemographicKey', 'LifestyleKey', 'HealthStatusKey', 'BMI', 'Diabetes_binary']]

    print("[3/3] Loading data into SQLite (Data Warehouse)...")
    # Đảm bảo thư mục lưu DB tồn tại
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
    
    # Kết nối SQLite và ghi dữ liệu
    conn = sqlite3.connect(db_path)
    
    # Ghi từng bảng, sử dụng index=False để không ghi index của DataFrame
    dim_demographic.to_sql('Dim_Demographic', conn, if_exists='replace', index=False)
    dim_lifestyle.to_sql('Dim_Lifestyle', conn, if_exists='replace', index=False)
    dim_healthstatus.to_sql('Dim_HealthStatus', conn, if_exists='replace', index=False)
    fact_patienthealth.to_sql('Fact_PatientHealth', conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"[*] ETL hoàn tất! Database đã được lưu tại: {db_path}")

if __name__ == "__main__":
    run_etl_pipeline(db_path="dw/data_warehouse.db")
