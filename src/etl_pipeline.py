import sys
import os
import sqlite3
import pandas as pd
from pathlib import Path

# Thêm thư mục gốc vào đường dẫn hệ thống để Python hiểu được module 'src'
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from scripts.data_loading import load_and_prepare_brfss

def run_etl_pipeline(db_path="dw/data_warehouse.db"):
    """
    Thực hiện ETL cho bộ dữ liệu 22 cột thực tế của nhóm:
    1. Lấy dữ liệu từ Staging (file CSV).
    2. Transform thành Star Schema chuẩn (5 Dim, 1 Fact).
    3. Load vào SQLite Database.
    """
    print("[1/3] Extracting data from Staging Area...")
    df = load_and_prepare_brfss()
    
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
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
    
    # Xóa file db cũ nếu có để khởi tạo lại từ đầu
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    
    # Đọc và thực thi file star_schema.sql để TẠO CẤU TRÚC BẢNG có sẵn Primary Key và Foreign Key
    schema_path = os.path.join(project_root, 'dw', 'schema', 'star_schema.sql')
    with open(schema_path, 'r', encoding='utf-8') as f:
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
    run_etl_pipeline(db_path="dw/data_warehouse.db")
