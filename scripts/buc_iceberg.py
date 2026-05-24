import os
import sqlite3
import pandas as pd

# ==========================================
# BƯỚC 1: CHUẨN BỊ DỮ LIỆU TỪ SQLITE
# ==========================================
print("1. Đang kết nối tới Data Warehouse...")

script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.normpath(os.path.join(script_dir, '..', 'dw', 'data_warehouse.db'))
if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
    print(f"Lỗi: Không tìm thấy hoặc file database trống: {db_path}")
    raise SystemExit(1)

# Kết nối với CSDL SQLite 
conn = sqlite3.connect(db_path)


# JOIN các bảng trong Data Warehouse
query = """
    SELECT 
        d.Age, 
        m.HighBP, 
        m.HighChol, 
        l.Smoker 
    FROM Fact_PatientHealth f
    JOIN Dim_Demographic d ON f.DemographicKey = d.DemographicKey
    JOIN Dim_MedicalHistory m ON f.MedicalHistoryKey = m.MedicalHistoryKey
    JOIN Dim_Lifestyle l ON f.LifestyleKey = l.LifestyleKey
    WHERE f.Diabetes_binary = 1
"""
try:
    df = pd.read_sql_query(query, conn)
    print(f"-> Đã tải {len(df)} bản ghi.")
    if df.empty:
        print("Lỗi: truy vấn trả về 0 bản ghi. Kiểm tra dữ liệu trong bảng và điều kiện WHERE.")
        conn.close()
        raise SystemExit(1)
except Exception as e:
    print("Lỗi đọc dữ liệu, vui lòng kiểm tra lại tên bảng/cột:", e)
    conn.close()
    raise SystemExit(1)

# Danh sách các chiều (Dimensions) cần đưa vào khối Cube
dimensions = ['Age', 'HighBP', 'HighChol', 'Smoker']
original_dims = dimensions.copy() # Lưu lại để xuất báo cáo
results = []

# ==========================================
# BƯỚC 2: CÀI ĐẶT LÕI THUẬT TOÁN BUC (ĐỆ QUY)
# ==========================================
def buc_algorithm(current_df, current_dims, current_combo, min_sup):
    """
    Hàm đệ quy BUC tính toán Iceberg Cube
    :param current_df: DataFrame hiện tại trong nhánh đệ quy
    :param current_dims: Các chiều còn lại cần xét
    :param current_combo: Tổ hợp các giá trị hiện tại (ví dụ: Age=50, HighBP=1)
    :param min_sup: Ngưỡng Support tối thiểu (Minimum Support)
    """
    # Tính measure 
    support = len(current_df)
    
    # CƠ CHẾ CẮT TỈA (PRUNING) 
    if support < min_sup:
        return 

    record = {dim: current_combo.get(dim, 'ALL') for dim in original_dims}
    record['Patient_Count'] = support
    results.append(record)

    # Tiếp tục partition và gọi đệ quy cho các chiều còn lại
    for i in range(len(current_dims)):
        dim = current_dims[i]

        for val, group_df in current_df.groupby(dim):
            new_combo = current_combo.copy()
            new_combo[dim] = val
            
            buc_algorithm(group_df, current_dims[i+1:], new_combo, min_sup)

# CHẠY THUẬT TOÁN
MIN_SUP = 2000 
print(f"2. Bắt đầu tính toán Iceberg Cube với min_sup = {MIN_SUP}...")
buc_algorithm(df, dimensions, {}, MIN_SUP)
print(f"-> Đã tạo xong {len(results)} cuboids thỏa mãn điều kiện.")

# ==========================================
# BƯỚC 3: LƯU KẾT QUẢ VÀ TRUY VẤN
# ==========================================
cube_df = pd.DataFrame(results)

# 3.1. LƯU VÀO SQLITE
print("3. Lưu kết quả Iceberg Cube vào Data Warehouse...")
cube_df.to_sql('IcebergCube_BUC', conn, if_exists='replace', index=False)
print("-> Đã lưu thành công vào bảng 'IcebergCube_BUC'.")

# 3.2. TRUY VẤN BÁO CÁO MẪU
print("\n--- BÁO CÁO: CÁC TỔ HỢP NGUY CƠ CAO NHẤT ---")

# Báo cáo 1: 
query_report_1 = """
    SELECT * FROM IcebergCube_BUC
    WHERE Age != 'ALL' AND HighBP != 'ALL' AND HighChol != 'ALL' AND Smoker != 'ALL'
    ORDER BY Patient_Count DESC
    LIMIT 5
"""
print("\n[Báo cáo 1] 5 Nhóm đặc trưng chi tiết phổ biến nhất:")
print(pd.read_sql_query(query_report_1, conn).to_string(index=False))

# Báo cáo 2: 
query_report_2 = """
    SELECT HighBP, HighChol, Patient_Count 
    FROM IcebergCube_BUC
    WHERE HighBP = 1 AND HighChol = 1 
      AND Age = 'ALL' AND Smoker = 'ALL'
"""
print("\n[Báo cáo 2] Số lượng bệnh nhân vừa Huyết áp cao vừa Cholesterol cao:")
print(pd.read_sql_query(query_report_2, conn).to_string(index=False))

# Đóng kết nối DB
conn.close()