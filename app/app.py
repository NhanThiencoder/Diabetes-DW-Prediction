import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================================
# CẤU HÌNH TRANG WEB & ĐƯỜNG DẪN
# ==========================================
st.set_page_config(page_title="Diabetes Prediction System", page_icon="🏥", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
NUMERIC_COLS_RF = ['BMI', 'PhysHlth', 'MentHlth']

# Hàm Winsorize 
def apply_winsor(df: pd.DataFrame, bounds: dict):
    out = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in out.columns and not (np.isnan(lo) or np.isnan(hi)):
            out[col] = out[col].clip(lower=lo, upper=hi)
    return out

# ==========================================
# THANH ĐIỀU HƯỚNG (SIDEBAR)
# ==========================================
st.sidebar.title("HỆ THỐNG Y TẾ AI 🏥")
st.sidebar.markdown("---")
st.sidebar.info("Hệ thống chẩn đoán cá nhân dự đoán nguy cơ mắc bệnh tiểu đường dựa trên bộ dữ liệu CDC BRFSS.")

# ==========================================
# GIAO DIỆN CHÍNH: CHẨN ĐOÁN CÁ NHÂN
# ==========================================
st.title("🩺 Chẩn đoán Cá nhân")
st.write("Nhập các chỉ số sức khỏe để hệ thống dự đoán nguy cơ mắc bệnh tiểu đường.")

# 1. LOAD MODELS & ARTIFACTS
try:
    rf_model = joblib.load(MODELS_DIR / 'random_forest.joblib')
    rf_preproc = joblib.load(MODELS_DIR / 'rf_preproc.joblib')
    imputer = rf_preproc['imputer']
    winsor_bounds = rf_preproc['winsor_bounds']
    model_loaded = True
except Exception as e:
    st.error(f"Lỗi tải mô hình: Vui lòng kiểm tra thư mục 'models'. Chi tiết: {e}")
    model_loaded = False

# --- CÁC HÀM QUY ĐỔI CHUẨN CDC ---
def map_age_to_cdc(age):
    if age < 25: return 1
    elif 25 <= age <= 29: return 2
    elif 30 <= age <= 34: return 3
    elif 35 <= age <= 39: return 4
    elif 40 <= age <= 44: return 5
    elif 45 <= age <= 49: return 6
    elif 50 <= age <= 54: return 7
    elif 55 <= age <= 59: return 8
    elif 60 <= age <= 64: return 9
    elif 65 <= age <= 69: return 10
    elif 70 <= age <= 74: return 11
    elif 75 <= age <= 79: return 12
    else: return 13

edu_options = {
    1: "Chưa tốt nghiệp Cấp 1", 2: "Tốt nghiệp Cấp 1", 3: "Tốt nghiệp Cấp 2",
    4: "Tốt nghiệp Cấp 3", 5: "Đại học/Cao đẳng (Chưa TN)", 6: "Tốt nghiệp Đại học/Cao đẳng"
}
income_options = {
    1: "Dưới 10.000$", 2: "10.000$ - 15.000$", 3: "15.000$ - 20.000$", 4: "20.000$ - 25.000$",
    5: "25.000$ - 35.000$", 6: "35.000$ - 50.000$", 7: "50.000$ - 75.000$", 8: "Trên 75.000$"
}
genhlth_options = {1: "Rất tốt (Excellent)", 2: "Tốt (Very Good)", 3: "Bình thường (Good)", 4: "Yếu (Fair)", 5: "Rất yếu (Poor)"}
yes_no = {0: "Không", 1: "Có"}

# 2. FORM NHẬP LIỆU 
with st.form("patient_form"):
    st.subheader("👤 Thông tin Nhân khẩu học & Chỉ số cơ thể")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Nhập tuổi thật -> Quy đổi ngầm
        raw_age = st.number_input("Độ tuổi thực tế", min_value=18, max_value=120, value=30)
        Sex = st.selectbox("Giới tính", options=[1, 0], format_func=lambda x: "Nam" if x==1 else "Nữ")
        Education = st.selectbox("Trình độ học vấn", options=list(edu_options.keys()), format_func=lambda x: edu_options[x], index=5)
        
    with col2:
        Income = st.selectbox("Mức thu nhập", options=list(income_options.keys()), format_func=lambda x: income_options[x], index=5)
        # Nhập Chiều cao, Cân nặng -> Tính BMI
        height_cm = st.number_input("Chiều cao (cm)", min_value=100.0, max_value=250.0, value=170.0)
        weight_kg = st.number_input("Cân nặng (kg)", min_value=30.0, max_value=200.0, value=65.0)
        
    with col3:
        GenHlth = st.selectbox("Đánh giá sức khỏe chung", options=list(genhlth_options.keys()), format_func=lambda x: genhlth_options[x], index=2)
        PhysHlth = st.number_input("Số ngày ốm thể chất (30 ngày qua)", 0, 30, 0)
        MentHlth = st.number_input("Số ngày tâm lý bất ổn (30 ngày qua)", 0, 30, 0)

    st.markdown("---")
    st.subheader("🩺 Tiền sử Bệnh lý & Lối sống")
    col4, col5, col6, col7 = st.columns(4)
    
    with col4:
        HighBP = st.selectbox("Huyết áp cao", options=[0, 1], format_func=lambda x: yes_no[x])
        HighChol = st.selectbox("Cholesterol cao", options=[0, 1], format_func=lambda x: yes_no[x])
        CholCheck = st.selectbox("Đã KT Cholesterol (5 năm)", options=[1, 0], format_func=lambda x: yes_no[x])
        
    with col5:
        HeartDiseaseorAttack = st.selectbox("Bệnh tim mạch", options=[0, 1], format_func=lambda x: yes_no[x])
        Stroke = st.selectbox("Từng bị đột quỵ", options=[0, 1], format_func=lambda x: yes_no[x])
        Smoker = st.selectbox("Hút thuốc lá (Ít nhất 100 điếu/đời)", options=[0, 1], format_func=lambda x: yes_no[x])
        
    with col6:
        PhysActivity = st.selectbox("Có tập thể dục (30 ngày qua)", options=[1, 0], format_func=lambda x: yes_no[x])
        Fruits = st.selectbox("Ăn trái cây mỗi ngày", options=[1, 0], format_func=lambda x: yes_no[x])
        Veggies = st.selectbox("Ăn rau củ mỗi ngày", options=[1, 0], format_func=lambda x: yes_no[x])
        
    with col7:
        HvyAlcoholConsump = st.selectbox("Uống nhiều cồn", options=[0, 1], format_func=lambda x: yes_no[x])
        AnyHealthcare = st.selectbox("Có Bảo hiểm y tế", options=[1, 0], format_func=lambda x: yes_no[x])
        NoDocbcCost = st.selectbox("Từng bỏ khám vì chi phí", options=[0, 1], format_func=lambda x: yes_no[x])
        DiffWalk = st.selectbox("Khó khăn khi đi lại/leo cầu thang", options=[0, 1], format_func=lambda x: yes_no[x])

    submit_button = st.form_submit_button("🔍 Tiến hành Dự đoán")
    
    # 3. XỬ LÝ KHI BẤM NÚT DỰ ĐOÁN
    if submit_button and model_loaded:
        # THỰC HIỆN TÍNH TOÁN TỪ DỮ LIỆU USER NHẬP VÀO
        Age = map_age_to_cdc(raw_age)
        BMI = weight_kg / ((height_cm / 100) ** 2)
        
        # Hiển thị thông số 
        st.info(f"💡 Hệ thống đã tự động tính toán: **Chỉ số BMI = {BMI:.2f}** | **Nhóm tuổi = {Age}** (theo chuẩn CDC).")
        
        # Gom dữ liệu đầu vào 
        input_dict = {
            'Age': [Age], 'Sex': [Sex], 'Education': [Education], 'Income': [Income],
            'Smoker': [Smoker], 'PhysActivity': [PhysActivity], 'Fruits': [Fruits],
            'Veggies': [Veggies], 'HvyAlcoholConsump': [HvyAlcoholConsump],
            'GenHlth': [GenHlth], 'MentHlth': [MentHlth], 'PhysHlth': [PhysHlth],
            'DiffWalk': [DiffWalk], 'HighBP': [HighBP], 'HighChol': [HighChol],
            'CholCheck': [CholCheck], 'Stroke': [Stroke], 'HeartDiseaseorAttack': [HeartDiseaseorAttack],
            'AnyHealthcare': [AnyHealthcare], 'NoDocbcCost': [NoDocbcCost], 'BMI': [BMI]
        }
        input_df = pd.DataFrame(input_dict)
        
        # Tiền xử lý & Dự đoán 
        input_num = pd.DataFrame(imputer.transform(input_df[NUMERIC_COLS_RF]), columns=NUMERIC_COLS_RF, index=input_df.index)
        input_df_imp = input_df.copy()
        input_df_imp[NUMERIC_COLS_RF] = input_num
        input_df_w = apply_winsor(input_df_imp, winsor_bounds)
        
        try:
            prediction = rf_model.predict(input_df_w)
            st.markdown("---")
            if prediction[0] == 0:
                st.success("🟢 KẾT QUẢ: Bệnh nhân **KHÔNG CÓ NGUY CƠ** mắc bệnh tiểu đường.")
            elif prediction[0] == 1: 
                st.warning("🟡 KẾT QUẢ: **TIỀN TIỂU ĐƯỜNG (Prediabetes)**. Cần điều chỉnh chế độ sinh hoạt!")
            else: 
                st.error("🔴 KẾT QUẢ: **CÓ NGUY CƠ CAO** mắc bệnh tiểu đường. Vui lòng can thiệp y tế!")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")