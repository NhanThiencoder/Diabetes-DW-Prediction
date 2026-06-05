import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.graph_objects as go

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
    # Optional: recommended threshold persisted from training
    thresholds_path = MODELS_DIR / 'thresholds.json'
    if thresholds_path.exists():
        with open(thresholds_path, 'r', encoding='utf-8') as f:
            thresholds = json.load(f)
        rf_threshold_default = float(thresholds.get('rf', {}).get('threshold', 0.5))
    else:
        rf_threshold_default = 0.5
    model_loaded = True
except Exception as e:
    st.error(f"Lỗi tải mô hình: Vui lòng kiểm tra thư mục 'models'. Chi tiết: {e}")
    model_loaded = False

# Fixed threshold for consistent demo behavior (set during training)
rf_threshold = float(rf_threshold_default) if model_loaded else 0.5

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
            # Ẩn việc tính toán xác suất ở background, chỉ dùng để ra quyết định
            proba = float(rf_model.predict_proba(input_df_w)[:, 1][0])
            prediction = int(proba >= float(rf_threshold))
            
            st.markdown("---")
            st.subheader("📊 KẾT QUẢ CHẨN ĐOÁN & SUY LUẬN Y KHOA TỪ AI")
            
            # 1. HIỂN THỊ KẾT LUẬN CHÍNH
            if prediction == 0:
                st.success("🟢 KẾT LUẬN TỔNG QUAN: Bệnh nhân **KHÔNG CÓ NGUY CƠ** mắc bệnh tiểu đường ở thời điểm hiện tại.")
            else:
                st.error("🔴 KẾT LUẬN TỔNG QUAN: Bệnh nhân **CÓ NGUY CƠ CAO** mắc bệnh tiểu đường. Khuyến nghị thực hiện xét nghiệm HbA1c!")

            st.markdown("---")
            
            # ==========================================
            # 2. BIỂU ĐỒ PHÂN TÍCH RỦI RO (CĂN GIỮA)
            # ==========================================
            st.write("**🔍 Biểu đồ Phân bổ Rủi ro Đa chiều:**")
            
            # Tính điểm đánh giá theo cụm
            body_score = 0
            if BMI >= 30: body_score += 7
            elif BMI >= 25: body_score += 4
            if raw_age >= 60: body_score += 3
            elif raw_age >= 45: body_score += 1
            body_score = min(body_score, 10)

            cardio_score = (HighBP * 4) + (HighChol * 4) + (HeartDiseaseorAttack * 2)
            cardio_score = min(cardio_score, 10)

            lifestyle_bad = (Smoker * 4) + (HvyAlcoholConsump * 3)
            lifestyle_good = (PhysActivity * 3) + (Fruits * 2) + (Veggies * 2)
            lifestyle_score = max(0, min(5 + lifestyle_bad - lifestyle_good, 10))

            # Vẽ Radar Chart
            categories = ['Thể trạng (BMI/Tuổi)', 'Bệnh lý Mạch máu', 'Lối sống & Thói quen']
            patient_scores = [body_score, cardio_score, lifestyle_score]
            warning_levels = [5, 5, 5]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=patient_scores + [patient_scores[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Hồ sơ của bạn',
                fillcolor="rgba(255, 0, 0, 0.4)" if prediction == 1 else "rgba(0, 128, 0, 0.4)",
                line_color="red" if prediction == 1 else "green"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=warning_levels + [warning_levels[0]],
                theta=categories + [categories[0]],
                fill='none',
                name='Ngưỡng cảnh báo',
                line_color='orange',
                line_dash='dash'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=True,
                height=450,
                margin=dict(l=40, r=40, t=40, b=10)
            )
            
            # Dùng st.columns để ép biểu đồ vào chính giữa trang
            col_spacer1, col_center, col_spacer2 = st.columns([1, 2, 1])
            with col_center:
                st.plotly_chart(fig_radar, use_container_width=True)

            # ==========================================
            # 3. SUY LUẬN LÂM SÀNG DỰA TRÊN DỮ LIỆU
            # ==========================================
            st.markdown("---")
            st.write("**🧠 Suy luận Lâm sàng từ Dữ liệu Bệnh lý & Lối sống:**")
            
            risk_factors = []
            protective_factors = []
            
            # --- Phân tích chéo Bệnh lý ---
            medical_issues = sum([HighBP, HighChol, Stroke, HeartDiseaseorAttack])
            
            if medical_issues >= 2 and BMI >= 25:
                risk_factors.append(f"**Hội chứng chuyển hóa:** Việc thừa cân (BMI={BMI:.1f}) đi kèm đa bệnh lý nền ({medical_issues} bệnh lý tim mạch/huyết áp) đang tạo sức ép lớn lên tuyến tụy, làm kháng insulin trầm trọng.")
            elif medical_issues >= 2:
                risk_factors.append(f"**Báo động Hệ tuần hoàn:** Dù cân nặng ổn định, việc mắc {medical_issues} chứng bệnh nền về mạch máu là rủi ro cốt lõi đẩy nhanh nguy cơ tiểu đường.")
            elif (HighBP == 1 or HighChol == 1) and BMI >= 30:
                risk_factors.append(f"**Tích tụ mỡ nội tạng:** Thể trạng béo phì (BMI={BMI:.1f}) kết hợp rối loạn huyết áp/mỡ máu là tín hiệu rõ ràng của việc suy giảm trao đổi chất.")
            elif HighBP == 1 or HighChol == 1:
                risk_factors.append("**Tín hiệu cảnh báo sớm:** Huyết áp hoặc Cholesterol đang ở mức cao, cần kiểm soát ngay để chặn đứng chuỗi rối loạn chuyển hóa.")
            else:
                protective_factors.append("**Nền tảng sinh lý ổn định:** Không phát hiện các bệnh lý nền nguy hiểm về hệ tuần hoàn, giảm tải đáng kể rủi ro cho cơ thể.")

            # --- Phân tích chéo Lối sống ---
            bad_habits = sum([Smoker, HvyAlcoholConsump])
            good_habits = sum([PhysActivity, Fruits, Veggies])
            
            if bad_habits > 0 and good_habits == 0:
                risk_factors.append("**Báo động đỏ Lối sống:** Sử dụng chất kích thích (khói thuốc/cồn) cộng với việc hoàn toàn tĩnh tại đang trực tiếp phá hủy chức năng điều tiết đường huyết tự nhiên.")
            elif bad_habits > 0 and good_habits >= 2:
                risk_factors.append("**Lối sống mâu thuẫn:** Dù có duy trì vận động/dinh dưỡng, độc tố từ thuốc lá hoặc cồn vẫn sinh ra stress oxy hóa âm thầm làm tổn thương tế bào.")
            elif bad_habits == 0 and good_habits == 3:
                protective_factors.append("**Thói quen bảo vệ xuất sắc:** Chế độ ăn giàu chất xơ kết hợp vận động thường xuyên tạo ra lá chắn vững chắc chống lại tiểu đường tuýp 2.")
            elif PhysActivity == 0 and BMI >= 25:
                risk_factors.append("**Vòng lặp tĩnh tại:** Thiếu vận động làm trầm trọng thêm tình trạng thừa cân, khiến lượng đường trong máu không được chuyển hóa vào cơ bắp.")
            elif good_habits >= 1:
                protective_factors.append("**Thói quen tích cực:** Việc duy trì được thói quen tốt như ăn rau xanh hoặc vận động nhẹ đang giúp cơ thể giữ thăng bằng nhất định.")

            # Hiển thị 2 cột rủi ro và bảo vệ
            col_warn, col_good = st.columns(2)
            with col_warn:
                st.warning("⚠️ **YẾU TỐ RỦI RO NGHIÊM TRỌNG**")
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"- {factor}")
                else:
                    st.write("- Tốt! Không phát hiện cụm rủi ro đáng kể nào.")
                    
            with col_good:
                st.success("✅ **YẾU TỐ BẢO VỆ & KHUYẾN NGHỊ**")
                if protective_factors:
                    for factor in protective_factors:
                        st.write(f"- {factor}")
                else:
                    st.write("- Đề nghị thiết lập lại toàn diện lối sống khỏe mạnh.")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình phân tích: {e}")