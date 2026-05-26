import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump

# Model building functions
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Thêm project_root vào sys.path để import các module local
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.models import build_logistic_regression, build_random_forest

# Cấu hình đường dẫn
SPLITS_DIR = PROJECT_ROOT / 'data' / 'processed' / 'splits'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'Diabetes_binary'
NUMERIC_COLS_LR = ['BMI', 'PhysHlth', 'MentHlth']
# Dành cho RF: những cột numeric cần impute & winsorize
NUMERIC_COLS_RF = ['BMI', 'PhysHlth', 'MentHlth']

def fit_winsor_bounds(df: pd.DataFrame, cols: list[str], low_q: float = 0.01, high_q: float = 0.99):
    bounds = {}
    for col in cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if series.empty:
            bounds[col] = (np.nan, np.nan)
            continue
        bounds[col] = (float(series.quantile(low_q)), float(series.quantile(high_q)))
    return bounds

def apply_winsor(df: pd.DataFrame, bounds: dict):
    out = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in out.columns and not (np.isnan(lo) or np.isnan(hi)):
            out[col] = out[col].clip(lower=lo, upper=hi)
    return out

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    
    print(f"\\n--- KẾT QUẢ ĐÁNH GIÁ: {name} ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, y_prob):.4f}")

def main():
    print("[*] KIỂM TRA DỮ LIỆU ĐẦU VÀO...")
    if not (SPLITS_DIR / 'train_lr.csv').exists():
        print("LỖI: Chưa tìm thấy dữ liệu Train/Test splits.")
        print("Vui lòng chạy lệnh: python scripts/split_from_dw.py trước khi chạy train.py")
        sys.exit(1)

    # 1. LOAD RAW SPLITS
    print("[1/4] Đang nạp dữ liệu train/test từ thư mục splits...")
    train_lr = pd.read_csv(SPLITS_DIR / 'train_lr.csv')
    test_lr = pd.read_csv(SPLITS_DIR / 'test_lr.csv')
    train_rf = pd.read_csv(SPLITS_DIR / 'train_rf.csv')
    test_rf = pd.read_csv(SPLITS_DIR / 'test_rf.csv')

    # Tách X, y cho LR
    X_train_lr = train_lr.drop(columns=[TARGET])
    y_train_lr = train_lr[TARGET].astype('int64')
    X_test_lr = test_lr.drop(columns=[TARGET])
    y_test_lr = test_lr[TARGET].astype('int64')

    # Tách X, y cho RF
    X_train_rf = train_rf.drop(columns=[TARGET])
    y_train_rf = train_rf[TARGET].astype('int64')
    X_test_rf = test_rf.drop(columns=[TARGET])
    y_test_rf = test_rf[TARGET].astype('int64')

    # 2. PREPROCESSING
    print("[2/4] Đang xử lý dữ liệu (Preprocessing)...")
    
    # --- Logistic Regression Preprocessing ---
    scaler = StandardScaler()
    X_train_lr_scaled = X_train_lr.copy()
    X_test_lr_scaled = X_test_lr.copy()
    
    X_train_lr_scaled[NUMERIC_COLS_LR] = scaler.fit_transform(X_train_lr[NUMERIC_COLS_LR])
    X_test_lr_scaled[NUMERIC_COLS_LR] = scaler.transform(X_test_lr[NUMERIC_COLS_LR])
    
    # Lưu artifacts
    dump(scaler, MODELS_DIR / 'lr_scaler.joblib')

    # --- Random Forest Preprocessing ---
    imputer = SimpleImputer(strategy='median')
    X_train_rf_num = pd.DataFrame(imputer.fit_transform(X_train_rf[NUMERIC_COLS_RF]), columns=NUMERIC_COLS_RF, index=X_train_rf.index)
    X_test_rf_num = pd.DataFrame(imputer.transform(X_test_rf[NUMERIC_COLS_RF]), columns=NUMERIC_COLS_RF, index=X_test_rf.index)

    X_train_rf_imp = X_train_rf.copy()
    X_test_rf_imp = X_test_rf.copy()
    X_train_rf_imp[NUMERIC_COLS_RF] = X_train_rf_num
    X_test_rf_imp[NUMERIC_COLS_RF] = X_test_rf_num

    bounds = fit_winsor_bounds(X_train_rf_imp, NUMERIC_COLS_RF, low_q=0.01, high_q=0.99)
    X_train_rf_w = apply_winsor(X_train_rf_imp, bounds)
    X_test_rf_w = apply_winsor(X_test_rf_imp, bounds)

    # Lưu artifacts
    dump({'imputer': imputer, 'winsor_bounds': bounds}, MODELS_DIR / 'rf_preproc.joblib')

    # 3. TRAINING
    print("[3/4] Bắt đầu quá trình Train Model...")
    
    print("  -> Training Logistic Regression...")
    lr_model = build_logistic_regression(random_state=42)
    lr_model.fit(X_train_lr_scaled, y_train_lr)

    print("  -> Training Random Forest...")
    rf_model = build_random_forest(random_state=42)
    rf_model.fit(X_train_rf_w, y_train_rf)

    # Lưu Model
    dump(lr_model, MODELS_DIR / 'logistic_regression.joblib')
    dump(rf_model, MODELS_DIR / 'random_forest.joblib')

    # 4. EVALUATION
    print("[4/4] Quá trình Train hoàn tất! Đang tính toán các chỉ số đánh giá...")
    evaluate_model('Logistic Regression', lr_model, X_test_lr_scaled, y_test_lr)
    evaluate_model('Random Forest', rf_model, X_test_rf_w, y_test_rf)

    print(f"\\n[*] HOÀN TẤT PIPELINE! Toàn bộ Models và Artifacts đã được lưu trong thư mục: {MODELS_DIR}")

if __name__ == '__main__':
    main()
