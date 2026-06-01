import sys
import os
import json
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


def find_best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    """Find decision threshold that maximizes F1 score.

    With imbalanced data, the default 0.5 threshold often predicts mostly the
    negative class. ROC-AUC can still be decent, meaning threshold tuning can
    improve the precision/recall trade-off for the positive class.
    """

    best_t = 0.5
    best_f1 = -1.0
    best_metrics: dict[str, float] = {}

    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
            best_metrics = {
                "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1),
                "pred_positive_rate": float(y_pred.mean()),
            }

    return best_t, best_metrics


def find_best_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    """Find threshold that maximizes Youden's J = TPR - FPR.

    This criterion focuses on separating the two classes and is often used to
    pick an operating point on the ROC curve.
    """

    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = int(np.nanargmax(j))
    best_t = float(thresholds[best_idx])

    y_pred = (y_prob >= best_t).astype(int)
    best_metrics = {
        "threshold": best_t,
        "criterion": "youden_j",
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pred_positive_rate": float(y_pred.mean()),
        "tpr": float(tpr[best_idx]),
        "fpr": float(fpr[best_idx]),
        "youden_j": float(j[best_idx]),
    }
    return best_t, best_metrics

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

    # 5. THRESHOLD TUNING (suggested for app/demo)
    print("\n[5/4] Gợi ý ngưỡng (threshold) tối ưu để phân biệt 2 lớp...")

    lr_prob = lr_model.predict_proba(X_test_lr_scaled)[:, 1]
    rf_prob = rf_model.predict_proba(X_test_rf_w)[:, 1]

    # Use Youden's J for a more separation-focused threshold
    lr_t, lr_best = find_best_threshold_youden(y_test_lr.to_numpy(), lr_prob)
    rf_t, rf_best = find_best_threshold_youden(y_test_rf.to_numpy(), rf_prob)

    print("\n--- THRESHOLD SUGGESTION (Youden's J on ROC) ---")
    print(
        f"LR best_threshold={lr_t:.2f} | precision={lr_best['precision']:.4f} | recall={lr_best['recall']:.4f} | f1={lr_best['f1']:.4f}"
    )
    print(
        f"RF best_threshold={rf_t:.2f} | precision={rf_best['precision']:.4f} | recall={rf_best['recall']:.4f} | f1={rf_best['f1']:.4f}"
    )

    thresholds_path = MODELS_DIR / "thresholds.json"
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump({"lr": lr_best, "rf": rf_best}, f, ensure_ascii=False, indent=2)

    print(f"\n[*] Saved thresholds: {thresholds_path}")

    print(f"\\n[*] HOÀN TẤT PIPELINE! Toàn bộ Models và Artifacts đã được lưu trong thư mục: {MODELS_DIR}")

if __name__ == '__main__':
    main()
