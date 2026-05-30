# Diabetes-DW-Prediction

## Pipeline chạy lại từ đầu (khuyến nghị)

### 1) Build Data Warehouse (có preprocess trước DW)

- Tạo lại DW từ dữ liệu raw + tiền xử lý (ép kiểu + optional winsorize cho `BMI/PhysHlth/MentHlth`) rồi load star schema vào SQLite:

`python src/etl_pipeline.py`

Tuỳ chọn:
- Tắt preprocess trước DW: `python src/etl_pipeline.py --no-preprocess`
- Tắt winsorize: `python src/etl_pipeline.py --no-winsorize`
- Đổi ngưỡng winsorize: `python src/etl_pipeline.py --winsor-low-q 0.01 --winsor-high-q 0.99`

Kết quả: file DW ở `dw/data_warehouse.db`.

### 2) (Tuỳ chọn) Tính Iceberg Cube (BUC) từ DW

- Chạy demo BUC đọc dữ liệu từ DW (join star schema) và in top kết quả:

`python scripts/run_buc_from_dw.py`

- Nếu muốn lưu kết quả vào SQLite (bảng mặc định `IcebergCube_BUC`):

`python scripts/run_buc_from_dw.py --save-to-dw`

Gợi ý tuỳ chọn:
- Chọn dimensions: `--dims Age,Sex,Smoker,HighBP`
- Đổi min_support: `--min-support 2000`

### 3) Trích xuất từ DW + chia train/test

- JOIN star schema -> tạo dataset phẳng -> stratified split -> lưu CSV cho LR/RF:

`python scripts/split_from_dw.py`

Kết quả: `data/processed/splits/train_lr.csv`, `test_lr.csv`, `train_rf.csv`, `test_rf.csv`.

### 4) Preprocess theo model + train (LR + RF)

- Train Logistic Regression và Random Forest (RF là main model), đồng thời lưu artifacts preprocess:

`python scripts/train.py`

Kết quả:
- Models: `models/logistic_regression.joblib`, `models/random_forest.joblib`
- Artifacts: `models/lr_scaler.joblib`, `models/rf_preproc.joblib`