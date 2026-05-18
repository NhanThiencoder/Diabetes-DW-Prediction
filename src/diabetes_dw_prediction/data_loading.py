from __future__ import annotations

from pathlib import Path

import pandas as pd


__all__ = [
    "load_brfss_diabetes_csv",
    "load_and_prepare_brfss",
]

def load_brfss_diabetes_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load the BRFSS diabetes indicators dataset from a CSV file."""
    return pd.read_csv(Path(csv_path))


def load_and_prepare_brfss() -> pd.DataFrame:
    """
    Tải dữ liệu BRFSS từ Google Drive nếu chưa có, giải nén, đọc CSV vào DataFrame.
    Trả về: DataFrame
    """
    import os
    import zipfile
    try:
        import gdown
    except ImportError:
        raise ImportError("Bạn cần cài đặt gdown: pip install gdown")

    FILE_ID = '1M2S6kV8cA3PVHsHF3czXbwVj43HWMRan'
    URL = f'https://drive.google.com/uc?id={FILE_ID}'
    zip_path = 'data/raw/dataset.zip'
    extract_folder = 'data/raw'
    csv_file = 'diabetes_binary_health_indicators_BRFSS2015.csv'
    csv_file_path = os.path.join(extract_folder, csv_file)

    if not os.path.exists(csv_file_path):
        print("[*] Không tìm thấy dữ liệu ở Local. Đang tiến hành tải từ Google Drive...")
        gdown.download(URL, zip_path, quiet=False)
        print("\n[*] Đang giải nén dữ liệu...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        os.remove(zip_path)
        print("[*] Hoàn tất quá trình đồng bộ dữ liệu!")
    else:
        print("[*] Dữ liệu đã có sẵn trong máy, bỏ qua bước tải.")

    print(f"[*] Đang đọc file vào Pandas từ: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    print(f"-> Thành công! Kích thước dữ liệu: {df.shape[0]:,} dòng.")
    return df
