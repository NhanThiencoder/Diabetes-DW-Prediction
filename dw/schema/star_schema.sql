-- Kịch bản tạo cấu trúc Star Schema cho hệ thống Data Warehouse Diabetes
-- File này thường được đặt trong dw/schema/ để định nghĩa DDL
-- Thiết kế CHUẨN dựa trên 22 cột thực tế của bộ dữ liệu BRFSS 2015
-- 1. Bảng Dim_Demographic (Thông tin nhân khẩu học)
CREATE TABLE IF NOT EXISTS Dim_Demographic (
    DemographicKey INTEGER PRIMARY KEY AUTOINCREMENT,
    Age INTEGER,
    Sex INTEGER,
    Education INTEGER,
    Income INTEGER
);
-- 2. Bảng Dim_Lifestyle (Thói quen sinh hoạt)
CREATE TABLE IF NOT EXISTS Dim_Lifestyle (
    LifestyleKey INTEGER PRIMARY KEY AUTOINCREMENT,
    Smoker INTEGER,
    PhysActivity INTEGER,
    Fruits INTEGER,
    Veggies INTEGER,
    HvyAlcoholConsump INTEGER
);
-- 3. Bảng Dim_HealthStatus (Tình trạng sức khỏe chung)
CREATE TABLE IF NOT EXISTS Dim_HealthStatus (
    HealthStatusKey INTEGER PRIMARY KEY AUTOINCREMENT,
    GenHlth INTEGER,
    MentHlth REAL,
    PhysHlth REAL,
    DiffWalk INTEGER
);
-- 4. Bảng Dim_MedicalHistory (Tiền sử bệnh lý)
CREATE TABLE IF NOT EXISTS Dim_MedicalHistory (
    MedicalHistoryKey INTEGER PRIMARY KEY AUTOINCREMENT,
    HighBP INTEGER,
    HighChol INTEGER,
    CholCheck INTEGER,
    Stroke INTEGER,
    HeartDiseaseorAttack INTEGER
);
-- 5. Bảng Dim_HealthcareAccess (Tiếp cận y tế)
CREATE TABLE IF NOT EXISTS Dim_HealthcareAccess (
    HealthcareAccessKey INTEGER PRIMARY KEY AUTOINCREMENT,
    AnyHealthcare INTEGER,
    NoDocbcCost INTEGER
);
-- 6. Bảng Fact_PatientHealth (Bảng Sự kiện trung tâm)
CREATE TABLE IF NOT EXISTS Fact_PatientHealth (
    PatientID INTEGER PRIMARY KEY,
    DemographicKey INTEGER,
    LifestyleKey INTEGER,
    HealthStatusKey INTEGER,
    MedicalHistoryKey INTEGER,
    HealthcareAccessKey INTEGER,
    BMI REAL,
    Diabetes_binary INTEGER,
    FOREIGN KEY (DemographicKey) REFERENCES Dim_Demographic(DemographicKey),
    FOREIGN KEY (LifestyleKey) REFERENCES Dim_Lifestyle(LifestyleKey),
    FOREIGN KEY (HealthStatusKey) REFERENCES Dim_HealthStatus(HealthStatusKey),
    FOREIGN KEY (MedicalHistoryKey) REFERENCES Dim_MedicalHistory(MedicalHistoryKey),
    FOREIGN KEY (HealthcareAccessKey) REFERENCES Dim_HealthcareAccess(HealthcareAccessKey)
);