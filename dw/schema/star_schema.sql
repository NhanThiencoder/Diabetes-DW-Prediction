-- Kịch bản tạo cấu trúc Star Schema cho hệ thống Data Warehouse Diabetes
-- File này thường được đặt trong dw/schema/ để định nghĩa DDL

-- 1. Bảng Dim_Demographic
CREATE TABLE IF NOT EXISTS Dim_Demographic (
    DemographicKey INTEGER PRIMARY KEY AUTOINCREMENT,
    AgeGroup INTEGER,
    Sex INTEGER,
    Education INTEGER,
    Income INTEGER
);

-- 2. Bảng Dim_Lifestyle
CREATE TABLE IF NOT EXISTS Dim_Lifestyle (
    LifestyleKey INTEGER PRIMARY KEY AUTOINCREMENT,
    Smoker INTEGER,
    PhysActivity INTEGER
);

-- 3. Bảng Dim_HealthStatus
CREATE TABLE IF NOT EXISTS Dim_HealthStatus (
    HealthStatusKey INTEGER PRIMARY KEY AUTOINCREMENT,
    GenHlth INTEGER,
    HighBP INTEGER,
    HighChol INTEGER
);

-- 4. Bảng Fact_PatientHealth
CREATE TABLE IF NOT EXISTS Fact_PatientHealth (
    PatientID INTEGER PRIMARY KEY,
    DemographicKey INTEGER,
    LifestyleKey INTEGER,
    HealthStatusKey INTEGER,
    BMI REAL,
    Diabetes_binary INTEGER,
    FOREIGN KEY (DemographicKey) REFERENCES Dim_Demographic(DemographicKey),
    FOREIGN KEY (LifestyleKey) REFERENCES Dim_Lifestyle(LifestyleKey),
    FOREIGN KEY (HealthStatusKey) REFERENCES Dim_HealthStatus(HealthStatusKey)
);
