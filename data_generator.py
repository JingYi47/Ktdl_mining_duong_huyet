import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_diabetes_data(n_patients=10, days=30):
    
    # Tạo dữ liệu giả lập chỉ số đường huyết cho bệnh nhân tiểu đường.
    # Bao gồm cả dữ liệu bảng (patient info) và chuỗi thời gian (glucose levels).

    np.random.seed(42)
    
    # 1. Dữ liệu bảng (Static Info)
    patient_ids = [f"P{i:03d}" for i in range(1, n_patients + 1)]
    static_data = pd.DataFrame({
        'PatientID': patient_ids,
        'Age': np.random.randint(20, 80, n_patients),
        'BMI': np.random.uniform(18.5, 35.0, n_patients),
        'HBA1C': np.random.uniform(5.0, 9.0, n_patients),
        'Gender': np.random.choice(['Male', 'Female'], n_patients)
    })
    
    # 2. Dữ liệu chuỗi thời gian (Glucose)
    timeseries_list = []
    start_date = datetime(2025, 1, 1)
    
    for pid in patient_ids:
        current_time = start_date
        # 4 lần đo mỗi ngày: Sáng sớm, sau ăn sáng, sau ăn trưa, trước ngủ
        for _ in range(days * 4):
            # Baseline glucose depends on HBA1C
            hba1c = static_data[static_data['PatientID'] == pid]['HBA1C'].values[0]
            base_glucose = hba1c * 15 + np.random.normal(0, 5)
            
            # Add some variability and noise
            glucose = base_glucose + np.random.normal(0, 10)
            
            timeseries_list.append({
                'PatientID': pid,
                'Timestamp': current_time,
                'Glucose': max(40, glucose) # Glucose can't be negative/too low
            })
            current_time += timedelta(hours=6)
            
    ts_data = pd.DataFrame(timeseries_list)
    
    # Merge static and ts data
    full_data = pd.merge(ts_data, static_data, on='PatientID')
    
    # Giả lập giá trị thiếu (5% data)
    mask = np.random.random(full_data.shape[0]) < 0.05
    full_data.loc[mask, 'Glucose'] = np.nan
    
    return full_data

if __name__ == "__main__":
    df = generate_diabetes_data()
    df.to_csv("diabetes_data.csv", index=False)
    print("Generated synthetic data: diabetes_data.csv")
