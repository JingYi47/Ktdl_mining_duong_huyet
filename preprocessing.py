import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataPreprocessor:
    def __init__(self, method='knn'):
        self.method = method
        self.scaler = MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        
    def handle_missing_values(self, df, column='Glucose'):
        # Xử lý giá trị thiếu bằng KNN hoặc Interpolation
        if self.method == 'knn':
            # KNN cần dữ liệu dạng mảng 2D
            df[[column]] = self.imputer.fit_transform(df[[column]])
        else:
            df[column] = df[column].interpolate(method='linear')
        return df

    def apply_moving_average(self, df, column='Glucose', window=4):
        # Làm mượt dữ liệu bằng Moving Average
        df[f'{column}_Smooth'] = df.groupby('PatientID')[column].transform(
            lambda x: x.rolling(window=window, min_periods=1, center=True).mean()
        )
        return df

    def feature_engineering(self, df, column='Glucose'):
        # Xây dựng Lag features và Rolling windows
        # Lag features
        for i in [1, 2, 4]: # 1 step, 2 steps, 1 day (4 steps)
            df[f'{column}_Lag_{i}'] = df.groupby('PatientID')[column].shift(i)
        
        # Rolling stats
        df[f'{column}_Roll_Mean'] = df.groupby('PatientID')[column].transform(
            lambda x: x.rolling(window=4).mean()
        )
        df[f'{column}_Roll_Std'] = df.groupby('PatientID')[column].transform(
            lambda x: x.rolling(window=4).std()
        )
        
        # Time components
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        
        # Chỉ xóa những dòng mà nồng độ Glucose mục tiêu bị thiếu hoặc lỗi Timestamp
        return df.dropna(subset=[column, 'Timestamp'])

    def scale_data(self, df, features):
        """Chuẩn hóa dữ liệu"""
        df[features] = self.scaler.fit_transform(df[features])
        return df, self.scaler

if __name__ == "__main__":
    # Test preprocessing
    from data_generator import generate_diabetes_data
    df = generate_diabetes_data(n_patients=2)
    preprocessor = DataPreprocessor()
    
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.apply_moving_average(df)
    df = preprocessor.feature_engineering(df)
    
    features = ['Glucose', 'Age', 'BMI', 'Glucose_Lag_1', 'Hour']
    df_scaled, _ = preprocessor.scale_data(df, features)
    
    print("Preprocessing complete. Sample data:")
    print(df_scaled.head())
