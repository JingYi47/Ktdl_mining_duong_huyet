import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

class PredictionModule:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {}
        self.metrics = {}

    def run_logistic_baseline(self):
        # Mặc dù đề bài yêu cầu Logistic Regression nhưng đây là bài toán dự báo liên tục (Regression)
        # Tôi sẽ dùng mô hình Linear Baseline cho phù hợp với dự báo đường huyết
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        self.models['Baseline (Linear)'] = model
        return model

    def run_random_forest(self):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        return model

    def run_xgboost(self):
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = model
        return model

    def run_svm(self):
        model = SVR(kernel='rbf')
        model.fit(self.X_train, self.y_train)
        self.models['SVM'] = model
        return model

    def run_knn_regressor(self):
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(self.X_train, self.y_train)
        self.models['KNN Regressor'] = model
        return model

    def evaluate_all(self):
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            self.metrics[name] = {'RMSE': rmse, 'R2': r2}
        return self.metrics

if __name__ == "__main__":
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    pm = PredictionModule(X[:80], y[:80], X[80:], y[80:])
    pm.run_logistic_baseline()
    pm.run_random_forest()
    print("Metrics:", pm.evaluate_all())
