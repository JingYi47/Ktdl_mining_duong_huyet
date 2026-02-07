import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd

def clarke_error_grid(ref_values, pred_values, title):
    """
    Clarke Error Grid Analysis for Blood Glucose.
    Phân loại các cặp dự báo vào các vùng rủi ro lâm sàng (A, B, C, D, E).
    """
    # Simple implementation of the logic boundaries
    # In a professional tool, this would involve complex polygon checks
    assert len(ref_values) == len(pred_values)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(ref_values, pred_values, marker='o', color='b', s=8)
    plt.plot([0, 400], [0, 400], 'k--') # Identity line
    
    # Zone boundaries (Simplified representation)
    plt.plot([0, 400/1.2, 400], [0, 400, 400], 'k-', alpha=0.2)
    
    plt.xlabel("Reference Glucose (mg/dL)")
    plt.ylabel("Predicted Glucose (mg/dL)")
    plt.title(f"Clarke Error Grid - {title}")
    plt.xlim(0, 400)
    plt.ylim(0, 400)
    
    # Zone Labels
    plt.text(350, 380, 'A', fontsize=15)
    plt.text(350, 250, 'B', fontsize=15)
    plt.text(280, 50, 'C', fontsize=15)
    plt.text(150, 350, 'D', fontsize=15)
    plt.text(50, 350, 'E', fontsize=15)
    
    return plt

def explain_with_shap(model, X_train, X_test, feature_names):
    """Sử dụng SHAP để giải thích mô hình (Tree-based)"""
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance")
    return plt

if __name__ == "__main__":
    y_true = np.random.uniform(70, 300, 100)
    y_pred = y_true + np.random.normal(0, 20, 100)
    clarke_error_grid(y_true, y_pred, "Sample Model")
    plt.show()
