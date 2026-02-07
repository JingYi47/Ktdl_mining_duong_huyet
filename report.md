# BÁO CÁO KỸ THUẬT: PHÂN TÍCH VÀ DỰ BÁO CHỈ SỐ ĐƯỜNG HUYẾT

**Nhóm thực hiện: Nhóm 10**

## 1. Business Understanding (CRISP-DM Phase 1)

Mục tiêu cốt lõi của đề tài là ứng dụng các kỹ thuật khai phá dữ liệu để giúp bệnh nhân và bác sĩ theo dõi, phân loại tình trạng bệnh lý và dự báo sớm chỉ số đường huyết, từ đó giảm thiểu rủi ro biến chứng tiểu đường. Hệ thống tập trung vào việc dự báo chính xác nồng độ Glucose dựa trên lịch sử dữ liệu và các đặc trưng sinh học.

## 2. Data Understanding & Preparation

- **Dữ liệu**: Thông tin định danh, chỉ số sinh tồn (BMI, Age, HBA1C) và chuỗi thời gian nồng độ Glucose từ thiết bị CGM.
- **Tiền xử lý**:
  - Xử lý giá trị thiếu bằng phương pháp **Nội suy (Interpolation)** và **KNN Imputer**.
  - Làm mượt chuỗi thời gian bằng **Moving Average (Cửa sổ trượt)** để giảm nhiễu từ cảm biến.
  - Trích xuất đặc trưng trễ (**Lag-features**) và các đặc trưng thời gian (Giờ, Ngày trong tuần).

## 3. Unsupervised Learning (Clustering)

Chúng tôi triển khai 05 thuật toán để phân nhóm bệnh nhân dựa trên đặc điểm sinh học:

1. **K-Means**: Phân cụm dựa trên khoảng cách.
2. **Hierarchical**: Xây dựng cấu trúc phân cấp.
3. **DBSCAN**: Xác định vùng mật độ và loại bỏ nhiễu.
4. **GMM**: Mô hình hỗn hợp Gaussian dùng xác suất.
5. **Mean Shift**: Tìm trọng tâm mật độ dữ liệu.

## 4. Supervised Learning (Prediction)

Hệ thống dự báo gồm nhiều mô hình được huấn luyện và so sánh:

- **Baseline (Linear Regression)**: Mô hình tham chiếu.
- **Random Forest**: Xử lý tốt dữ liệu phi tuyến.
- **XGBoost**: Thuật toán Gradient Boosting hiệu năng cao nhất.
- **SVM**: Phù hợp với các bài toán biên độ hẹp.
- **KNN Regressor**: Dự báo dựa trên sự tương đồng của lịch sử.

## 5. Evaluation & Explainable AI (XAI)

- **Đánh giá**: Sử dụng chỉ số R2 Score và RMSE. Ứng dụng **Clarke Error Grid** để đánh giá mức độ rủi ro lâm sàng trong y tế.
- **XAI**: Sử dụng **SHAP** để giải thích đóng góp của từng biến (BMI, Age, Lag_1) vào kết quả dự báo của AI.

## 6. Deployment

Ứng dụng được triển khai dưới dạng **Interactive Dashboard** bằng Streamlit, hỗ trợ:

- Tải file dữ liệu thực tế (CSV) dung lượng lớn.
- Mô phỏng dự báo nhanh (What-if Analysis).
- Cảnh báo ngưỡng y tế (Hyper/Hypoglycemia).

## 7. Kết luận và Hướng phát triển

Dự án đã xây dựng thành công bộ công cụ khai thác dữ liệu toàn diện. Hướng phát triển tương lai sẽ tập trung vào việc tích hợp dữ liệu Insulin và các hoạt động vận động của bệnh nhân để tăng độ chính xác của dự báo dài hạn.
