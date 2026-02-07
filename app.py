import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_generator import generate_diabetes_data
from preprocessing import DataPreprocessor
from clustering_models import ClusteringModule
from prediction_models import PredictionModule
from evaluation import clarke_error_grid, explain_with_shap
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Diabetes Insight AI | Nhóm 10",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0e1117;
    }
    
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #3d4455;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e2130;
        padding: 10px;
        border-radius: 15px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 10px;
        padding: 0 20px;
        background-color: transparent;
        color: #888;
        font-weight: 600;
        transition: all 0.3s;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b !important;
        color: white !important;
    }
    
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .header-box {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8a8a 100%);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("""
    <div class="header-box">
        <h1 style='margin:0; font-size: 2.5rem;'>🚀 Diabetes Insight AI Portal</h1>
        <p style='margin:0; font-size: 1.1rem; opacity: 0.9;'>Hệ thống Khai thác dữ liệu & Dự báo đường huyết thông minh - Nhóm 10</p>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=100)
    st.title("🎛️ Control Panel")
    data_source = st.radio("Nguồn dữ liệu đầu vào", ["Dữ liệu giả lập (Simulation)", "Tải lên Dataset (.CSV)"])
    
    st.divider()
    
    if data_source == "Dữ liệu giả lập (Simulation)":
        df = generate_diabetes_data()
        st.success(" Đã tạo 1,200 bản ghi giả lập")
    else:
        uploaded_file = st.file_uploader("Chọn file CSV của bạn", type="csv")
        if uploaded_file:
            # Cho phép người dùng chọn số lượng dòng để nạp (Tối ưu tốc độ)
            sample_size = st.sidebar.slider("Số lượng dòng nạp vào (Tối ưu tốc độ)", 1000, 50000, 10000, step=1000)
            df = pd.read_csv(uploaded_file, nrows=sample_size)
            st.sidebar.info(f"⚡ Đã nạp {len(df):,} dòng đầu tiên để xử lý nhanh.")
            
            # 1. Tự động xử lý Timestamp nếu có cột Date & Time
            if 'Date' in df.columns and 'Time' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                st.sidebar.info("📅 Đã gộp Date & Time thành Timestamp")
            elif 'Timestamp' not in df.columns:
                # Nếu không có gì hết, tạo timestamp giả lập dựa trên Index hoặc thứ tự
                df['Timestamp'] = pd.date_range(start='2025-01-01', periods=len(df), freq='5min')
                st.sidebar.warning("⚠️ Không tìm thấy cột thời gian, đã tạo giả lập")

            # 2. Tự động phát hiện cột Glucose (Ví dụ: CGM, BG, Sensor Glucose (mg/dL))
            possible_names = ['Glucose', 'CGM', 'BG', 'GlucoseValue', 'Value', 'Sensor Glucose (mg/dL)']
            found_glucose = False
            for col in df.columns:
                if col in possible_names or 'glucose' in col.lower():
                    df = df.rename(columns={col: 'Glucose'})
                    st.info(f"🔍 Đã nhận diện cột: **{col}** -> Glucose")
                    found_glucose = True
                    break
            
            if not found_glucose:
                st.error("❌ Không tìm thấy cột Glucose")
                st.stop()

            # 3. Xử lý PatientID nếu thiếu
            if 'PatientID' not in df.columns:
                df['PatientID'] = 'Patient_001'
                st.sidebar.info("👤 Đã gán nhãn Patient_001 cho toàn bộ dữ liệu")
            
            # Xử lý các cột Age, BMI, HBA1C (Điền giá trị mặc định nếu thiếu hoặc có NaN)
            defaults = {'Age': 50.0, 'BMI': 30.0, 'HBA1C': 6.5}
            for col, val in defaults.items():
                if col not in df.columns:
                    df[col] = val
                else:
                    df[col] = df[col].fillna(val)

            # Đảm bảo Glucose là số và điền khuyết bước đầu bằng nội suy/ffill
            df['Glucose'] = pd.to_numeric(df['Glucose'], errors='coerce')
            df['Glucose'] = df.groupby('PatientID', group_keys=False)['Glucose'].apply(lambda x: x.ffill().bfill())

            # Báo cáo tình trạng nạp dữ liệu (Sidebar)
            st.sidebar.success("✅ Nạp dữ liệu thành công!")
            with st.sidebar.expander("🛠️ Chi tiết Nhận diện Dữ liệu"):
                st.write(f"📂 File: `{uploaded_file.name}`")
                st.write(f"📊 Tổng số cột: {len(df.columns)}")
                st.write(f"🧬 Cột Glucose: {'OK' if not df['Glucose'].isna().all() else 'Trống!'}")
                st.write(f"📅 Timestamp: {'OK' if not df['Timestamp'].isna().all() else 'Lỗi định dạng!'}")
                st.write(f"👤 Bệnh nhân: {df['PatientID'].nunique()} người")
        else:
            st.warning("⚠️ Vui lòng nạp dữ liệu để bắt đầu")
            st.stop()
    
    st.info("💡 Tip: Sử dụng tab 'Đánh giá & XAI' để giải thích kết quả AI.")

# --- PROCESSING ---
preprocessor = DataPreprocessor()

# Global Configuration for Analysis
target = 'Glucose'
features_pr = ['Age', 'BMI', 'HBA1C', 'Glucose_Lag_1', 'Glucose_Lag_2', 'Hour', 'DayOfWeek']

# Đảm bảo cột Glucose là kiểu số (đề phòng có chuỗi rác)
df['Glucose'] = pd.to_numeric(df['Glucose'], errors='coerce')

df_clean = preprocessor.handle_missing_values(df.copy())
df_smooth = preprocessor.apply_moving_average(df_clean)
df_final = preprocessor.feature_engineering(df_smooth)

if df_final.empty:
    st.error("❌ Dữ liệu sau khi xử lý bị trống. Có thể do file quá ngắn hoặc quá nhiều giá trị lỗi. Vui lòng thử dùng Dữ liệu giả lập để kiểm tra Dashboard.")
    st.stop()

# --- MAIN TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Tiền xử lý Dữ liệu", 
    "🧩 Phân cụm (Unsupervised)", 
    "📈 Dự báo & Mô hình", 
    "🛡️ Phân tích rủi ro & XAI"
])

# --- TAB 1: PREPROCESSING ---
with tab1:
    st.subheader("🛠️ Phân tích & Tiền xử lý Dữ liệu Chi tiết")
    
    # --- Data Insights Cards ---
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Tổng số mẫu", f"{len(df):,}")
    with col_stat2:
        missing_pct = (df['Glucose'].isna().sum() / len(df)) * 100
        st.metric("Tỷ lệ thiếu (NaN)", f"{missing_pct:.1f}%")
    with col_stat3:
        st.metric("Giá trị Trung bình", f"{df['Glucose'].mean():.1f} mg/dL")
    with col_stat4:
        st.metric("Độ lệch chuẩn", f"{df['Glucose'].std():.1f}")

    st.divider()

    # --- Statistics & Quality ---
    col_ins1, col_ins2 = st.columns([1, 1])
    with col_ins1:
        st.markdown("### 📊 Thống kê mô tả (Descriptive Statistics)")
        st.dataframe(df.describe().T, use_container_width=True)
    
    with col_ins2:
        st.markdown("### 🔍 Phân phối nồng độ Đường huyết")
        fig_dist = px.histogram(df_final, x="Glucose", nbins=50, 
                                marginal="box", color_discrete_sequence=['#ff4b4b'])
        fig_dist.update_layout(template="plotly_dark", margin=dict(l=0,r=0,b=0,t=30))
        st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # --- Data Comparison ---
    st.markdown("### 🔄 So sánh Dữ liệu trước và sau Tiền xử lý")
    col_data1, col_data2 = st.columns([1, 1])
    with col_data1:
        st.markdown("**Bản ghi gốc (Raw Data)**")
        st.dataframe(df.head(10), use_container_width=True)
    with col_data2:
        st.markdown("**Bản ghi sau khi xử lý (Lag & Rolling Features)**")
        st.dataframe(df_final.head(10), use_container_width=True)

    st.divider()

    st.markdown("### 📊 Kiểm tra độ mượt của tín hiệu")
    p_id = st.selectbox("Chọn bệnh nhân phân tích", df_final['PatientID'].unique())
    sub_df = df_final[df_final['PatientID'] == p_id].sort_values('Timestamp')
    
    fig_smooth = go.Figure()
    fig_smooth.add_trace(go.Scatter(x=sub_df['Timestamp'], y=sub_df['Glucose'], name="Gốc (Raw)", line=dict(color='#ff8a86', width=1, dash='dot')))
    fig_smooth.add_trace(go.Scatter(x=sub_df['Timestamp'], y=sub_df['Glucose_Smooth'], name="Mượt (Smooth)", line=dict(color='#ff4b4b', width=3)))
    
    # Thêm ngưỡng y tế (Medical Thresholds)
    fig_smooth.add_hline(y=180, line_dash="dash", line_color="orange", annotation_text="Ngưỡng cao (Hyper)", annotation_position="top right")
    fig_smooth.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Ngưỡng thấp (Hypo)", annotation_position="bottom right")
    
    st.plotly_chart(fig_smooth, use_container_width=True)

    st.markdown("### 🌡️ Ma trận tương quan (Correlation Heatmap)")
    corr = df_final[features_pr + [target]].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", 
                         color_continuous_scale='RdBu_r',
                         title="Mối quan hệ giữa các biến số")
    fig_corr.update_layout(template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 2: CLUSTERING ---
with tab2:
    st.subheader("🧩 Phân nhóm bệnh nhân thông minh")
    cl_col1, cl_col2 = st.columns([1, 2])
    
    features_cl = ['Age', 'BMI', 'HBA1C', 'Glucose']
    X_cl = df_final[features_cl].drop_duplicates()
    cm = ClusteringModule(X_cl)

    with cl_col1:
        st.markdown("""
        Tại giai đoạn này, hệ thống sẽ tự động tìm kiếm các đặc điểm chung của bệnh nhân để phân loại vào các nhóm bệnh lý khác nhau.
        """)
        algo = st.selectbox("Thuật toán phân cụm", ["K-Means", "Hierarchical", "DBSCAN", "GMM", "Mean Shift"])
        n_clusters = st.slider("Số lượng cụm mục tiêu", 2, 6, 3)
        
        if algo == "K-Means": labels = cm.run_kmeans(n_clusters); 
        elif algo == "Hierarchical": labels = cm.run_hierarchical(n_clusters);
        elif algo == "DBSCAN": labels = cm.run_dbscan();
        elif algo == "GMM": labels = cm.run_gmm(n_clusters);
        else: labels = cm.run_meanshift();

    with cl_col2:
        pca_df = cm.get_pca_projection(labels)
        fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', 
                           symbol='Cluster', template='plotly_dark',
                           color_continuous_scale=px.colors.sequential.Reds,
                           title=f"Bản đồ phân cụm không gian 2D (Sử dụng {algo})")
        st.plotly_chart(fig_pca, use_container_width=True)

# --- TAB 3: PREDICTION ---
with tab3:
    st.subheader("📈 Dự báo đường huyết tương lai")
    
    # Global Configuration for Analysis
    target = 'Glucose'
    features_pr = ['Age', 'BMI', 'HBA1C', 'Glucose_Lag_1', 'Glucose_Lag_2', 'Hour', 'DayOfWeek']
    
    # Đảm bảo không còn giá trị NaN nào đi vào mô hình training
    df_model = df_final.copy()
    df_model[features_pr] = df_model[features_pr].ffill().bfill() # Điền khuyết các đặc trưng trễ (Lag)
    df_model = df_model.dropna(subset=features_pr + [target]) # Xóa nốt những gì không thể điền
    
    X = df_model[features_pr]
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pm = PredictionModule(X_train, y_train, X_test, y_test)
    
    with st.spinner("🧠 AI đang học dữ liệu..."):
        pm.run_logistic_baseline()
        pm.run_random_forest()
        pm.run_xgboost()
        pm.run_svm()
        pm.run_knn_regressor()
        metrics = pm.evaluate_all()

    # Metrics Display
    st.markdown("### 🏆 Bảng xếp hạng độ chính xác")
    
    # Chuẩn bị dữ liệu cho biểu đồ (Chuyển đổi từ dict sang DataFrame)
    perf_data = []
    for m_name, m_vals in metrics.items():
        perf_data.append({"Model": m_name, "R2": m_vals['R2'], "RMSE": m_vals['RMSE']})
    perf_df = pd.DataFrame(perf_data).sort_values("R2", ascending=False)

    # Biểu đồ Benchmarking
    fig_bench = px.bar(perf_df, x='Model', y='R2', color='R2', text_auto='.3f',
                       title="So sánh chỉ số R2 giữa các mô hình (Càng cao càng tốt)",
                       color_continuous_scale='RdBu_r')
    fig_bench.update_layout(template="plotly_dark", yaxis_range=[0, 1.1])
    st.plotly_chart(fig_bench, use_container_width=True)

    st.divider()
    
    pr_col1, pr_col2 = st.columns([2, 1])
    with pr_col1:
        model_choice = st.selectbox("Chọn mô hình để trực quan hóa dự báo", list(pm.models.keys()))
        y_pred = pm.models[model_choice].predict(X_test)
        res_df = pd.DataFrame({'Thực tế': y_test, 'Dự báo': y_pred}).reset_index(drop=True)
        
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(y=res_df['Thực tế'][:50], name="Thực tế", mode='lines+markers', line=dict(color='#888', width=1)))
        fig_res.add_trace(go.Scatter(y=res_df['Dự báo'][:50], name="AI Dự báo", mode='lines+markers', line=dict(color='#ff4b4b', width=3)))
        fig_res.update_layout(template="plotly_dark", title="So sánh Thực tế vs Dự báo (50 mẫu)", margin=dict(l=0,r=0,b=0,t=40))
        st.plotly_chart(fig_res, use_container_width=True)
    
    with pr_col2:
        st.markdown("### 📁 Kết xuất dữ liệu")
        st.dataframe(res_df.head(15), use_container_width=True)
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Tải báo cáo CSV", data=csv, file_name='ai_predictions.csv', use_container_width=True)

# --- TAB 4: EVALUATION & XAI ---
with tab4:
    st.subheader("🛡️ Phân tích rủi ro & Tính giải thích AI (XAI)")
    
    ev_col1, ev_col2 = st.columns(2)
    
    with ev_col1:
        st.markdown("### 🎯 Clarke Error Grid Analysis")
        st.write("Đánh giá mức độ an toàn lâm sàng của các dự báo AI.")
        fig_clarke = clarke_error_grid(y_test, y_pred, model_choice)
        st.pyplot(fig_clarke)
        
    with ev_col2:
        st.markdown("### 🤖 Giải thích SHAP Values")
        st.write("Xác định biến số nào đang gây tác động mạnh nhất đến đường huyết.")
        if "Forest" in model_choice or "XGBoost" in model_choice:
            fig_shap = explain_with_shap(pm.models[model_choice], X_train, X_test[:50], features_pr)
            st.pyplot(fig_shap)
        else:
            st.info("💡 Tính năng SHAP ưu tiên cho các mô hình Ensemble (Random Forest/XGBoost). Vui lòng chọn mô hình này ở Tab Dự báo.")

    with st.expander("🧪 Công cụ dự báo nhanh (What-if Analysis)"):
        st.write("Nhập thông số giả định để AI dự báo nồng độ đường huyết ngay lập tức:")
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            in_age = st.number_input("Tuổi", 1, 100, 50)
            in_bmi = st.number_input("Chỉ số BMI", 10.0, 50.0, 25.0)
        with wc2:
            in_glu = st.number_input("Đường huyết hiện tại", 40, 400, 120)
            in_hba1c = st.number_input("Chỉ số HBA1C", 4.0, 15.0, 6.5)
        with wc3:
            in_hour = st.slider("Giờ trong ngày", 0, 23, 12)
            
        if st.button("🚀 Chạy dự báo tức thì"):
            # Chuẩn bị dữ liệu input
            input_data = pd.DataFrame([[in_age, in_bmi, in_hba1c, in_glu, in_glu, in_hour, 0]], 
                                     columns=features_pr)
            
            # Sử dụng mô hình tốt nhất (thường là XGBoost hoặc Random Forest)
            best_model_name = metrics.iloc[0]['Model']
            prediction = pm.models[best_model_name].predict(input_data)[0]
            
            st.code(f"Mô hình phối hợp tốt nhất ({best_model_name}) dự báo chỉ số tiếp theo là: {prediction:.2f} mg/dL", language="python")
            
            if prediction > 180: st.warning("⚠️ Cảnh báo: Nguy cơ đường huyết cao!")
            elif prediction < 70: st.error("🚨 Cảnh báo: Nguy cơ tụt đường huyết!")
            else: st.success("✅ Chỉ số dự báo nằm trong ngưỡng an toàn.")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Đồ án Khai thác dữ liệu - Nhóm 10 | 2025</p>", unsafe_allow_html=True)
