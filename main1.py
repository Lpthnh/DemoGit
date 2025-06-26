import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from mlxtend.frequent_patterns import apriori, association_rules

# Tắt cảnh báo liên quan đến downcasting
pd.set_option('future.no_silent_downcasting', True)

# Cấu hình trang
st.set_page_config(page_title="Ứng Dụng Dự Đoán Bệnh", layout="wide")

# CSS để đóng khung phần "Gợi ý hành động" và bảng luật kết hợp
st.markdown("""
<style>
.suggestion-box {
    border: 2px solid #28a745;
    border-radius: 10px;
    padding: 15px;
    background-color: #f8fff8;
    margin-bottom: 20px;
}
.suggestion-box h4 {
    color: #28a745;
    margin-bottom: 10px;
}
.suggestion-box p {
    margin: 5px 0;
    font-size: 16px;
}
.stDataFrame table {
    width: 100%;
    border-collapse: collapse;
}
.stDataFrame th, .stDataFrame td {
    padding: 8px;
    text-align: center;
    border: 1px solid #ddd;
}
.stDataFrame th {
    background-color: #f2f2f2;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Tiêu đề và mô tả
st.title("🩺 Dự Đoán Bệnh Từ Triệu Chứng")
st.markdown("""
Ứng dụng này sử dụng mô hình máy học và phân cụm để dự đoán bệnh dựa trên triệu chứng và bệnh tiền sử.
""")

# Hàm tìm bệnh gần giống nhất
def find_closest_disease(disease, reference_list):
    disease = disease.lower()
    for ref_disease in reference_list:
        ref_disease_lower = ref_disease.lower()
        if disease in ref_disease_lower or ref_disease_lower in disease:
            return ref_disease
    return None

# Hàm gợi ý bệnh viện và hành động
def suggest_hospitals_and_actions(disease, hospital_df, action_dict, region):
    matches = hospital_df[(hospital_df["Bệnh"] == disease) & (hospital_df["Khu vực"] == region)]
    if matches.empty:
        closest_disease = find_closest_disease(disease, hospital_df["Bệnh"].unique())
        if closest_disease:
            matches = hospital_df[(hospital_df["Bệnh"] == closest_disease) & (hospital_df["Khu vực"] == region)]
            disease = closest_disease
    hospitals = matches["Bệnh viện"].dropna().tolist()
    
    if not hospitals:
        hospitals = ["Không có bệnh viện gợi ý trong khu vực này, vui lòng đến cơ sở y tế gần nhất."]
    
    if disease in action_dict:
        action_info = action_dict[disease]
    else:
        closest_disease = find_closest_disease(disease, action_dict.keys())
        if closest_disease:
            action_info = action_dict[closest_disease]
        else:
            action_info = {"severity": "Chưa có thông tin", "action": "Cần tham khảo ý kiến bác sĩ để đánh giá mức độ nghiêm trọng."}
    
    return hospitals, action_info["severity"], action_info["action"]

# Khởi tạo session_state
if 'le_disease' not in st.session_state:
    st.session_state.le_disease = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'symptom_columns' not in st.session_state:
    st.session_state.symptom_columns = None
if 'pre_existing_conditions' not in st.session_state:
    st.session_state.pre_existing_conditions = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'le_gender' not in st.session_state:
    st.session_state.le_gender = None
if 'reset_symptom_form' not in st.session_state:
    st.session_state.reset_symptom_form = False
if 'reset_cluster_form' not in st.session_state:
    st.session_state.reset_cluster_form = False
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'cluster_diseases' not in st.session_state:
    st.session_state.cluster_diseases = None
if 'cluster_rules' not in st.session_state:
    st.session_state.cluster_rules = None

# Hard-code gợi ý hành động và mức độ nghiêm trọng
action_dict = {
    "Covid-19": {"severity": "Cao", "action": "Cách ly tại nhà, liên hệ y tế ngay nếu khó thở."},
    "Cúm": {"severity": "Trung bình", "action": "Nghỉ ngơi, uống nhiều nước, dùng thuốc hạ sốt nếu cần."},
    "Suy nhược cơ thể": {"severity": "Thấp", "action": "Tăng cường dinh dưỡng, nghỉ ngơi hợp lý."},
    "Viêm phổi": {"severity": "Cao", "action": "Đi khám ngay tại bệnh viện, tránh tự điều trị."},
    "Viêm họng": {"severity": "Thấp", "action": "Súc miệng nước muối, nghỉ ngơi, uống nhiều nước."},
    "Đau dạ dày": {"severity": "Trung bình", "action": "Ăn nhẹ, tránh đồ chua cay, tham khảo ý kiến bác sĩ nếu đau kéo dài."},
    "Sốt xuất huyết": {"severity": "Cao", "action": "Theo dõi sát sao, đi khám ngay nếu có dấu hiệu xuất huyết."},
    "Tiểu đường": {"severity": "Trung bình", "action": "Kiểm tra đường huyết thường xuyên, tuân thủ chế độ ăn uống."},
    "Cao huyết áp": {"severity": "Trung bình", "action": "Theo dõi huyết áp, hạn chế muối, tham khảo ý kiến bác sĩ."},
    "Viêm xoang": {"severity": "Trung bình", "action": "Giữ mũi sạch, tránh khói bụi, tham khảo ý kiến bác sĩ nếu kéo dài."},
    "Hen suyễn": {"severity": "Cao", "action": "Sử dụng thuốc cắt cơn nếu có, đi khám ngay nếu khó thở nặng."},
    "Viêm amidan": {"severity": "Thấp", "action": "Súc miệng nước muối, nghỉ ngơi, dùng thuốc kháng sinh nếu bác sĩ kê đơn."},
    "Trầm cảm": {"severity": "Trung bình", "action": "Tham khảo ý kiến bác sĩ tâm lý, duy trì lối sống lành mạnh."},
    "Viêm phế quản": {"severity": "Trung bình", "action": "Nghỉ ngơi, uống nhiều nước, đi khám nếu ho kéo dài."},
    "Rối loạn tiêu hóa": {"severity": "Thấp", "action": "Ăn uống điều độ, tránh đồ ăn khó tiêu, tham khảo ý kiến bác sĩ nếu kéo dài."},
    "Viêm khớp": {"severity": "Trung bình", "action": "Giữ ấm khớp, tham khảo ý kiến bác sĩ để dùng thuốc giảm đau."},
    "Viêm ruột": {"severity": "Trung bình", "action": "Ăn uống nhẹ nhàng, tham khảo ý kiến bác sĩ nếu đau bụng kéo dài."},
    "Viêm tụy": {"severity": "Cao", "action": "Đi khám ngay tại bệnh viện, tránh tự điều trị."},
    "Viêm gan": {"severity": "Cao", "action": "Đi khám ngay, tuân thủ chế độ ăn uống theo chỉ dẫn bác sĩ."},
    "Chàm": {"severity": "Thấp", "action": "Giữ da sạch, dùng kem dưỡng ẩm, tham khảo ý kiến bác sĩ nếu nặng."},
    "Viêm tai giữa": {"severity": "Trung bình", "action": "Đi khám tai mũi họng, tránh để nước vào tai."},
    "Viêm da": {"severity": "Thấp", "action": "Giữ da sạch, tránh chất kích ứng, tham khảo ý kiến bác sĩ nếu nặng."},
    "Viêm thanh quản": {"severity": "Thấp", "action": "Nghỉ ngơi giọng nói, uống nhiều nước, tham khảo ý kiến bác sĩ nếu kéo dài."},
    "Dị ứng": {"severity": "Thấp", "action": "Tránh tác nhân gây dị ứng, tham khảo ý kiến bác sĩ nếu nặng."},
    "Táo bón mãn tính": {"severity": "Thấp", "action": "Tăng cường chất xơ, uống nhiều nước, tham khảo ý kiến bác sĩ nếu kéo dài."},
    "Viêm dạ dày": {"severity": "Trung bình", "action": "Ăn nhẹ, tránh đồ chua cay, tham khảo ý kiến bác sĩ nếu đau kéo dài."},
    "Đột quỵ nhẹ": {"severity": "Cao", "action": "Đi khám ngay tại bệnh viện, không tự điều trị."},
    "Viêm màng não": {"severity": "Cao", "action": "Đi khám ngay tại bệnh viện, không tự điều trị."},
    "Mất ngủ": {"severity": "Thấp", "action": "Duy trì thói quen ngủ đều đặn, tham khảo ý kiến bác sĩ nếu kéo dài."},
    "Rối loạn lo âu": {"severity": "Trung bình", "action": "Tham khảo ý kiến bác sĩ tâm lý, thực hành thư giãn."},
    "Zona thần kinh": {"severity": "Trung bình", "action": "Đi khám da liễu, dùng thuốc theo chỉ định bác sĩ."},
    "Loét dạ dày": {"severity": "Trung bình", "action": "Ăn uống điều độ, tránh căng thẳng, tham khảo ý kiến bác sĩ."},
    "Viêm cơ tim": {"severity": "Cao", "action": "Đi khám ngay tại bệnh viện, không tự điều trị."},
    "Viêm đại tràng": {"severity": "Trung bình", "action": "Ăn uống lành mạnh, tham khảo ý kiến bác sĩ nếu triệu chứng nặng."}
}

# Khởi tạo df và hospital_df mặc định
df = pd.read_excel("data_benh.xlsx")
hospital_df = pd.read_excel("benhviengoiy.xlsx")

# Xử lý cột của df và khởi tạo symptom_columns, pre_existing_conditions
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(r"\.\d+$", "", regex=True)
df = df.loc[:, ~df.columns.duplicated()]
symptom_columns = [col for col in df.columns if col.startswith("Triệu chứng")]
pre_existing_conditions = [col for col in df.columns if col.startswith("Tiền sử")]

st.session_state.symptom_columns = symptom_columns
st.session_state.pre_existing_conditions = pre_existing_conditions

# Sidebar cho tải file
with st.sidebar:
    st.header("📤 Tải Dữ Liệu")
    uploaded_files = st.file_uploader(
        "Tải 2 file: data_benh.xlsx và benhviengoiy.xlsx", 
        type=["xlsx"], 
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) == 2:
        df_file = None
        hospital_file = None
        for file in uploaded_files:
            if "data_benh" in file.name.lower():
                df_file = file
            elif "benhviengoiy" in file.name.lower():
                hospital_file = file
        
        if df_file and hospital_file:
            df = pd.read_excel(df_file)
            hospital_df = pd.read_excel(hospital_file)
            st.success("Tải 2 file thành công!")
            
            df.columns = df.columns.str.strip()
            df.columns = df.columns.str.replace(r"\.\d+$", "", regex=True)
            df = df.loc[:, ~df.columns.duplicated()]
            symptom_columns = [col for col in df.columns if col.startswith("Triệu chứng")]
            pre_existing_conditions = [col for col in df.columns if col.startswith("Tiền sử")]

            st.session_state.symptom_columns = symptom_columns
            st.session_state.pre_existing_conditions = pre_existing_conditions

# Tiền xử lý dữ liệu toàn cục
df['Giới tính'] = df['Giới tính'].astype(str).str.strip()
gender_mapping = {'0': 'Nam', '1': 'Nữ'}
valid_values = ['0', '1', 'Nam', 'Nữ']
if not df['Giới tính'].isin(valid_values).all():
    invalid_genders = df[~df['Giới tính'].isin(valid_values)]['Giới tính'].unique()
    st.error(f"Cột 'Giới tính' chứa các giá trị không hợp lệ: {invalid_genders}. Chỉ chấp nhận '0', '1', 'Nam', hoặc 'Nữ'.")
    st.stop()
df['Giới tính'] = df['Giới tính'].replace(gender_mapping)

le_gender = LabelEncoder()
df['Giới tính'] = le_gender.fit_transform(df['Giới tính'])
st.session_state.le_gender = le_gender

le_disease = LabelEncoder()
df['Bệnh hiện tại'] = le_disease.fit_transform(df['Bệnh hiện tại'].astype(str))
st.session_state.le_disease = le_disease

base_cols = ['Tuổi', 'Giới tính', 'Bệnh hiện tại']
binary_cols = [col for col in df.columns if df[col].dropna().isin([0, 1]).all() and col not in base_cols]

known_symptoms = [
    "Sốt", "Ho", "Đau họng", "Khó thở", "Chảy nước mũi", "Nghẹt mũi", "Đau đầu",
    "Đau ngực", "Mệt mỏi", "Buồn nôn", "Nôn", "Tiêu chảy", "Táo bón", "Đau bụng",
    "Ớn lạnh", "Phát ban", "Ngứa da", "Chóng mặt", "Mất vị giác", "Mất khứu giác",
    "Lo âu", "Khó ngủ", "Khó nuốt", "Sụt cân", "Đổ mồ hôi nhiều"
]

symptom_columns = [col for col in binary_cols if col in known_symptoms]
pre_existing_conditions = [col for col in binary_cols if col not in symptom_columns]
feature_columns = list(dict.fromkeys(['Tuổi', 'Giới tính'] + symptom_columns + pre_existing_conditions))
st.session_state.feature_columns = feature_columns

# Tạo các tab
tab1, tab2, tab3, tab4 = st.tabs(["📊 Phân tích dữ liệu", "🤖 Dự đoán bệnh", "🔍 Phân cụm", "💬 Chatbox"])
# Tab 1: Xem dữ liệu
with tab1:
    st.header("Xem Dữ Liệu")
    st.subheader("Phân Bố Các Bệnh")
    disease_counts = df['Bệnh hiện tại'].value_counts().reset_index()
    disease_counts.columns = ["Bệnh", "Số lượng"]
    st.dataframe(disease_counts, use_container_width=True)
    fig = px.bar(disease_counts, x="Bệnh", y="Số lượng", 
                 text="Số lượng", 
                 title="Phân Bố Các Bệnh Trong Dữ Liệu",
                 color="Bệnh",
                 height=400)
    fig.update_traces(texttemplate='%{text}', textposition='auto')
    st.plotly_chart(fig, use_container_width=True)

    # Phân tích tương quan
    st.subheader("Phân Tích Tương Quan Giữa Triệu Chứng/Bệnh Tiền Sử và Bệnh")
    correlation_data = df[feature_columns + ['Bệnh hiện tại']].copy()
    correlation_data = pd.get_dummies(correlation_data, columns=['Bệnh hiện tại'])
    disease_cols = [col for col in correlation_data.columns if col.startswith('Bệnh hiện tại_')]
    correlation_results = {}
    for feature in feature_columns:
        correlations = {}
        for disease_col in disease_cols:
            disease_name = le_disease.inverse_transform([int(disease_col.split('_')[-1])])[0]
            corr = correlation_data[feature].corr(correlation_data[disease_col], method='pearson')
            if not np.isnan(corr) and abs(corr) > 0.1:
                correlations[disease_name] = corr
        if correlations:
            correlation_results[feature] = correlations

    # Lưu kết quả tương quan vào session_state
    st.session_state.correlation_results = correlation_results

    # Hiển thị kết quả tương quan
    for feature, correlations in correlation_results.items():
        st.write(f"**{feature}**")
        for disease, corr in correlations.items():
            st.write(f"- {disease}: {corr:.3f}")

    st.subheader("Dữ liệu bệnh")
    rows_per_page = st.selectbox("Số dòng mỗi trang", options=[10, 20, 50, 100], index=0)
    total_rows = len(df)
    total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page else 0)
    page = st.number_input("Trang", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    page_data = df.iloc[start_idx:end_idx]
    st.dataframe(page_data, height=300, use_container_width=True)
    st.markdown(f"Đang hiển thị trang {page}/{total_pages} ({start_idx + 1} - {min(end_idx, total_rows)} trên {total_rows} dòng)")

    st.subheader("Dữ liệu bệnh viện")
    st.dataframe(hospital_df, height=300, use_container_width=True)

# Tab 2: Dự đoán mô hình
with tab2:
    st.header("Dự đoán mô hình")

    # Tiền xử lý
    X = df[feature_columns]
    y = df['Bệnh hiện tại']
    
    use_smote = st.checkbox("Sử dụng SMOTE để cân bằng dữ liệu", value=True)

    if use_smote:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Huấn luyện mô hình
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
    }
    model_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        model_results[name] = {
            "model": model,
            "accuracy": accuracy,
            "report": report
        }
    st.session_state.model_results = model_results

    st.subheader("Tổng Hợp Độ Chính Xác")
    accuracy_df = pd.DataFrame({
        "Mô Hình": [name for name in model_results.keys()],
        "Độ Chính Xác": [result["accuracy"] for result in model_results.values()]
    })
    st.dataframe(accuracy_df, use_container_width=True)

    fig = px.bar(accuracy_df, x="Mô Hình", y="Độ Chính Xác", 
                 text="Độ Chính Xác", 
                 title="So Sánh Độ Chính Xác Của Các Mô Hình",
                 color="Mô Hình",
                 height=400)
    fig.update_traces(texttemplate='%{text:.4f}', textposition='auto')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Chi Tiết Báo Cáo Phân Loại")
    cols = st.columns(2)
    for idx, (name, result) in enumerate(model_results.items()):
        with cols[idx]:
            with st.expander(f"Báo Cáo Phân Loại: {name}"):
                report_df = pd.DataFrame({
                    "Lớp": [str(k) for k in result["report"].keys() if k not in ['accuracy', 'macro avg', 'weighted avg']],
                    "Precision": [result["report"][k]["precision"] for k in result["report"].keys() if k not in ['accuracy', 'macro avg', 'weighted avg']],
                    "Recall": [result["report"][k]["recall"] for k in result["report"].keys() if k not in ['accuracy', 'macro avg', 'weighted avg']],
                    "F1-Score": [result["report"][k]["f1-score"] for k in result["report"].keys() if k not in ['accuracy', 'macro avg', 'weighted avg']],
                    "Support": [result["report"][k]["support"] for k in result["report"].keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
                })
                st.dataframe(report_df, use_container_width=True)
                st.write(f"**Độ chính xác tổng thể**: {result['accuracy']:.4f}")

    st.subheader("🔍 Dự Đoán Bệnh Từ Triệu Chứng")
    st.markdown("Nhập thông tin để dự đoán bệnh.")
    with st.form("symptom_form"):
        age = st.number_input("Tuổi", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Giới tính", options=["Nam", "Nữ"])
        region = st.selectbox("Khu vực", options=["Miền Bắc", "Miền Trung", "Miền Nam"])
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Chọn Triệu Chứng**")
            with st.container(height=300):
                symptoms = {}
                for col in symptom_columns:
                    default_value = False if st.session_state.reset_symptom_form else st.session_state.get(f"symptom_{col}", False)
                    symptoms[col] = st.checkbox(col, value=default_value, key=f"symptom_{col}")
        with col2:
            st.markdown("**Chọn Bệnh Tiền Sử**")
            with st.container(height=300):
                conditions = {}
                for condition in pre_existing_conditions:
                    default_value = False if st.session_state.reset_symptom_form else st.session_state.get(f"condition_{condition}", False)
                    conditions[condition] = st.checkbox(condition, value=default_value, key=f"condition_{condition}")
        col_submit, col_reset = st.columns([1, 1])
        with col_submit:
            submitted = st.form_submit_button("Dự đoán")
        with col_reset:
            reset = st.form_submit_button("Reset Triệu Chứng")

        if reset:
            st.session_state.reset_symptom_form = True
            st.rerun()

        if st.session_state.reset_symptom_form:
            st.session_state.reset_symptom_form = False

        if submitted:
            if not st.session_state.model_results or not st.session_state.le_disease or not st.session_state.le_gender:
                st.error("Vui lòng huấn luyện mô hình trước!")
                st.stop()
            try:
                input_data = pd.DataFrame(columns=st.session_state.feature_columns)
                input_data.loc[0, 'Tuổi'] = age
                gender_normalized = gender.strip().capitalize()
                input_data.loc[0, 'Giới tính'] = st.session_state.le_gender.transform([gender_normalized])[0]
                for col in symptom_columns:
                    input_data.loc[0, col] = 1 if symptoms[col] else 0
                for condition in pre_existing_conditions:
                    input_data.loc[0, condition] = 1 if conditions[condition] else 0
                input_data = input_data.fillna(0).infer_objects(copy=False).astype(int)

                num_symptoms = sum(1 for col, val in symptoms.items() if val)
                if num_symptoms < 2:
                    st.warning("⚠️ Vui lòng chọn ít nhất 2 triệu chứng để dự đoán chính xác hơn.")
                    st.stop()

                light_diseases = ["Cúm", "Viêm họng", "Viêm amidan"]
                severe_symptoms = ["Khó thở", "Đau ngực", "Mất vị giác", "Mất khứu giác"]
                has_severe_symptom = any(symptoms[symptom] for symptom in severe_symptoms)

                st.write("#### Kết Quả Dự Đoán")
                all_diseases = []
                model_predictions = {}
                top_disease_for_hospital = None
                for name, result in st.session_state.model_results.items():
                    model = result["model"]
                    model_predictions[name] = []
                    st.write(f"**{name}**:")
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(input_data)[0]
                        # Điều chỉnh xác suất dựa trên tương quan
                        adjusted_probs = probs.copy()
                        correlation_adjustments = []
                        if 'correlation_results' in st.session_state:
                            for feature in feature_columns:
                                if (feature in symptoms and symptoms[feature]) or (feature in conditions and conditions[feature]):
                                    if feature in st.session_state.correlation_results:
                                        for disease, corr in st.session_state.correlation_results[feature].items():
                                            if disease in st.session_state.le_disease.classes_:
                                                disease_idx = np.where(st.session_state.le_disease.classes_ == disease)[0]
                                                if len(disease_idx) > 0:
                                                    old_prob = adjusted_probs[disease_idx[0]]
                                                    new_prob = old_prob + (old_prob * corr * 0.5)
                                                    adjusted_probs[disease_idx[0]] = new_prob
                                                    correlation_adjustments.append({
                                                        "Feature": feature,
                                                        "Bệnh": disease,
                                                        "Tương quan": f"{corr:.3f}",
                                                        "Tăng xác suất": f"{disease}: từ {old_prob:.2%} lên {new_prob:.2%}"
                                                    })
                        adjusted_probs = adjusted_probs / adjusted_probs.sum()

                        # Điều chỉnh xác suất nếu số lượng triệu chứng <= 4 và không có triệu chứng đặc trưng
                        if num_symptoms <= 4 and not has_severe_symptom:
                            for idx, disease in enumerate(st.session_state.le_disease.classes_):
                                if disease in light_diseases:
                                    adjusted_probs[idx] *= 3.0
                            adjusted_probs = adjusted_probs / adjusted_probs.sum()

                        top_indices = np.argsort(adjusted_probs)[-3:][::-1]
                        model_diseases = [(st.session_state.le_disease.inverse_transform([i])[0], adjusted_probs[i]) for i in top_indices]
                        for disease, prob in model_diseases:
                            st.write(f"- {disease}: {prob:.2%}")
                            model_predictions[name].append((disease, prob))
                        st.success(f"✅ Dự đoán chính: **{model_diseases[0][0]}** với xác suất {model_diseases[0][1]:.2%}")
                        all_diseases.extend([(disease, prob, name) for disease, prob in model_diseases])
                        if name == "Random Forest":
                            top_disease_for_hospital = model_diseases[0][0]
                    else:
                        prediction = model.predict(input_data)
                        predicted_disease = st.session_state.le_disease.inverse_transform(prediction)[0]
                        st.write(f"{predicted_disease} (mô hình không hỗ trợ xác suất)")
                        model_predictions[name].append((predicted_disease, 0.0))
                        all_diseases.append((predicted_disease, 0.0, name))

                rf_diseases = sorted(model_predictions["Random Forest"], key=lambda x: x[1], reverse=True)[:2]
                final_diseases = [(disease, prob, "Random Forest") for disease, prob in rf_diseases]
                rf_diseases_set = set(disease for disease, _, _ in final_diseases)

                other_diseases = [(disease, prob, model) for disease, prob, model in all_diseases if model == "Decision Tree" and disease not in rf_diseases_set]
                if other_diseases:
                    other_diseases = sorted(other_diseases, key=lambda x: x[1], reverse=True)
                    final_diseases.append(other_diseases[0])
                else:
                    if len(model_predictions["Random Forest"]) > 2:
                        third_rf_disease = sorted(model_predictions["Random Forest"], key=lambda x: x[1], reverse=True)[2]
                        if third_rf_disease[0] not in rf_diseases_set:
                            final_diseases.append((third_rf_disease[0], third_rf_disease[1], "Random Forest"))

                final_diseases = sorted(final_diseases, key=lambda x: x[1], reverse=True)[:3]

                colors = ["#28a745", "#fd7e14", "#6f42c1"]

                st.markdown("### 🩺 **Kết Quả Tư Vấn Cá Nhân Hóa**")
                st.markdown(f"""
                **Thông tin người dùng:**
                - Tuổi: {age}
                - Giới tính: {gender}
                - Khu vực: {region}
                - Triệu chứng: {', '.join([col for col, val in symptoms.items() if val]) or 'Không có'}
                - Bệnh tiền sử: {', '.join([cond for cond, val in conditions.items() if val]) or 'Không có'}

                **Mô hình dự đoán:**
                Kết quả dự đoán chính là:
                """)
                for idx, (disease, prob, model) in enumerate(final_diseases):
                    st.markdown(f"- <span style='color:{colors[idx]}'>{disease}</span>", unsafe_allow_html=True)
                st.markdown("- Dựa trên tổ hợp các triệu chứng và tiền sử bệnh liên quan")

                st.markdown("**Gợi ý hành động:**")
                for disease, prob, model in final_diseases:
                    hospitals, severity, action = suggest_hospitals_and_actions(disease, hospital_df, action_dict, region)
                    with st.container():
                        st.markdown(f"""
                        <div class="suggestion-box">
                            <h4>Bệnh: {disease} ({prob:.2%})</h4>
                            <p><strong>Mức độ nghiêm trọng:</strong> {severity}</p>
                            <p><strong>Hành động:</strong> {action}</p>
                            <p><strong>Bệnh viện gợi ý:</strong> {', '.join(hospitals)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                st.info("⚠️ Đây chỉ là gợi ý từ mô hình học máy. Hãy đến cơ sở y tế để được khám và chẩn đoán chính xác.")

            except ValueError as e:
                st.error(f"Lỗi khi dự đoán: {str(e)}. Vui lòng kiểm tra giá trị đầu vào.")
                st.stop()

# Tab 3: Phân cụm
with tab3:
    st.header("🔍 Dự Đoán Bệnh Bằng Phân Cụm và Luật Kết Hợp")

    X = df[feature_columns]
    y = df['Bệnh hiện tại']

    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
        df_resampled['Bệnh hiện tại'] = y_resampled
    except ValueError:
        X_resampled, y_resampled = X, y
        df_resampled = df.copy()

    num_clusters = st.slider("Số lượng cụm", min_value=3, max_value=15, value=7)

    st.subheader("⚙️ Cấu Hình Apriori")
    min_support = st.slider("Min Support", min_value=0.01, max_value=0.5, value=0.05, step=0.01, format="%.2f")
    min_confidence = st.slider("Min Confidence", min_value=0.1, max_value=1.0, value=0.6, step=0.1, format="%.1f")
    min_lift = st.slider("Min Lift", min_value=1.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_resampled)
    df_resampled['Cụm'] = cluster_labels
    st.session_state.clusters = cluster_labels

    st.subheader("📜 Tìm Luật Kết Hợp Trong Từng Cụm")
    cluster_rules = {}
    for cluster in range(num_clusters):
        cluster_data = df_resampled[df_resampled['Cụm'] == cluster]
        if len(cluster_data) < 2:
            cluster_rules[cluster] = []
            continue

        transactions = []
        for idx, row in cluster_data.iterrows():
            transaction = []
            for col in symptom_columns:
                if row[col] == 1:
                    transaction.append(col)
            for col in pre_existing_conditions:
                if row[col] == 1:
                    transaction.append(col)
            disease = st.session_state.le_disease.inverse_transform([row['Bệnh hiện tại']])[0]
            transaction.append(f"Bệnh_{disease}")
            transactions.append(transaction)

        if not transactions:
            cluster_rules[cluster] = []
            continue
        transaction_binary = pd.DataFrame(0, index=range(len(transactions)), columns=np.unique(np.concatenate(transactions)))
        for idx, transaction in enumerate(transactions):
            for item in transaction:
                transaction_binary.loc[idx, item] = 1

        frequent_itemsets_df = apriori(transaction_binary, min_support=min_support, use_colnames=True)
        if frequent_itemsets_df.empty:
            cluster_rules[cluster] = []
            continue

        rules = association_rules(frequent_itemsets_df, metric="confidence", min_threshold=min_confidence)
        rules = rules[rules['lift'] >= min_lift]
        cluster_rules[cluster] = rules

    st.session_state.cluster_rules = cluster_rules

    st.subheader("📊 Thông Tin Về Các Cụm")
    for cluster in range(num_clusters):
        cluster_data = df_resampled[df_resampled['Cụm'] == cluster]
        with st.expander(f"Cụm {cluster} ({len(cluster_data)} bệnh nhân)"):
            symptom_means = cluster_data[symptom_columns].mean().sort_values(ascending=False)
            top_symptoms = symptom_means.head(2)
            symptom_df = pd.DataFrame({
                "Triệu chứng": top_symptoms.index,
                "Tỷ lệ": [f"{value:.2%}" for value in top_symptoms.values]
            })
            st.write("**Triệu chứng phổ biến nhất:**")
            st.dataframe(symptom_df, use_container_width=True, hide_index=True)

            condition_means = cluster_data[pre_existing_conditions].mean().sort_values(ascending=False)
            top_conditions = condition_means.head(2)
            condition_df = pd.DataFrame({
                "Bệnh tiền sử": top_conditions.index,
                "Tỷ lệ": [f"{value:.2%}" for value in top_conditions.values]
            })
            st.write("**Bệnh tiền sử phổ biến nhất:**")
            st.dataframe(condition_df, use_container_width=True, hide_index=True)

            disease_counts = cluster_data['Bệnh hiện tại'].value_counts().head(2)
            disease_df = pd.DataFrame({
                "Bệnh": [st.session_state.le_disease.inverse_transform([idx])[0] for idx in disease_counts.index],
                "Số bệnh nhân": disease_counts.values,
                "Tỷ lệ": [f"{count/len(cluster_data):.2%}" for count in disease_counts.values]
            })
            st.write("**Bệnh phổ biến nhất:**")
            st.dataframe(disease_df, use_container_width=True, hide_index=True)

            rules = cluster_rules.get(cluster, [])
            if not rules.empty:
                st.write("**Luật kết hợp phổ biến nhất:**")
                rule_display = []
                for idx, row in rules.head(3).iterrows():
                    antecedents = list(row['antecedents'])
                    consequents = list(row['consequents'])
                    confidence = row['confidence']
                    lift = row['lift']
                    rule_display.append({
                        "Điều kiện": ", ".join(antecedents),
                        "Kết luận": ", ".join(consequents),
                        "Độ tin cậy": f"{confidence:.2%}",
                        "Độ nâng": f"{lift:.2f}"
                    })
                st.dataframe(pd.DataFrame(rule_display), use_container_width=True, hide_index=True)
            else:
                st.write("Không tìm thấy luật kết hợp nào trong cụm này.")

    st.subheader("🔍 Dự Đoán Bệnh Từ Cụm")
    with st.form("cluster_prediction_form"):
        age = st.number_input("Tuổi", min_value=0, max_value=120, value=30, key="cluster_age")
        gender = st.selectbox("Giới tính", options=["Nam", "Nữ"], key="cluster_gender")
        region = st.selectbox("Khu vực", options=["Miền Bắc", "Miền Trung", "Miền Nam"], key="cluster_region")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Chọn Triệu Chứng**")
            with st.container(height=300):
                cluster_symptoms = {}
                for col in symptom_columns:
                    default_value = False if st.session_state.reset_cluster_form else st.session_state.get(f"cluster_symptom_{col}", False)
                    cluster_symptoms[col] = st.checkbox(col, value=default_value, key=f"cluster_symptom_{col}")
        with col2:
            st.markdown("**Chọn Bệnh Tiền Sử**")
            with st.container(height=300):
                cluster_conditions = {}
                for condition in pre_existing_conditions:
                    default_value = False if st.session_state.reset_cluster_form else st.session_state.get(f"cluster_condition_{condition}", False)
                    cluster_conditions[condition] = st.checkbox(condition, value=default_value, key=f"cluster_condition_{condition}")
        
        col_submit, col_reset = st.columns([1, 1])
        with col_submit:
            submitted = st.form_submit_button("🔍 Dự đoán từ cụm")
        with col_reset:
            reset = st.form_submit_button("Reset Triệu Chứng")
        
        if reset:
            st.session_state.reset_cluster_form = True
            st.rerun()
        
        if st.session_state.reset_cluster_form:
            st.session_state.reset_cluster_form = False
        
        if submitted:
            input_data = pd.DataFrame(columns=feature_columns)
            input_data.loc[0, 'Tuổi'] = age
            gender_normalized = gender.strip().capitalize()
            input_data.loc[0, 'Giới tính'] = st.session_state.le_gender.transform([gender_normalized])[0]
            for col in symptom_columns:
                input_data.loc[0, col] = 1 if cluster_symptoms[col] else 0
            for condition in pre_existing_conditions:
                input_data.loc[0, condition] = 1 if cluster_conditions[condition] else 0
            input_data = input_data.fillna(0).infer_objects(copy=False).astype(int)

            num_symptoms = sum(1 for col, val in cluster_symptoms.items() if val)
            if num_symptoms < 3:
                st.warning("⚠️ Vui lòng chọn ít nhất 3 triệu chứng để tăng khả năng tìm luật kết hợp.")
                st.stop()

            cluster_assignment = kmeans.predict(input_data)[0]
            st.write(f"Triệu chứng đầu vào thuộc **Cụm {cluster_assignment}**")

            cluster_data = df_resampled[df_resampled['Cụm'] == cluster_assignment]
            if len(cluster_data) < 2:
                st.warning("⚠️ Cụm này có quá ít dữ liệu để dự đoán chính xác. Vui lòng thử với số lượng cụm khác.")
                st.stop()
            disease_counts = cluster_data['Bệnh hiện tại'].value_counts()
            st.write("**Phân bố bệnh trong cụm:**")
            disease_distribution = pd.DataFrame({
                "Bệnh": [st.session_state.le_disease.inverse_transform([idx])[0] for idx in disease_counts.index],
                "Số bệnh nhân": disease_counts.values,
                "Tỷ lệ": [f"{count/len(cluster_data):.2%}" for count in disease_counts.values]
            })
            st.dataframe(disease_distribution, use_container_width=True, hide_index=True)

            X_cluster = cluster_data[feature_columns]
            y_cluster = cluster_data['Bệnh hiện tại']

            lr = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial', class_weight='balanced')
            lr.fit(X_cluster, y_cluster)

            probs = lr.predict_proba(input_data)[0]

            matched_rules = []
            if st.session_state.cluster_rules and cluster_assignment in st.session_state.cluster_rules:
                rules = st.session_state.cluster_rules[cluster_assignment]
                if not rules.empty:
                    input_items = [col for col, val in cluster_symptoms.items() if val]
                    input_items += [col for col, val in cluster_conditions.items() if val]
                    input_set = set(input_items)
                    
                    for idx, row in rules.iterrows():
                        antecedents = set(row['antecedents'])
                        consequents = set(row['consequents'])
                        confidence = row['confidence']
                        matched_items = len(antecedents.intersection(input_set))
                        match_ratio = matched_items / len(antecedents) if len(antecedents) > 0 else 0
                        if match_ratio >= 0.5:
                            for consequent in consequents:
                                if consequent.startswith("Bệnh_"):
                                    disease = consequent.replace("Bệnh_", "")
                                    matched_rules.append({
                                        "disease": disease,
                                        "confidence": confidence * match_ratio,
                                        "antecedents": ", ".join(antecedents),
                                        "match_ratio": match_ratio
                                    })

            if matched_rules:
                st.write("**Luật kết hợp khớp với triệu chứng đầu vào (tham khảo):**")
                rule_display = []
                for rule in matched_rules:
                    disease = rule['disease']
                    confidence = rule['confidence']
                    antecedents = rule['antecedents']
                    match_ratio = rule['match_ratio']
                    rule_display.append({
                        "Điều kiện": antecedents,
                        "Kết luận": disease,
                        "Độ tin cậy": f"{confidence:.2%}",
                        "Tỷ lệ khớp": f"{match_ratio:.0%}"
                    })
                st.dataframe(pd.DataFrame(rule_display), use_container_width=True, hide_index=True)

            adjusted_probs = probs + 1e-3
            adjusted_probs = adjusted_probs / adjusted_probs.sum()

            top_indices = np.argsort(adjusted_probs)[-3:][::-1]
            top_diseases = [(st.session_state.le_disease.inverse_transform([i])[0], adjusted_probs[i]) for i in top_indices]

            st.write("**Kết quả dự đoán:**")
            for disease, prob in top_diseases:
                st.write(f"- {disease}: {prob:.2%}")
            st.success(f"✅ Dự đoán chính: **{top_diseases[0][0]}** với xác suất {top_diseases[0][1]:.2%}")

            st.write("**Triệu chứng quan trọng nhất ảnh hưởng đến dự đoán:**")
            feature_importance = pd.Series(lr.coef_[lr.classes_ == top_indices[0]][0], index=feature_columns)
            top_features = feature_importance.sort_values(ascending=False).head(3)
            for feature, coef in top_features.items():
                st.write(f"- {feature}: hệ số {coef:.3f}")

            st.markdown("**Gợi ý bệnh viện và hành động cho các bệnh dự đoán:**")
            for idx, (disease, prob) in enumerate(top_diseases):
                hospitals, severity, action = suggest_hospitals_and_actions(disease, hospital_df, action_dict, region)
                with st.container():
                    st.markdown(f"""
                    <div class="suggestion-box">
                        <h4>Bệnh: {disease} ({prob:.2%})</h4>
                        <p><strong>Mức độ nghiêm trọng:</strong> {severity}</p>
                        <p><strong>Hành động:</strong> {action}</p>
                        <p><strong>Bệnh viện gợi ý:</strong> {', '.join(hospitals)}</p>
                    </div>
                    """, unsafe_allow_html=True)
            st.info("⚠️ Đây chỉ là gợi ý từ phương pháp phân cụm và luật kết hợp. Hãy đến cơ sở y tế để được khám và chẩn đoán chính xác.")
# Tab 4: Chatbox
# Tab 4: Chatbox
with tab4:
    st.header("💬 Tư Vấn Sức Khỏe Tự Động")
    st.markdown("Hãy nhập câu hỏi mô tả triệu chứng, ví dụ: `Tôi bị ho và sốt 2 ngày nay.`")

    region = st.selectbox("🌍 Chọn khu vực bạn sống", ["Miền Bắc", "Miền Trung", "Miền Nam"], key="chat_region")

    age = st.number_input("🎂 Tuổi của bạn", min_value=0, max_value=120, value=30, step=1, key="chat_age")
    gender = st.radio("👤 Giới tính của bạn", ["Nam", "Nữ"], key="chat_gender")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("🧑 Bạn:", placeholder="Tôi bị đau họng và khó thở...")

    def normalize_text(text):
        import unicodedata
        return ''.join(c for c in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(c) != 'Mn')

    def generate_chatbot_response(message, age, gender, region):
        if not st.session_state.model_results or "Random Forest" not in st.session_state.model_results:
            return "⚠️ Mô hình chưa được huấn luyện. Vui lòng vào tab 'Dự đoán bệnh' để huấn luyện trước."

        message = normalize_text(message)
        found_symptoms = [symptom for symptom in symptom_columns if normalize_text(symptom) in message]

        if not found_symptoms:
            return "🤖 Tôi chưa nhận diện được triệu chứng nào rõ ràng. Bạn có thể mô tả cụ thể hơn không?"

        input_data = {col: 0 for col in feature_columns}
        for symptom in found_symptoms:
            input_data[symptom] = 1
        input_data["Tuổi"] = age
        input_data["Giới tính"] = st.session_state.le_gender.transform([gender])[0]

        input_df = pd.DataFrame([input_data])

        model = st.session_state.model_results["Random Forest"]["model"]
        probs = model.predict_proba(input_df)[0]
        top_idx = probs.argmax()
        disease = st.session_state.le_disease.inverse_transform([top_idx])[0]
        confidence = probs[top_idx]

        action_info = action_dict.get(disease, {
            "severity": "Chưa rõ",
            "action": "Vui lòng đến cơ sở y tế để được tư vấn cụ thể."
        })

        matches = hospital_df[(hospital_df["Bệnh"] == disease) & (hospital_df["Khu vực"] == region)]
        hospitals = matches["Bệnh viện"].dropna().tolist()
        if not hospitals:
            hospitals = ["Không có bệnh viện gợi ý trong khu vực. Hãy đến cơ sở y tế gần nhất."]

        return (
            f"🩺 **CHẨN ĐOÁN CÓ THỂ: {disease.upper()}**\n\n"
            f"📛 **Mức độ nghiêm trọng:** {action_info['severity']}\n\n"
            f"📌 **Hướng xử lý:** {action_info['action']}\n\n"
            f"🏥 **Gợi ý bệnh viện ({region}):** {', '.join(hospitals)}"
        )

    if st.button("📨 Gửi"):
        if user_input.strip() != "":
            reply = generate_chatbot_response(user_input, age, gender, region)
            st.session_state.chat_history.append(("🧑 Bạn", user_input))
            st.session_state.chat_history.append(("🤖 Hệ thống", reply))
            user_input = ""

    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📜 Lịch sử trò chuyện")
        for sender, msg in st.session_state.chat_history[-6:]:  # Hiển thị 6 dòng gần nhất
            if sender == "🧑 Bạn":
                st.markdown(f"**{sender}:** {msg}")
            else:
                st.markdown(
                    f"<div style='background-color:#f1f1f1;padding:10px;border-radius:8px;margin-bottom:8px'>"
                    f"<strong>{sender}:</strong><br>{msg}</div>",
                    unsafe_allow_html=True
                )
