import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
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
    # Tìm bệnh viện theo bệnh và khu vực
    matches = hospital_df[(hospital_df["Bệnh"] == disease) & (hospital_df["Khu vực"] == region)]

    # Nếu không có kết quả, thử tìm bệnh gần giống
    if matches.empty:
        closest_disease = find_closest_disease(disease, hospital_df["Bệnh"].unique())
        if closest_disease:
            matches = hospital_df[(hospital_df["Bệnh"] == closest_disease) & (hospital_df["Khu vực"] == region)]
            disease = closest_disease

    hospitals = matches["Bệnh viện"].dropna().tolist()
    if not hospitals:
        hospitals = ["Không có bệnh viện gợi ý trong khu vực này, vui lòng đến cơ sở y tế gần nhất."]

    # Lấy thông tin hành động từ action_dict, nếu không tìm được thì tìm gần giống
    if disease in action_dict:
        action_info = action_dict[disease]
    else:
        closest_disease = find_closest_disease(disease, action_dict.keys())
        if closest_disease:
            action_info = action_dict[closest_disease]
        else:
            action_info = {
                "severity": "Chưa có thông tin",
                "action": "Cần tham khảo ý kiến bác sĩ để đánh giá mức độ nghiêm trọng."
            }

    return hospitals, action_info["severity"], action_info["action"]


# Khởi tạo session_state
if 'le_disease' not in st.session_state:
    st.session_state.le_disease = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'symptom_columns' not in st.session_state:
    st.session_state.symptom_columns = None
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
    "Nhiễm nấm": {"severity": "Trung bình", "action": "Đi khám da liễu, tuân thủ điều trị theo chỉ định."},
    "Dị ứng": {"severity": "Thấp", "action": "Tránh tác nhân gây dị ứng, dùng thuốc theo hướng dẫn bác sĩ."},
    "Trào ngược dạ dày": {"severity": "Trung bình", "action": "Ăn uống điều độ, dùng thuốc theo chỉ định bác sĩ."},
    "Ứ mật mạn tính": {"severity": "Trung bình", "action": "Đi khám chuyên khoa gan mật, tuân thủ chỉ định."},
    "Phản ứng thuốc": {"severity": "Cao", "action": "Ngưng thuốc, đi khám bác sĩ ngay."},
    "Loét dạ dày tá tràng": {"severity": "Trung bình", "action": "Ăn uống hợp lý, dùng thuốc theo chỉ định."},
    "Hội chứng suy giảm miễn dịch": {"severity": "Cao", "action": "Đi khám chuyên khoa, điều trị theo phác đồ."},
    "Tiểu đường": {"severity": "Trung bình", "action": "Kiểm soát đường huyết, tuân thủ chế độ ăn và thuốc."},
    "Viêm dạ dày ruột": {"severity": "Trung bình", "action": "Đi khám chuyên khoa tiêu hóa, điều trị đúng cách."},
    "Hen suyễn": {"severity": "Cao", "action": "Sử dụng thuốc cắt cơn, đi khám nếu khó thở nặng."},
    "Tăng huyết áp": {"severity": "Trung bình", "action": "Theo dõi huyết áp, dùng thuốc đúng chỉ định."},
    "Đau nửa đầu": {"severity": "Trung bình", "action": "Tránh stress, dùng thuốc theo chỉ định."},
    "Thoái hóa đốt sống cổ": {"severity": "Trung bình", "action": "Vật lý trị liệu, dùng thuốc giảm đau."},
    "Liệt (xuất huyết não)": {"severity": "Cao", "action": "Đi khám ngay, điều trị cấp cứu."},
    "Vàng da": {"severity": "Trung bình", "action": "Đi khám chuyên khoa gan mật, xét nghiệm cần thiết."},
    "Sốt rét": {"severity": "Cao", "action": "Đi khám và điều trị ngay, theo dõi sát."},
    "Thủy đậu": {"severity": "Trung bình", "action": "Chăm sóc da, tránh gãi, điều trị hỗ trợ."},
    "Sốt xuất huyết": {"severity": "Cao", "action": "Theo dõi sát, đến cơ sở y tế nếu xuất huyết."},
    "Thương hàn": {"severity": "Cao", "action": "Đi khám và điều trị kịp thời."},
    "Viêm gan A": {"severity": "Trung bình", "action": "Nghỉ ngơi, theo dõi và điều trị hỗ trợ."},
    "Viêm gan B": {"severity": "Cao", "action": "Đi khám chuyên khoa gan, điều trị theo phác đồ."},
    "Viêm gan C": {"severity": "Cao", "action": "Đi khám chuyên khoa gan, điều trị theo phác đồ."},
    "Viêm gan D": {"severity": "Cao", "action": "Đi khám chuyên khoa gan, điều trị theo phác đồ."},
    "Viêm gan E": {"severity": "Trung bình", "action": "Đi khám chuyên khoa gan, điều trị hỗ trợ."},
    "Viêm gan do rượu": {"severity": "Cao", "action": "Ngưng rượu, điều trị chuyên khoa gan."},
    "Lao phổi": {"severity": "Cao", "action": "Đi khám và điều trị lao theo phác đồ."},
    "Cảm lạnh": {"severity": "Thấp", "action": "Nghỉ ngơi, dùng thuốc hạ sốt, giữ ấm."},
    "Viêm phổi": {"severity": "Cao", "action": "Đi khám và điều trị tại bệnh viện."},
    "Trĩ hỗn hợp": {"severity": "Trung bình", "action": "Đi khám chuyên khoa tiêu hóa."},
    "Đau tim": {"severity": "Cao", "action": "Đi khám cấp cứu ngay nếu có triệu chứng."},
    "Giãn tĩnh mạch": {"severity": "Trung bình", "action": "Đi khám chuyên khoa mạch máu."},
    "Suy giáp": {"severity": "Trung bình", "action": "Đi khám nội tiết và điều trị."},
    "Cường giáp": {"severity": "Trung bình", "action": "Đi khám nội tiết và điều trị."},
    "Hạ đường huyết": {"severity": "Cao", "action": "Ăn uống kịp thời, đi khám nếu nặng."},
    "Thoái hóa khớp": {"severity": "Trung bình", "action": "Vật lý trị liệu, dùng thuốc giảm đau."},
    "Thấp khớp": {"severity": "Trung bình", "action": "Đi khám và điều trị chuyên khoa."},
    "Chóng mặt": {"severity": "Thấp", "action": "Đi khám để xác định nguyên nhân."},
    "Mụn trứng cá": {"severity": "Thấp", "action": "Đi khám da liễu, điều trị tại chỗ."},
    "Nhiễm trùng đường tiểu": {"severity": "Trung bình", "action": "Đi khám và dùng thuốc theo chỉ định."},
    "Vảy nến": {"severity": "Trung bình", "action": "Đi khám da liễu và điều trị."},
    "Chốc lở": {"severity": "Thấp", "action": "Giữ vệ sinh, dùng thuốc theo chỉ định."}
}

# Khởi tạo df và hospital_df mặc định
df = pd.read_excel("data_benh.xlsx")
hospital_df = pd.read_excel("benhviengoiy.xlsx")

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(r"\.\d+$", "", regex=True)
df = df.loc[:, ~df.columns.duplicated()]
base_cols = ['Tuổi', 'Giới tính', 'Bệnh hiện tại']
symptom_columns = [col for col in df.columns if col not in base_cols and df[col].dropna().isin([0, 1]).all()]

st.session_state.symptom_columns = symptom_columns

# Sidebar cho tải file
with st.sidebar:
    st.header("📤 Tải Dữ Liệu")
    uploaded_files = st.file_uploader("Chọn file để tải lên", accept_multiple_files=True)

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
            base_cols = ['Tuổi', 'Giới tính', 'Bệnh hiện tại']
            symptom_columns = [col for col in df.columns if col not in base_cols and df[col].dropna().isin([0, 1]).all()]

            st.session_state.symptom_columns = symptom_columns

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

base_cols = ['Tuổi', 'Giới tính', 'Bệnh hiện tại']
symptom_columns = [col for col in binary_cols]
feature_columns = ['Tuổi', 'Giới tính'] + symptom_columns
st.session_state.feature_columns = feature_columns

# Xóa các mẫu bị trùng hoàn toàn trong dữ liệu
df_no_dup = df.drop_duplicates()
#df_no_dup.to_csv("final_benh_dataset.csv", index=False)

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
    # Tạo thư mục lưu mô hình
    model_folder = "saved_models"
    os.makedirs(model_folder, exist_ok=True)

    # Đường dẫn file
    rf_path = os.path.join(model_folder, "random_forest.pkl")
    dt_path = os.path.join(model_folder, "decision_tree.pkl")

    st.header("Dự đoán mô hình")

    # Dùng df_no_dup (đã loại bỏ mẫu trùng)
    X = df_no_dup[feature_columns]
    y = df_no_dup['Bệnh hiện tại']

    # Chia dữ liệu với stratify để phân phối nhãn đều
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    import numpy as np

    # In thông tin cơ bản
    st.write(f"Tổng số mẫu dữ liệu: {len(df_no_dup)}")
    st.write(f"Số mẫu train: {len(X_train)}")
    st.write(f"Số mẫu test: {len(X_test)}")

    with st.expander("Phân phối nhãn train"):
        st.dataframe(y_train.value_counts().rename_axis('Bệnh hiện tại').reset_index(name='count'))

    with st.expander("Phân phối nhãn test"):
        st.dataframe(y_test.value_counts().rename_axis('Bệnh hiện tại').reset_index(name='count'))

    # Kiểm tra số mẫu trùng train-test
    def count_duplicates(X_train, X_test):
        train_np = X_train.to_numpy()
        duplicate_count = 0
        for i in range(len(X_test)):
            sample = X_test.iloc[i].to_numpy()
            if np.any(np.all(train_np == sample, axis=1)):
                duplicate_count += 1
        return duplicate_count

    st.write(f"Số mẫu trùng train-test: {count_duplicates(X_train, X_test)}")

    # Huấn luyện 2 mô hình
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=300, class_weight='balanced'),
    }
# ------------------ TÍCH HỢP LƯU / LOAD MÔ HÌNH ------------------ #
    model_folder = "saved_models"
    os.makedirs(model_folder, exist_ok=True)
    rf_path = os.path.join(model_folder, "random_forest.pkl")
    dt_path = os.path.join(model_folder, "decision_tree.pkl")

    # Nếu mô hình đã tồn tại → load
    rf_model = joblib.load(rf_path) if os.path.exists(rf_path) else None
    dt_model = joblib.load(dt_path) if os.path.exists(dt_path) else None

    model_results = {}

    if rf_model is None or dt_model is None:
        # Train mới nếu chưa có
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=300, class_weight='balanced'),
        }
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
            # Lưu lại mô hình
            if name == "Random Forest":
                joblib.dump(model, rf_path)
            elif name == "Decision Tree":
                joblib.dump(model, dt_path)
    else:
        # Nếu đã có mô hình → đánh giá lại
        for name, model in zip(["Decision Tree", "Random Forest"], [dt_model, rf_model]):
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            model_results[name] = {
                "model": model,
                "accuracy": accuracy,
                "report": report
            }

    # Lưu vào session_state
    st.session_state.model_results = model_results
# --------------------------------------------------------------- #


    # Hiển thị độ chính xác
    accuracy_df = pd.DataFrame({
        "Mô Hình": list(model_results.keys()),
        "Độ Chính Xác": [result["accuracy"] for result in model_results.values()]
    })
    st.subheader("Tổng Hợp Độ Chính Xác")
    st.dataframe(accuracy_df, use_container_width=True)

    fig = px.bar(accuracy_df, x="Mô Hình", y="Độ Chính Xác", 
                 text="Độ Chính Xác", 
                 title="So Sánh Độ Chính Xác Của Các Mô Hình",
                 color="Mô Hình",
                 height=400)
    fig.update_traces(texttemplate='%{text:.4f}', textposition='auto')
    st.plotly_chart(fig, use_container_width=True)

    # Hiển thị báo cáo phân loại
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
        age = st.number_input("Tuổi", min_value=0, max_value=120, value=30, key="age")
        gender = st.selectbox("Giới tính", options=["Nam", "Nữ"], key="gender")
        region = st.selectbox("Khu vực", options=["Miền Bắc", "Miền Trung", "Miền Nam"], key="region")

        st.markdown("**Chọn Triệu Chứng**")
        if st.session_state.get("reset_symptom_form", False):
            for symptom in symptom_columns:
                key = f"symptom_{symptom}"
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state["reset_symptom_form"] = False
            st.rerun()

        # 🧾 Render checkbox và form
        cols = st.columns(3)
        symptoms = {}
        for idx, symptom in enumerate(symptom_columns):
            col_idx = idx % 3
            with cols[col_idx]:
                symptoms[symptom] = st.checkbox(symptom, key=f"symptom_{symptom}")

        col1, col2 = st.columns([1, 1])
        with col1:
            submitted = st.form_submit_button("Dự đoán")
        with col2:
            reset = st.form_submit_button("Reset Triệu Chứng")

        # 🔘 Nếu nhấn Reset, đặt cờ và rerun
        if reset:
            st.session_state["reset_symptom_form"] = True
            st.rerun()

        if submitted:
            if not st.session_state.model_results or not st.session_state.le_disease or not st.session_state.le_gender:
                st.error("Vui lòng huấn luyện mô hình trước!")
                st.stop()
            try:
                age_val = st.session_state["age"]
                gender_val = st.session_state["gender"]
                region_val = st.session_state["region"]
                symptoms_selected = {col: st.session_state.get(f"symptom_{col}", False) for col in symptom_columns}

                input_data = pd.DataFrame(columns=st.session_state.feature_columns)
                input_data.loc[0, 'Tuổi'] = age
                gender_normalized = gender.strip().capitalize()
                input_data.loc[0, 'Giới tính'] = st.session_state.le_gender.transform([gender_normalized])[0]
                for col in symptom_columns:
                    input_data.loc[0, col] = 1 if symptoms[col] else 0
                input_data = input_data.fillna(0).infer_objects(copy=False).astype(int)

                num_symptoms = sum(1 for col, val in symptoms.items() if val)
                if num_symptoms < 2:
                    st.warning("⚠️ Vui lòng chọn ít nhất 2 triệu chứng để dự đoán chính xác hơn.")
                    st.stop()

                light_diseases = ["Cúm", "Viêm họng", "Viêm amidan"]
                severe_symptoms = ["Khó thở", "Đau ngực", "Mất vị giác", "Mất khứu giác"]
                has_severe_symptom = any(symptom in symptoms and symptoms[symptom] for symptom in severe_symptoms)

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

                # ⚠️ Lọc các bệnh từ Decision Tree có xác suất > 0
                other_diseases = [
                    (disease, prob, model)
                    for disease, prob, model in all_diseases
                    if model == "Decision Tree" and disease not in rf_diseases_set and prob > 0.001
                ]

                if other_diseases:
                    other_diseases = sorted(other_diseases, key=lambda x: x[1], reverse=True)
                    final_diseases.append(other_diseases[0])
                else:
                    # Nếu RF còn bệnh thứ 3 và chưa có trong final_diseases thì thêm vào
                    sorted_rf = sorted(model_predictions["Random Forest"], key=lambda x: x[1], reverse=True)
                    if len(sorted_rf) > 2:
                        third_rf_disease = sorted_rf[2]
                        if third_rf_disease[0] not in rf_diseases_set:
                            final_diseases.append((third_rf_disease[0], third_rf_disease[1], "Random Forest"))

                # Chọn lại top 3 theo xác suất
                final_diseases = sorted(final_diseases, key=lambda x: x[1], reverse=True)[:3]

                colors = ["#28a745", "#fd7e14", "#6f42c1"]

                st.markdown("### 🩺 **Kết Quả Tư Vấn Cá Nhân Hóa**")
                st.markdown(f"""
                **Thông tin người dùng:**
                - Tuổi: {age}
                - Giới tính: {gender}
                - Khu vực: {region}
                - Triệu chứng: {', '.join([col for col, val in symptoms.items() if val]) or 'Không có'}

                **Mô hình dự đoán:**
                Kết quả dự đoán chính là:
                """)
                for idx, (disease, prob, model) in enumerate(final_diseases):
                    st.markdown(f"- <span style='color:{colors[idx]}'>{disease}</span>", unsafe_allow_html=True)
                st.markdown("- Dựa trên tổ hợp các triệu chứng và tiền sử bệnh liên quan")

                st.markdown("**Gợi ý hành động:**")
                for disease, prob, model in final_diseases:
                    hospitals, severity, action = suggest_hospitals_and_actions(disease, hospital_df, action_dict, region)
                    note = ""
                    if prob < 0.1:
                        note = "<p><em>⚠️ Lưu ý: Bệnh này được mô hình gợi ý nhưng có xác suất rất thấp.</em></p>"
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
                for idx, row in rules.iterrows():
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

        st.markdown("**Chọn Triệu Chứng**")
        if st.session_state.get("reset_cluster_form", False):
            for symptom in symptom_columns:
                key = f"cluster_symptom_{symptom}"
                if key in st.session_state:
                    del st.session_state[key]  # ❗Xóa khỏi session_state để không lỗi
            st.session_state["reset_cluster_form"] = False
            st.rerun()
        cols = st.columns(3)
        cluster_symptoms = {}
        for idx, symptom in enumerate(symptom_columns):
            col_idx = idx % 3
            with cols[col_idx]:
                default_value = False if st.session_state.get("reset_cluster_form", False) else st.session_state.get(f"cluster_symptom_{symptom}", False)
                cluster_symptoms[symptom] = st.checkbox(symptom, value=default_value, key=f"cluster_symptom_{symptom}")

        col1, col2 = st.columns([1, 1])
        with col1:
            submitted = st.form_submit_button("🔍 Dự đoán từ cụm")
        with col2:
           reset = st.form_submit_button("Reset Triệu Chứng")
        if reset:
            st.session_state["reset_cluster_form"] = True
            st.rerun()

        # Nếu bấm Reset, cập nhật session_state trước khi render
        if st.session_state.get("reset_cluster_form", False):
            for symptom in symptom_columns:
                st.session_state[f"cluster_symptom_{symptom}"] = False
            st.session_state["reset_cluster_form"] = False
            st.rerun()


        if st.session_state.get("reset_cluster_form", False):
            st.session_state["reset_cluster_form"] = False

        if submitted:
            # Lấy dữ liệu nhập vào
            age_val = st.session_state["cluster_age"]
            gender_val = st.session_state["cluster_gender"]
            region_val = st.session_state["cluster_region"]
            symptoms_selected = {col: st.session_state.get(f"cluster_symptom_{col}", False) for col in symptom_columns}

            # Tạo dataframe input đúng định dạng
            input_data = pd.DataFrame(columns=feature_columns)
            input_data.loc[0, 'Tuổi'] = age_val
            gender_normalized = gender_val.strip().capitalize()
            input_data.loc[0, 'Giới tính'] = st.session_state.le_gender.transform([gender_normalized])[0]

            for col in symptom_columns:
                input_data.loc[0, col] = 1 if symptoms_selected[col] else 0
            input_data = input_data.fillna(0).astype(int)

            num_symptoms = sum(val for val in symptoms_selected.values())
            if num_symptoms < 3:
                st.warning("⚠️ Vui lòng chọn ít nhất 3 triệu chứng để tăng khả năng tìm luật kết hợp.")
                st.stop()

            # Dự đoán cụm
            cluster_assignment = kmeans.predict(input_data)[0]
            st.write(f"Triệu chứng đầu vào thuộc **Cụm {cluster_assignment}**")

            # Lấy dữ liệu của cụm dự đoán
            cluster_data = df_resampled[df_resampled['Cụm'] == cluster_assignment]
            if len(cluster_data) < 2:
                st.warning("⚠️ Cụm này có quá ít dữ liệu để dự đoán chính xác. Vui lòng thử với số lượng cụm khác.")
                st.stop()

            # Phân phối bệnh trong cụm
            disease_counts = cluster_data['Bệnh hiện tại'].value_counts()
            st.write("**Phân bố bệnh trong cụm:**")
            disease_distribution = pd.DataFrame({
                "Bệnh": [st.session_state.le_disease.inverse_transform([idx])[0] for idx in disease_counts.index],
                "Số bệnh nhân": disease_counts.values,
                "Tỷ lệ": [f"{count/len(cluster_data):.2%}" for count in disease_counts.values]
            })
            st.dataframe(disease_distribution, use_container_width=True, hide_index=True)

            # Huấn luyện logistic regression trong cụm
            X_cluster = cluster_data[feature_columns]
            y_cluster = cluster_data['Bệnh hiện tại']
            lr = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial', class_weight='balanced')
            lr.fit(X_cluster, y_cluster)

            # Dự đoán xác suất
            probs = lr.predict_proba(input_data)[0]

            # Tìm luật kết hợp khớp với triệu chứng
            matched_rules = []
            if st.session_state.cluster_rules and cluster_assignment in st.session_state.cluster_rules:
                rules = st.session_state.cluster_rules[cluster_assignment]
                if not rules.empty:
                    input_items = [col for col, val in symptoms_selected.items() if val]
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
                    rule_display.append({
                        "Điều kiện": rule['antecedents'],
                        "Kết luận": rule['disease'],
                        "Độ tin cậy": f"{rule['confidence']:.2%}",
                        "Tỷ lệ khớp": f"{rule['match_ratio']:.0%}"
                    })
                st.dataframe(pd.DataFrame(rule_display), use_container_width=True, hide_index=True)

            # Chuẩn hóa xác suất tránh 0
            adjusted_probs = probs + 1e-3
            adjusted_probs = adjusted_probs / adjusted_probs.sum()

            # Hiển thị 3 bệnh có xác suất cao nhất
            top_indices = np.argsort(adjusted_probs)[-3:][::-1]
            top_diseases = [(st.session_state.le_disease.inverse_transform([i])[0], adjusted_probs[i]) for i in top_indices]

            st.write("### Top 3 bệnh dự đoán:")
            for disease, prob in top_diseases:
                st.write(f"- {disease}")
            st.success(f"✅ Dự đoán chính: {top_diseases[0][0]}")

            st.write("**Triệu chứng quan trọng nhất ảnh hưởng đến dự đoán:**")
            feature_importance = pd.Series(lr.coef_[lr.classes_ == top_indices[0]][0], index=feature_columns)
            top_features = feature_importance.sort_values(ascending=False).head(3)
            for feature, coef in top_features.items():
                st.write(f"- {feature}: hệ số {coef:.3f}")

            st.markdown("**Gợi ý bệnh viện và hành động cho các bệnh dự đoán:**")
            for idx, (disease, prob) in enumerate(top_diseases):
                hospitals, severity, action = suggest_hospitals_and_actions(disease, hospital_df, action_dict, region)
                note = ""
                if prob < 0.1:
                    note = "<p><em>⚠️ Lưu ý: Bệnh này được mô hình gợi ý nhưng có xác suất rất thấp.</em></p>"
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
with tab4:
    # Nếu bấm Reset, cập nhật session_state trước khi render
    if st.session_state.get("reset_cluster_form", False):
        for symptom in symptom_columns:
            st.session_state[f"cluster_symptom_{symptom}"] = False
        st.session_state["reset_cluster_form"] = False
        st.rerun()

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
        top_indices = probs.argsort()[-2:][::-1]  # Lấy 2 bệnh có xác suất cao nhất
        top_diseases = [(st.session_state.le_disease.inverse_transform([i])[0], probs[i]) for i in top_indices]

        result_lines = ["🤖 **CHẨN ĐOÁN CÓ THỂ:**"]
        for disease, confidence in top_diseases:
            severity = action_dict.get(disease, {}).get("severity", "Chưa rõ")
            action = action_dict.get(disease, {}).get("action", "Vui lòng đến cơ sở y tế để được tư vấn cụ thể.")

            matches = hospital_df[(hospital_df["Bệnh"] == disease) & (hospital_df["Khu vực"] == region)]
            hospitals = matches["Bệnh viện"].dropna().tolist()
            if not hospitals:
                hospitals = ["Không có bệnh viện gợi ý trong khu vực. Hãy đến cơ sở y tế gần nhất."]

            result_lines.append(
                f"""🩺 **{disease.upper()}** 
    ❤️ **Mức độ nghiêm trọng:** {severity}  
    📌 **Hướng xử lý:** {action}  
    🏥 **Gợi ý bệnh viện ({region}):** {', '.join(hospitals)}"""
            )

        return "\n\n".join(result_lines)


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
