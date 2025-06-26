import joblib

# Tải mô hình
dt_model = joblib.load("decision_tree.pkl")
rf_model = joblib.load("random_forest.pkl")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu
df = pd.read_excel("data_benh.xlsx")

# Mã hóa bệnh
le = LabelEncoder()
df["Bệnh"] = le.fit_transform(df["Bệnh"])

# Tách features và labels
X = df.drop("Bệnh", axis=1)
y = df["Bệnh"]

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Decision Tree
train_acc_dt = accuracy_score(y_train, dt_model.predict(X_train))
test_acc_dt = accuracy_score(y_test, dt_model.predict(X_test))

# Random Forest
train_acc_rf = accuracy_score(y_train, rf_model.predict(X_train))
test_acc_rf = accuracy_score(y_test, rf_model.predict(X_test))

# In kết quả
print("Decision Tree:")
print("  Train Accuracy:", train_acc_dt)
print("  Test Accuracy :", test_acc_dt)
print("  Test Report:\n", classification_report(y_test, dt_model.predict(X_test)))

print("\nRandom Forest:")
print("  Train Accuracy:", train_acc_rf)
print("  Test Accuracy :", test_acc_rf)
print("  Test Report:\n", classification_report(y_test, rf_model.predict(X_test)))
