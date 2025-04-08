import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Bước 1: Đọc dữ liệu
df = pd.read_csv('synthetic_cancer_dataset.csv')  # File từ bài trước

# Bước 2: Encode nhãn ung thư
le = LabelEncoder()
df['cancer_type_encoded'] = le.fit_transform(df['cancer_type'])

# Bước 3: Tách dữ liệu thành X và y
X = df[['tumor_size', 'cell_density', 'cell_morphology']]
y = df['cancer_type_encoded']

# Bước 4: Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Bước 5: Khởi tạo mô hình
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Chuyển nhãn thành chuỗi để tránh lỗi khi dùng classification_report
target_names = [str(cls) for cls in le.classes_]

# Bước 6: Mở file để ghi kết quả
with open("results.txt", "w", encoding="utf-8") as f:
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f.write(f"\n--- {name} ---\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=target_names))
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
        f.write("\n" + "=" * 60 + "\n")

print("✅ Đã lưu kết quả vào file results.txt rồi nha")
