import pandas as pd
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Data_Generation(): sinh dữ liệu giả lập và lưu ra CSV
def Data_Generation():
    columns = ['TumorSize', 'CellDensity', 'CellShape', 'CancerType']
    cancer_types = {0: 'Breast', 1: 'Lung', 2: 'Stomach'}
    
    # sinh 100 mẫu training
    train = []
    for _ in range(100):
        size = round(random.uniform(0.5, 5.0), 2)         # kích thước khối u (cm)
        density = round(random.uniform(10, 100), 2)       # mật độ tế bào
        shape = round(random.uniform(0.1, 1.0), 2)        # hình thái tế bào (độ bất thường)
        label = random.choice(list(cancer_types.keys()))
        train.append([size, density, shape, label])
    
    # sinh 10 mẫu testing
    test = []
    for _ in range(10):
        size = round(random.uniform(0.5, 5.0), 2)
        density = round(random.uniform(10, 100), 2)
        shape = round(random.uniform(0.1, 1.0), 2)
        label = random.choice(list(cancer_types.keys()))
        test.append([size, density, shape, label])
    
    # lưu ra file
    pd.DataFrame(train, columns=columns).to_csv('training_dataset.csv', index=False)
    pd.DataFrame(test,  columns=columns).to_csv('testing_dataset.csv',  index=False)
    print("Đã sinh và lưu dữ liệu training_dataset.csv, testing_dataset.csv")

# 2. Load_Data(): nạp CSV vào DataFrame
def Load_Data(path):
    return pd.read_csv(path)

# 3. Bayes(): huấn luyện Gaussian Naive Bayes
def Bayes(train_df):
    X = train_df[['TumorSize', 'CellDensity', 'CellShape']]
    y = train_df['CancerType']
    model = GaussianNB()
    model.fit(X, y)
    return model

# 4. Testing(): dự đoán trên tập test
def Testing(model, test_df):
    X_test = test_df[['TumorSize', 'CellDensity', 'CellShape']]
    y_true = test_df['CancerType']
    y_pred = model.predict(X_test)
    return y_true, y_pred

# 5. Performance(): đánh giá mô hình
def Performance(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Breast','Lung','Stomach']))

# 6. __main__: chạy tuần tự tất cả
def __main__():
    # 1. Sinh dữ liệu
    Data_Generation()
    
    # 2. Load training, huấn luyện
    train_df = Load_Data('training_dataset.csv')
    model = Bayes(train_df)
    
    # 3. Load testing, dự đoán
    test_df = Load_Data('testing_dataset.csv')
    y_true, y_pred = Testing(model, test_df)
    
    # 4. Đánh giá
    Performance(y_true, y_pred)

if __name__ == "__main__":
    __main__()
