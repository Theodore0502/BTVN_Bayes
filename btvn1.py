import pandas as pd
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Hàm Data_Generation() để sinh dữ liệu
def Data_Generation():
    columns = ['Age', 'Gender', 'BloodPressure', 'Cholesterol', 'HeartRate', 'HasHeartDisease']
    
    # Sinh dữ liệu cho training dataset
    training_data = []
    for _ in range(100):
        age = random.randint(20, 80)
        gender = random.choice([0, 1])  # 0: Female, 1: Male
        blood_pressure = random.randint(100, 180)
        cholesterol = random.randint(150, 250)
        heart_rate = random.randint(60, 100)
        has_heart_disease = random.choice([0, 1])  # 0: No, 1: Yes
        training_data.append([age, gender, blood_pressure, cholesterol, heart_rate, has_heart_disease])
    
    # Sinh dữ liệu cho testing dataset
    testing_data = []
    for _ in range(10):
        age = random.randint(20, 80)
        gender = random.choice([0, 1])
        blood_pressure = random.randint(100, 180)
        cholesterol = random.randint(150, 250)
        heart_rate = random.randint(60, 100)
        has_heart_disease = random.choice([0, 1])
        testing_data.append([age, gender, blood_pressure, cholesterol, heart_rate, has_heart_disease])
    
    # Lưu dữ liệu vào file CSV
    training_df = pd.DataFrame(training_data, columns=columns)
    testing_df = pd.DataFrame(testing_data, columns=columns)
    training_df.to_csv('training_dataset.csv', index=False)
    testing_df.to_csv('testing_dataset.csv', index=False)
    print("Data has been generated and saved.")

# 2. Hàm Load_Data() để nạp dữ liệu từ file
def Load_Data(file_path):
    data = pd.read_csv(file_path)
    return data

# 3. Hàm Bayes() để huấn luyện mô hình Naive Bayes
def Bayes(training_data):
    X = training_data[['Age', 'Gender', 'BloodPressure', 'Cholesterol', 'HeartRate']]
    y = training_data['HasHeartDisease']
    
    model = GaussianNB()
    model.fit(X, y)
    
    return model

# 4. Hàm Testing() để kiểm tra mô hình
def Testing(model, testing_data):
    X_test = testing_data[['Age', 'Gender', 'BloodPressure', 'Cholesterol', 'HeartRate']]
    y_test = testing_data['HasHeartDisease']
    
    y_pred = model.predict(X_test)
    
    return y_test, y_pred

# 5. Hàm Performance() để đánh giá hiệu quả của mô hình
def Performance(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

# 6. Hàm __main__ thực hiện tất cả các bước
def __main__():
    # Sinh dữ liệu
    Data_Generation()
    
    # Nạp dữ liệu
    training_data = Load_Data('training_dataset.csv')
    testing_data = Load_Data('testing_dataset.csv')
    
    # Huấn luyện mô hình
    model = Bayes(training_data)
    
    # Kiểm tra mô hình
    y_test, y_pred = Testing(model, testing_data)
    
    # Đánh giá mô hình
    Performance(y_test, y_pred)

if __name__ == "__main__":
    __main__()
