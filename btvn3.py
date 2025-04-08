import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Đọc file train
train_df = pd.read_csv('train.csv')

# Tiền xử lý train
train_df['Sex'] = LabelEncoder().fit_transform(train_df['Sex'])

train_df.loc[:, 'Embarked'] = train_df['Embarked'].fillna('S')
train_df['Embarked'] = LabelEncoder().fit_transform(train_df['Embarked'])

train_df.loc[:, 'Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df.loc[:, 'Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_train = train_df[features]
y_train = train_df['Survived']

# Huấn luyện
model = GaussianNB()
model.fit(X_train, y_train)

# Đọc file test
test_df = pd.read_csv('test.csv')

test_df['Sex'] = LabelEncoder().fit_transform(test_df['Sex'])

test_df.loc[:, 'Embarked'] = test_df['Embarked'].fillna('S')
test_df['Embarked'] = LabelEncoder().fit_transform(test_df['Embarked'])

test_df.loc[:, 'Age'] = test_df['Age'].fillna(train_df['Age'].median())
test_df.loc[:, 'Fare'] = test_df['Fare'].fillna(train_df['Fare'].median())

X_test = test_df[features]
y_pred = model.predict(X_test)

# Lưu kết quả
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_pred
})
submission.to_csv('submission.csv', index=False)
print("✅ Done! File submission.csv đã được lưu rồi")
