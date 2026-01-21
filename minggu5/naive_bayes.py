# ==============================
# NAIVE BAYES CLASSIFICATION
# DATA MINING - MINGGU 5
# ==============================

# 1. IMPORT LIBRARY
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 2. LOAD DATASET
dataset = pd.read_csv("DataSet.csv", sep=';')

# 3. CLEAN SALARY COLUMN
dataset['Salary'] = dataset['Salary'].str.replace('[^0-9]', '', regex=True)
dataset['Salary'] = dataset['Salary'].astype(int)

# 4. BUAT KATEGORI SALARY (TARGET)
def salary_category(salary):
    if salary < 70000:
        return 'Low'
    elif salary <= 120000:
        return 'Medium'
    else:
        return 'High'

dataset['Salary_Level'] = dataset['Salary'].apply(salary_category)

# 5. PILIH FITUR & TARGET
X = dataset[['Age', 'Gender', 'Education Level', 'Years of Experience']]
y = dataset['Salary_Level']

# 6. ENCODING DATA KATEGORI
encoder = LabelEncoder()
X['Gender'] = encoder.fit_transform(X['Gender'])
X['Education Level'] = encoder.fit_transform(X['Education Level'])

# 7. SPLIT DATA TRAIN & TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. TRAIN MODEL NAIVE BAYES
model = GaussianNB()
model.fit(X_train, y_train)

# 9. PREDIKSI
y_pred = model.predict(X_test)

# 10. EVALUASI MODEL
print("Akurasi Model:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
