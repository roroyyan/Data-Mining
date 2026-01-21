# ==============================
# K-NN CLASSIFICATION
# DATA MINING - MINGGU 6
# ==============================

# 1. IMPORT LIBRARY
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. LOAD DATASET
dataset = pd.read_csv("DataSet.csv", sep=';')

# 3. CLEAN DATA SALARY
dataset['Salary'] = dataset['Salary'].str.replace('[^0-9]', '', regex=True)
dataset['Salary'] = dataset['Salary'].astype(int)

# 4. BUAT KATEGORI SALARY (TARGET KLASIFIKASI)
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

# 7. NORMALISASI DATA (PENTING UNTUK K-NN)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 8. SPLIT DATA TRAIN & TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. TRAIN MODEL K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 10. PREDIKSI
y_pred = knn.predict(X_test)

# 11. EVALUASI MODEL
print("Akurasi Model K-NN:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
