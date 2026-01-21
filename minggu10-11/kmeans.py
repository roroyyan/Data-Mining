# ==============================
# K-MEANS CLUSTERING
# DATA MINING - MINGGU 6
# ==============================

# 1. IMPORT LIBRARY
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 2. LOAD DATASET
dataset = pd.read_csv("DataSet.csv", sep=';')

# 3. CLEAN DATA SALARY
dataset['Salary'] = dataset['Salary'].str.replace('[^0-9]', '', regex=True)
dataset['Salary'] = dataset['Salary'].astype(int)

# 4. PILIH FITUR UNTUK KLASTERING
X = dataset[['Age', 'Gender', 'Education Level', 'Years of Experience', 'Salary']]

# 5. ENCODING DATA KATEGORI
encoder = LabelEncoder()
X['Gender'] = encoder.fit_transform(X['Gender'])
X['Education Level'] = encoder.fit_transform(X['Education Level'])

# 6. NORMALISASI DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. PROSES KLASTERING K-MEANS
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 8. TAMBAHKAN HASIL KLASTER KE DATASET
dataset['Cluster'] = clusters

# 9. TAMPILKAN JUMLAH DATA TIAP KLASTER
print("Jumlah data tiap klaster:")
print(dataset['Cluster'].value_counts())

# 10. VISUALISASI (2 DIMENSI)
plt.figure(figsize=(8,6))
plt.scatter(
    dataset['Age'],
    dataset['Salary'],
    c=dataset['Cluster'],
    cmap='viridis'
)
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Hasil Klastering K-Means")
plt.colorbar(label='Cluster')
plt.show()
