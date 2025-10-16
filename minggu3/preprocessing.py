# LANGKAH 1: IMPORT SEMUA LIBRARY YANG DIBUTUHKAN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# LANGKAH 2: BACA DATASET
# Pastikan nama file 'DataSet.csv' sudah benar dan ada di folder yang sama.
print("Membaca file DataSet.csv...")
dataset = pd.read_csv('DataSet.csv', sep=';')

# Tampilkan 5 baris pertama untuk memastikan data terbaca dengan benar
print("5 Data Teratas:")
print(dataset.head())

# LANGKAH 3: PISAHKAN FITUR (X) DAN TARGET (y)
# X adalah semua kolom KECUALI kolom terakhir (Salary)
# y adalah HANYA kolom terakhir (Salary)
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

print("\nFitur (X) berhasil dipisahkan dari Target (y).")

# LANGKAH 4: BAGI DATA MENJADI TRAINING & TESTING
# 80% data untuk melatih model, 20% untuk menguji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("\nData berhasil dibagi menjadi data training dan testing.")
print("Jumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))

print("\n--- Proses Selesai ---")
