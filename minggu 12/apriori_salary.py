import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Load dataset
data = pd.read_csv("DataSet.csv", sep=';')

# 2. Clean data Salary
data['Salary'] = data['Salary'].str.replace('[^0-9]', '', regex=True)
data['Salary'] = data['Salary'].astype(int)

# 3. Pilih kolom penting
df = data[['Age', 'Gender', 'Education Level', 'Years of Experience', 'Salary']].copy()

# 4. Diskretisasi (numerik -> kategori / boolean)
df['Gaji_Rendah'] = df['Salary'] < 70000
df['Gaji_Sedang'] = (df['Salary'] >= 70000) & (df['Salary'] <= 120000)
df['Gaji_Tinggi'] = df['Salary'] > 120000

df['Pengalaman_Pendek'] = df['Years of Experience'] < 5
df['Pengalaman_Panjang'] = df['Years of Experience'] >= 5

df['Usia_Muda'] = df['Age'] < 30
df['Usia_Tua'] = df['Age'] >= 30

df['Pendidikan_Tinggi'] = df['Education Level'].isin(['S2', 'S3'])

# 5. Dataset transaksi (boolean)
transactions = df[
    [
        'Gaji_Rendah',
        'Gaji_Sedang',
        'Gaji_Tinggi',
        'Pengalaman_Pendek',
        'Pengalaman_Panjang',
        'Usia_Muda',
        'Usia_Tua',
        'Pendidikan_Tinggi'
    ]
]

# 6. Frequent Itemset (Apriori)
frequent_itemsets = apriori(
    transactions,
    min_support=0.3,
    use_colnames=True
)

print("=== FREQUENT ITEMSETS ===")
print(frequent_itemsets.sort_values('support', ascending=False))

# 7. Association Rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

print("\n=== ASSOCIATION RULES ===")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
