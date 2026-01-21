import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# =========================
# DATASET 1
# =========================
data1 = np.array([
    [1, 1],
    [4, 1],
    [1, 2],
    [3, 4],
    [5, 4]
])

methods = ['single', 'complete', 'average']

for method in methods:
    Z = linkage(data1, method=method, metric='euclidean')
    plt.figure(figsize=(6, 4))
    dendrogram(Z)
    plt.title(f'Dendrogram Dataset 1 ({method.capitalize()} Linkage)')
    plt.xlabel('Data')
    plt.ylabel('Jarak')
    plt.tight_layout()
    plt.savefig(f'dendrogram_dataset1_{method}.png', dpi=300)
    plt.close()

# =========================
# DATASET 2
# =========================
data2 = np.array([
    [1,1],[4,1],[6,1],[1,2],[2,3],
    [5,3],[2,5],[3,5],[2,6],[3,8]
])

Z2 = linkage(data2, method='average', metric='euclidean')
plt.figure(figsize=(6, 4))
dendrogram(Z2)
plt.title('Dendrogram Dataset 2 (Average Linkage)')
plt.xlabel('Data')
plt.ylabel('Jarak')
plt.tight_layout()
plt.savefig('dendrogram_dataset2_average.png', dpi=300)
plt.close()

print("Semua dendrogram berhasil dibuat.")

