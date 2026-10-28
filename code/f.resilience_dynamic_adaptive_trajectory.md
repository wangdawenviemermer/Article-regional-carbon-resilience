```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.metrics import davies_bouldin_score

# ==========================================
# 0. 加载数据
# ==========================================
file_path = r'E:\study\g.DATA\b.data_COVID_carbon\phase4_method\state_wave_wide_format_data.csv'
raw_data = pd.read_csv(file_path, index_col='state')

# ==========================================
# 1. 数据清洗与格式化
# ==========================================
series_list, state_names = [], []
for name, row in raw_data.iterrows():
    valid_data = row.dropna().values
    if len(valid_data) > 0:
        series_list.append(valid_data.reshape(-1, 1))
        state_names.append(name)

# ==========================================
# 2. 变换与训练
# ==========================================
scaler = TimeSeriesScalerMinMax(value_range=(0, 1))
X_train = scaler.fit_transform(to_time_series_dataset(series_list))

n_clusters = 10
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=100, n_init=20, random_state=42, n_jobs=-1)
labels = model.fit_predict(X_train)

# ==========================================
# 3. 聚类有效性指标计算 (验证分得好不好)
# ==========================================
# 指标1：轮廓系数 (Silhouette Score) - 越接近1越好
# 注意：DTW运算较慢，计算指标时也使用dtw作为度量
sil_score = silhouette_score(X_train, labels, metric="dtw")

# 指标2：Inertia (凝聚度) - 越小代表类内越紧凑
inertia = model.inertia_

# ==========================================
# 4. 图表一：30个省份缩放后的全样本图 (Spaghetti Plot)
# ==========================================
plt.figure(figsize=(10, 6))
for i in range(len(X_train)):
    plt.plot(X_train[i].flatten(), color='gray', alpha=0.3, linewidth=1)
plt.title("Figure 1: Normalized Time Series of All 30 Provinces", fontsize=14)
plt.xlabel("Time Stages")
plt.ylabel("Normalized Value (0-1)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==========================================
# 5. 图表二：聚类结果详细分布图
# ==========================================
plt.figure(figsize=(12, 16))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd','#8c564b','#e377c2','#bcbd22','#17becf','#e6ab02']
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for cluster_id in range(n_clusters):
    plt.subplot(n_clusters, 1, cluster_id + 1)
    indices = np.where(labels == cluster_id)[0]
    for idx in indices:
        plt.plot(X_train[idx].flatten(), color='gray', alpha=0.2)

    centroid = model.cluster_centers_[cluster_id].flatten()
    plt.plot(centroid, color=colors[cluster_id], linewidth=4, label=f'Pattern {cluster_id + 1}')
    plt.title(f"Cluster Group {cluster_id + 1} (N={len(indices)})", fontsize=12)
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='upper right')

plt.suptitle("Figure 2: Time Series Clustering Analysis Results", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ==========================================
# 6. 最终报告输出
# ==========================================
print("\n" + "=" * 50)
print("【聚类质量评价报告】")
print(f"1. 轮廓系数 (Silhouette Score): {sil_score:.4f}  (越接近1代表类间区分度越高)")
print(f"2. 类内距离平方和 (Inertia): {inertia:.4f}  (越小代表类内越紧凑)")
print("=" * 50)

results = pd.DataFrame({'Province': state_names, 'Cluster_ID': labels + 1})
for i in range(1, n_clusters + 1):
    p_list = results[results['Cluster_ID'] == i]['Province'].tolist()
    print(f"模式 {i}: {', '.join(p_list)}")
```