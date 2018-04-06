import numpy as np
from sklearn.cluster import KMeans

data = np.random.rand(100, 3)
data2 = np.random.rand(100,3)
print(data)
estimator = KMeans(n_clusters=3)
estimator.fit(data)
# 获取聚类标签
label_pred = estimator.labels_
print(label_pred)
# 获取聚类中心
centroids = estimator.cluster_centers_
print(centroids)
# 聚类中心均值向量的总和
inertia = estimator.inertia_
print(inertia)
# 对数据集进行预测
result = estimator.predict(data2)
print(result)
