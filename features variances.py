import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

# 假设您有一个分类模型的特征矩阵 X 和目标向量 y
data = pd.read_csv('data/data1.csv')
X = data.iloc[:,1:19].values
# print(X)
y = data.iloc[:,20].values
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X = min_max_scaler.fit_transform(X)
# 计算特征与目标值之间的方差
variances = np.var(X, axis=0)

# 获取特征数量
num_features = len(variances)

# 获取特征名称
feature_names = ['va','d','blm','bpv','ppv','bg','fepa','na','da','v','epa','cbm','ef','ne','cat','ani','car','uepa']  # 替换为您的特征名称列表

# 对方差进行排序
sorted_indices = np.argsort(variances)[::-1]
sorted_variances = variances[sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]

# 绘制横向柱状图
plt.barh(range(num_features), sorted_variances, align='center')
plt.yticks(range(num_features), sorted_feature_names)
plt.xlabel('Variance')
plt.ylabel('Feature')
plt.title('Variance of Features with Target')

# 调整坐标轴标签的显示位置
plt.gca().invert_yaxis()

plt.show()