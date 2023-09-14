import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('validation.csv')
X = data.iloc[:,1:6].values
# print(X)
y = data.iloc[:,6].values
data_name=data['formula']
# print(y)
np.random.seed(2022)
np.random.shuffle(X)
np.random.seed(2022)
np.random.shuffle(y)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X = min_max_scaler.fit_transform(X)
# 创建SVC模型
model = SVC(kernel='rbf',C=81,gamma=2)

# 打乱数据集和数据名
# np.random.seed(2022)  # 设置随机种子，保证结果可重现
random_indices = np.random.permutation(len(X))
X_shuffled = X[random_indices]
y_shuffled = y[random_indices]
data_names_shuffled = [data_name[i] for i in random_indices]

# raw_data_shuffled=[raw_data[i]for i in random_indices]
# 进行交叉验证并输出分类结果
predictions = cross_val_predict(model, X_shuffled, y_shuffled, cv=10)
# print(X_shuffled[i])
# 打印每个数据的分类结果和对应的数据名
for i in range(len(predictions)):
    print(f"数据名：{data_names_shuffled[i]}，分类结果：{predictions[i]}")
    # print(X_shuffled[i])
