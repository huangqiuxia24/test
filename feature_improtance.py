import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv('data/data2.csv')
X = data.iloc[:,1:13].values
y = data.iloc[:,13].values

np.random.seed(2022)
np.random.shuffle(X)
np.random.seed(2022)
np.random.shuffle(y)
# print(X)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X = min_max_scaler.fit_transform(X)

feature_names = ['va','d','bpv','bg','fepa','na','da','cbm','ef','ne','cat','car']

# 训练决策树分类器
dt = DecisionTreeClassifier(random_state=42,max_depth=8,min_samples_split=4)
dt.fit(X, y)
dt_importance = dt.feature_importances_

# 训练RF分类器
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=8,n_estimators=150,max_features=1)
rf.fit(X,y)
rf_importance=rf.feature_importances_

# 训练GDBT分类器
from sklearn.ensemble import GradientBoostingClassifier
gdbt = GradientBoostingClassifier(n_estimators=150,learning_rate=2,max_depth=4)
gdbt.fit(X,y)
gdbt_importance=gdbt.feature_importances_

# 创建并绘制特征重要性图
fig, ax = plt.subplots()
ax.barh(np.arange(len(feature_names)), dt_importance, align='center', height=0.2, label='DT')
ax.barh(np.arange(len(feature_names)) + 0.2, rf_importance, align='center', height=0.2, label='RF')
ax.barh(np.arange(len(feature_names)) + 0.4, gdbt_importance, align='center', height=0.2, label='GDBT')

ax.set_yticks(np.arange(len(feature_names)))
ax.set_yticklabels(feature_names)
ax.set_xlabel('Importance')
ax.set_title('Feature Importance Comparison')
ax.legend()
plt.tight_layout()
plt.show()
