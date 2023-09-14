import pandas
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('svc.rbf.csv')
X = data.iloc[:,1:6].values
# print(X)
y = data.iloc[:,6].values
# print(y)
np.random.seed(2022)
np.random.shuffle(X)
np.random.seed(2022)
np.random.shuffle(y)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X = min_max_scaler.fit_transform(X)

#
from sklearn import svm
model_SVC = svm.SVC()
param_range_C = [i for i in range(1,100,10)]
param_range_gamma = [1,2,3,4,5,6,7,8,9,10]
param_grid = [{"C":param_range_C,"gamma":param_range_gamma,"kernel":["rbf"]}]
grid = GridSearchCV(model_SVC,param_grid,cv=10,scoring='accuracy')
grid.fit(X,y)
print("SVM.rbf Best parameters: ", grid.best_params_)
print("最优得分：", grid.best_score_)
