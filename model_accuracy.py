#acc
import np as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_val_score

#svm.r
data4 = pd.read_csv('svc.rbf.csv')
X4 = data4.iloc[:,1:6].values
# print(X)
y4 = data4.iloc[:,6].values
# print(y)
np.random.seed(2022)
np.random.shuffle(X4)
np.random.seed(2022)
np.random.shuffle(y4)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X4 = min_max_scaler.fit_transform(X4)

model = svm.SVC(kernel='rbf', C=81, gamma=2)
SVM_r_scores = cross_val_score(model, X4,y4, cv=10, scoring='accuracy')
mean_score = SVM_r_scores.mean()
print(round(mean_score,3))