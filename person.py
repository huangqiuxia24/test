import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot as plt
import seaborn as sns

# sns.set_theme(style="white",font='Times New Roman')
plt.rc('font', family='Times New Roman')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

data = pd.read_csv('data/data1.csv')
# print(data)
feature= data.iloc[:,1:19]
target = data['qf']
print(feature)
print(target)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
feature_final = min_max_scaler.fit_transform(feature)
# print(feature_final)
feature_transform = pd.DataFrame(feature_final, columns = feature.columns)
# print(feature_transform)
selector = VarianceThreshold(threshold=0.02)
result_select = selector.fit_transform(feature_transform)
# print(result_select)
var = selector.variances_
# 方差
print(var)
result_support = selector.get_support(indices=True)
# print(result_support)
select_list = result_support
select1=feature_transform.iloc[:,select_list]
# print(select1)
corr = select1.corr(method='pearson')
# print(corr)

fig, ax = plt.subplots(figsize=(15, 12))
heatmap = sns.heatmap(corr, annot=True, cbar=False, vmax=1, vmin=-1, linewidths=4, linecolor='white', xticklabels=True, yticklabels=True, square=True, cmap="RdBu")

heatmapcb = heatmap.figure.colorbar(heatmap.collections[0])
heatmapcb.ax.tick_params(labelsize=24)
heatmapcb.set_ticks([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])

ax.xaxis.tick_top()
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.xticks(fontsize=32,weight='bold')
plt.yticks(fontsize=32,weight='bold')

plt.savefig('Pearson', dpi=500)
plt.show()