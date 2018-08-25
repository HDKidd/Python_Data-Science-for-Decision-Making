# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 22:04:10 2018

@author: Xie Wenjun 
Copyright: Nanyang Technological University
"""
#%%
#Lesson 5

#%%
# 决策树
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz

import os
os.chdir('C:/Users/XIEW0/Desktop/Data Science with Python/第五课')

data = pd.read_csv('titanic_data.csv', encoding='utf-8')
data.drop(['PassengerId'], axis=1, inplace=True)    # 舍弃ID列，不适合作为特征

# 数据是类别标签，将其转换为数，用1表示男，0表示女。
data.loc[data['Sex'] == 'male', 'Sex'] = 1
data.loc[data['Sex'] == 'female', 'Sex'] = 0
data.fillna(int(data.Age.mean()), inplace=True)
print (data.head(5))

X = data.iloc[:, 1:3]    # 为便于展示，未考虑年龄（最后一列）
y = data.iloc[:, 0]

dtc = DTC(criterion='entropy')   
dtc.fit(X, y)    
print ('输出准确率：', dtc.score(X,y))

# 可视化决策树，导出结果是一个dot文件，需要安装Graphviz才能转换为.pdf或.png格式
with open('tree.dot', 'w') as f:
    f = export_graphviz(dtc, feature_names=X.columns, out_file=f)

#%% 
# 决策树
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

# 仍然使用自带的iris数据
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target

# 训练模型，限制树的最大深度4
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X, y)

# 画图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()

#%%
# KNN (最近邻居)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()     # 加载数据
X = iris.data[:, :2]    # 为方便画图，仅采用数据的其中两个特征
y = iris.target
print (iris.DESCR)
print (iris.feature_names)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

clf = KNeighborsClassifier(n_neighbors=15, weights='uniform')    # 初始化分类器对象
clf.fit(X, y)

# 画出决策边界，用不同颜色表示
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)    # 绘制预测结果图

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)    # 补充训练数据点
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = 15, weights = 'uniform')")
plt.show()
    

#%% 支持向量机
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris = load_iris()     # 加载数据
x = iris.data[:, :2]    # 为方便画图，仅采用数据的其中两个特征
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())
print (clf.score(x_train, y_train))  # 精度
y_hat = clf.predict(x_train)
print('训练集准确度:%f' % (sum(y_hat==y_train)/len(y_train)))
print (clf.score(x_test, y_test))
y_hat = clf.predict(x_test)
print('测试集准确度:%f' % (sum(y_hat==y_test)/len(y_test)))
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
grid_hat = clf.predict(grid_test)       # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
alpha = 0.5
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)     # 预测值的显示
plt.plot(x[:, 0], x[:, 1], 'o', alpha=alpha, color='blue', markeredgecolor='k')
plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel(u'花萼长度', fontsize=13)
plt.ylabel(u'花萼宽度', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
plt.show()

#%% 
# 随机森林 (random forest)
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor  
import numpy as np  
from sklearn.model_selection import train_test_split   
from sklearn.datasets import load_iris  
iris=load_iris()  

x_train, x_test, y_train, y_test = train_test_split(iris.data[:150],\
                    iris.target[:150], random_state=1, train_size=0.6)

#print iris#iris的４个属性是：萼片宽度　萼片长度　花瓣宽度　花瓣长度
#　标签是花的种类：setosa versicolour virginica  
print (iris['target'].shape)
rf=RandomForestRegressor()#这里使用了默认的参数设置  
rf.fit(x_train,y_train)#进行模型的训练  
#    
#随机挑选两个预测不相同的样本  
y_hat = rf.predict(x_train)
print('训练集准确度:%f' % (sum(y_hat==y_train)/len(y_train)))
y_hat = rf.predict(x_test)
print('测试集准确度:%f' % (sum(y_hat==y_test)/len(y_test)))

names = ['sepal length','sepal width','petal length','petal width']
#Variable Importances
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(names, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
#Visualizations of variable importances
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')
x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation = 'vertical')
plt.xticks(x_values, names, rotation='horizontal')
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

#%%
# 逻辑回归
import pandas as pd
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.cross_validation import train_test_split

import os
os.chdir('C:/Users/XIEW0/Desktop/Data Science with Python/第五课')
data = pd.read_csv('LogisticRegression.csv', encoding='utf-8')
print (data.head())
data_dum = pd.get_dummies(data, prefix='rank', columns=['rank'], drop_first=True)
print (data_dum.tail())    # 查看数据框的最后五行

# 切分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data_dum.ix[:, 1:],\
                    data_dum.ix[:, 0], test_size=.1, random_state=520)
lr = LogisticRegression()    # 建立LR模型
lr.fit(x_train, y_train)    # 用处理好的数据训练模型
y_hat = lr.predict(x_train)
print('训练集准确度:%f' % (sum(y_hat==y_train)/len(y_train)))
y_hat = lr.predict(x_test)
print('测试集准确度:%f' % (sum(y_hat==y_test)/len(y_test)))


#%% 
# BP 神经网络
import pandas as pd
wine = pd.read_csv('wine.csv', names = ["Cultivator", "Alchol", "Malic_Acid", \
   "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Falvanoids", \
   "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity",\
   "Hue", "OD280", "Proline"])
wine.head()
wine.describe().transpose()
X = wine.drop('Cultivator',axis=1)
y = wine['Cultivator']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
y_hat = mlp.predict(X_train)
print('训练集准确度:%f' % (sum(y_hat==y_train)/len(y_train)))
y_hat = mlp.predict(X_test)
print('测试集准确度:%f' % (sum(y_hat==y_test)/len(y_test)))

#%%
# Kmeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
plt.figure(figsize=(12, 12))
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)
# 聚类数量不正确时的效果
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)
plt.subplot(221)
plt.scatter(X[y_pred==0][:, 0], X[y_pred==0][:, 1], marker='x',color='b')
plt.scatter(X[y_pred==1][:, 0], X[y_pred==1][:, 1], marker='+',color='r')
plt.title("Incorrect Number of Blobs")
# 聚类数量正确时的效果
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)
plt.subplot(222)
plt.scatter(X[y_pred==0][:, 0], X[y_pred==0][:, 1], marker='x',color='b')
plt.scatter(X[y_pred==1][:, 0], X[y_pred==1][:, 1], marker='+',color='r')
plt.scatter(X[y_pred==2][:, 0], X[y_pred==2][:, 1], marker='1',color='m')
plt.title("Correct Number of Blobs")
# 类间的方差存在差异的效果
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)
plt.subplot(223)
plt.scatter(X_varied[y_pred==0][:, 0], X_varied[y_pred==0][:, 1], marker='x',color='b')
plt.scatter(X_varied[y_pred==1][:, 0], X_varied[y_pred==1][:, 1], marker='+',color='r')
plt.scatter(X_varied[y_pred==2][:, 0], X_varied[y_pred==2][:, 1], marker='1',color='m')
plt.title("Unequal Variance")

# 类的规模差异较大的效果
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)
plt.subplot(224)
plt.scatter(X_filtered[y_pred==0][:, 0], X_filtered[y_pred==0][:, 1], marker='x',color='b')
plt.scatter(X_filtered[y_pred==1][:, 0], X_filtered[y_pred==1][:, 1], marker='+',color='r')
plt.scatter(X_filtered[y_pred==2][:, 0], X_filtered[y_pred==2][:, 1], marker='1',color='m')
plt.title("Unevenly Sized Blobs")
plt.show()
