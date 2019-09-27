#knn
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore',category=FutureWarning,module='sklearn',lineno=1978)
# from sklearn import datasets
# iris=datasets.load_iris()
# x=iris.data
# y=iris.target

#识别手写数字
digits=datasets.load_digits()
# print(digits.keys())
# dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])
x=digits.data
y=digits.target
# print(x.shape)
# (1797, 64)
# print(y.shape)
# (1797,)

from PLAY_KNN.KNN import KNNClassifier
from PLAY_KNN.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_seed=666)

# my_knn_clf=KNNClassifier(k=3)
# print(my_knn_clf)
# my_knn_clf.fit(x_train,y_train)
# y_predict=my_knn_clf.predict(x_test)
# print(y_predict)
# from PLAY_KNN.metrics import accuracy_score
# print(accuracy_score(y_test,y_predict))
# print(my_knn_clf.score(x_test,y_test))

#自己写的网格搜索
#超参数k，与超参数weights，距离做权重
# #超参数p明可夫斯距离，1曼哈顿，2欧式，
#unform不考虑距离，无p；distance才有p

# best_method=''
# best_score=0.0
# best_k=0.0
# for i in ['uniform','distance']:
#     for k in range(1,11):
#         knn=KNeighborsClassifier(n_neighbors=k,weights=i,p=3)
#         knn.fit(x_train,y_train)
#         score=knn.score(x_test,y_test)
#         if score>best_score:
#             best_score=score
#             best_k=k
#             best_method=i
# print(best_k,best_score,best_method)

#使用sklearn的网格搜索
param_grid=[
    {
        'weights':['uniform'],
        'n_neighbors':[i for i in range(2,4)],

    },
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(2,4)],
        'p':[i for i in range(1,3)],

    }
]

knn=KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(knn,param_grid,n_jobs=-1,verbose=2)#verbose值越大， 计算中显示的信息越冗长
grid.fit(x_train,y_train)
#
print(grid.best_estimator_)
print(grid.best_score_)
print(grid.best_params_)
knn=grid.best_estimator_
print(knn.predict(x_test))


