#knn
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
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
x_train,x_test,y_train,y_test=train_test_split(x,y)

my_knn_clf=KNNClassifier(k=3)
print(my_knn_clf)
my_knn_clf.fit(x_train,y_train)
y_predict=my_knn_clf.predict(x_test)
print(y_predict)
from PLAY_KNN.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))
print(my_knn_clf.score(x_test,y_test))

