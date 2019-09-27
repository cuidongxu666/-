# 自己写的knn代码

# def KNN_classify(k,x_train,y_train,x):
#     assert 1<=k<=x_train[0],'k must be valid'
#     assert x_train.shape[0]==y_train.shape[0],\
#         'the size of x be equal to the size of y_train'
#     assert x_train.shape[1]==x.shape[0],\
#         'the feature number of x must be equal to x_train'
#     计算欧式距离
#     distances=[sqrt(np.sum((x_train-x)**2)) for x_train in x_train]
#     nearest=np.argsort(distances)
#     测试点分类
#     topk_y=[y_train[i] for i in nearest[:k]]
#     votes=Counter(topk_y)
#     return votes.most_common(1)[0][0]

# x_train=np.array([[1,3],[2,4],[3,8],[4,9],[5,1],[7,8],[8,5],[9,10],[10,4],[11,5]])
# y_train=np.array([0,0,0,0,0,1,1,1,1,1])

# x=np.array([8.2,3.3])# 一维数组shape[0]是个数
# KNN_classify(6,x_train,y_train,x)
#knn是不需要训练过程的算法，但是与其他算法统一，所以有fit

#使用sklearn中的knn
# from sklearn.neighbors import KNeighborsClassifier
# knn=KNeighborsClassifier(n_neighbors=6)
# knn.fit(x_train,y_train)
# y_predict=knn.predict(x.reshape[1,-1]) #返回数组；新版所有传入数据比为二维数组
# y_predict[0]

#封装knn代码
import numpy as np
from math import sqrt
from collections import Counter
from PLAY_KNN.metrics import accuracy_score
class KNNClassifier:
    def __init__(self,k):
        assert k>=1,'k must be valid'
        self.k=k
        self._x_train=None
        self._y_train=None

    def fit(self,x_train,y_train):
        assert self.k <= x_train.shape[0], 'k must be valid'
        assert x_train.shape[0] == y_train.shape[0], \
            'the size of x be equal to the size of y_train'

        self._x_train=x_train
        self._y_train=y_train
        return self

    def predict(self,x_test):
        assert self._x_train is not None and self._y_train is not None,\
        'must fit before predict'
        assert x_test.shape[1]==self._x_train.shape[1],\
        'the feature number of x_predict must be equal to x_train'

        y_predict=[self._predict(x) for x in x_test]
        return np.array(y_predict)

    def _predict(self,x):
        assert x.shape[0]==self._x_train.shape[1],\
        'the feature number of x must be equal to x_train'

        distances=[sqrt(np.sum((x_train-x)**2)) for x_train in self._x_train]
        nearest=np.argsort(distances)
        topk_y=[self._y_train[i] for i in nearest[:self.k]]
        votes=Counter(topk_y)
        return votes.most_common(1)[0][0]
    def score(self,x_test,y_test):
        y_predict=self.predict(x_test)
        return accuracy_score(y_test,y_predict)

    def __repr__(self):
        #打印对象时调用或ipython解释器输入对象调用；__str__只有打印对象时调用
        return 'knn(k=%d)'%self.k

# if __name__ == '__main__':
#     from sklearn import datasets
#     iris=datasets.load_iris()
#     x=iris.data
#     y=iris.target
#     from PLAY_KNN.model_selection import train_test_split
#     x_train,x_test,y_train,y_test=train_test_split(x,y)
#
#     knn=KNNClassifier(k=3)
#     knn.fit(x_train,y_train)
#     print(knn.predict(x_test))
