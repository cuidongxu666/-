#sklearn中的train_test_split
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y)

# 自己写train_test_split
import numpy as np
def train_test_split(x,y,test_radio=0.2,random_seed=None):
    assert x.shape[0]==y.shape[0],\
    'the size of x must be equal tp the size of y'
    assert 0.0<= test_radio<=1.0,\
    'test_ratio must be valid'

    if random_seed:
        np.random.seed(random_seed)
    # 索引随机排列
    shuffle_indexes=np.random.permutation(len(x))
    test_size=int(len(x)*test_radio)
    test_indexes=shuffle_indexes[:test_size]
    train_indexes=shuffle_indexes[test_size:]

    x_train=x[train_indexes]
    y_train=y[train_indexes]
    x_test=x[test_indexes]
    y_test=y[test_indexes]

    return x_train,x_test,y_train,y_test