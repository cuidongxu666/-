import numpy as np
class StandardScaler:
    def __init__(self):
        self.mean_=None
        self.std_=None

    def fit(self,x):
        assert x.ndim==2,'the dimension of x must be 2'
        self.mean_=np.mean(x,axis=0)
        self.std_=np.std(x,axis=0)
        return self

    def tranform(self,x):
        assert x.ndim==2,'the dimension of x must be 2'
        assert self.mean_ is not None and self.std_ is not None,\
            'must fit before transform'
        assert x.shape[1]==len(self.mean_) and x.shape[1]==len(self.mean_)

        resx=(x-self.mean_)/self.std_
        return resx


