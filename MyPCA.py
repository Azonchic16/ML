import numpy as np
import pandas as pd


class MyPCA():
    
    def __init__(self, n_components=3):
        self.n_components = n_components
        
    def __str__(self):
        return f'MyPCA class: n_components={self.n_components}'
    
    def fit_transform(self, X):
        X = X.transform(lambda x: x - x.mean())
        cov = np.cov(X.T)
        l, v = np.linalg.eigh(cov)
        v = v.transpose()
        l = pd.Series(l)
        l = l.sort_values(ascending=False)
        l = l[:self.n_components].index
        w = np.zeros((len(l), len(X.columns)))
        for i in range(len(l)):
            w[i,:] = v[l[i],:]
        w = w.transpose()
        X_reduced = np.dot(X, w)
        df = pd.DataFrame(data=X_reduced)
        return df
        