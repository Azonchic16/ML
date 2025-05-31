import pandas as pd
import numpy as np

class MyKNNReg:
    
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.train_size = None
        self.features = None
        self.target = None
        self.metric = metric
        self.weight =  weight
        
    def __str__(self):
        return f'MyKNNReg class: k={self.k}'
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.features = X
        self.target = y
        self.train_size = (len(X), len(X.columns))
        
    def predict(self, Y):
        y_pred = []
        M = len(Y)
        N = len(self.features)
        for i in range(M):
            Y_features = np.array(Y.iloc[i])
            distances = dict()
            for j in range(N):
                X_features = np.array(self.features.iloc[j])
                dist = self.metrics(X_features, Y_features)
                distances[dist] = self.target[j]
            dist_sort = sorted(distances.items())
            target_k = np.array([dist_sort[i][1] for i in range(self.k)])
            if self.weight == 'uniform':
                y_pred.append(target_k.mean())
            elif self.weight == 'distance':
                dists = np.array([dist_sort[i][0] for i in range(self.k)])
                weights = (1 / dists) / sum(1 / dists)
                y_pred.append(sum(target_k * weights))
            else:
                rank = np.arange(1, self.k + 1)
                weights = (1 / rank) / (sum(1 / rank))
                y_pred.append(sum(target_k * weights))
        return np.array(y_pred)
        
    def metrics(self, y1, y2):
        if self.metric == 'euclidean':
            return np.sqrt(sum((y1 - y2)**2))
        elif self.metric == 'chebyshev':
            return np.max(np.abs(y1 - y2))
        elif self.metric == 'manhattan':
            return sum(np.abs(y1 - y2))
        elif self.metric == 'cosine':
            return 1 - sum((y1 * y2)) / (np.sqrt(sum(y1**2)) * np.sqrt(sum(y2**2)))
        