import numpy as np
import pandas as pd

class MyKNNClf():
    
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.train_size = None
        self.features = None
        self.target = None
        self.metric = metric
        self.weight = weight

    def __str__(self):
        return f'MyKNNClf k={self.k}' 
        
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
            distances = sorted(distances.items())
            classes = np.array([i[1] for i in distances[:self.k]])
            dists = np.array([i[0] for i in distances[:self.k]])
            self.weights_predict(classes, dists, y_pred)
        return np.array(y_pred)

    def predict_proba(self, Y):
        prb_lst = []
        M = len(Y)
        N = len(self.features)
        for i in range(M):
            Y_features = np.array(Y.iloc[i])
            distances = dict()
            for j in range(N):
                X_features = np.array(self.features.iloc[j])
                dist = self.metrics(X_features, Y_features)
                distances[dist] = self.target[j]
            distances = sorted(distances.items())
            classes = [i[1] for i in distances[:self.k]]
            dists = np.array([i[0] for i in distances[:self.k]])
            self.weights_predict_proba(classes, dists, prb_lst)
        return np.array(prb_lst)
    
    def metrics(self, y1, y2):
        if self.metric == 'euclidean':
            return np.sqrt(sum((y1 - y2)**2))
        elif self.metric == 'chebyshev':
            return np.max(np.abs(y1 - y2))
        elif self.metric == 'manhattan':
            return sum(np.abs(y1 - y2))
        elif self.metric == 'cosine':
            return 1 - sum((y1 * y2)) / (np.sqrt(sum(y1**2)) * np.sqrt(sum(y2**2)))
        
    def weights_predict_proba(self, classes, dists, pred):
        if self.weight == 'uniform':
                pred.append(classes.count(1) / len(classes))
        elif self.weight == 'rank':
            rank_sum = sum(1 / np.arange(1, self.k + 1))
            #Q_0 = sum([1 / (i + 1) for i in range(len(classes)) if classes[i] == 0]) / rank_sum
            Q_1 = sum([1 / (i + 1) for i in range(len(classes)) if classes[i] == 1]) / rank_sum
            pred.append(Q_1)
        else:
            dist_sum = sum(1 / dists)
            #Q_0 = sum([1 / dists[i] for i in range(len(dists)) if classes[i] == 0]) / dist_sum
            Q_1 = sum([1 / dists[i] for i in range(len(dists)) if classes[i] == 1]) / dist_sum
            pred.append(Q_1)
            
    def weights_predict(self, classes, dists, pred):
        if self.weight == 'uniform':
            number_of_ones = len(classes[classes == 1])
            number_of_zeros = len(classes) - number_of_ones
            if number_of_ones >= number_of_zeros:
                pred.append(1)
            else:
                pred.append(0)
        elif self.weight == 'rank':
            rank_sum = sum(1 / np.arange(1, self.k + 1))
            Q_0 = sum([1 / (i + 1) for i in range(len(classes)) if classes[i] == 0]) / rank_sum
            Q_1 = sum([1 / (i + 1) for i in range(len(classes)) if classes[i] == 1]) / rank_sum
            pred.append(0 + 1 * (Q_1 > Q_0))
        else:
            dist_sum = sum(1 / dists)
            Q_0 = sum([1 / dists[i] for i in range(len(dists)) if classes[i] == 0]) / dist_sum
            Q_1 = sum([1 / dists[i] for i in range(len(dists)) if classes[i] == 1]) / dist_sum
            pred.append(0 + 1 * (Q_1 > Q_0))
