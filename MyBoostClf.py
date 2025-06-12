import pandas as pd
import numpy as np
from MyTreeReg import MyTreeReg


class MyBoostClf():

    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=5, min_samples_split=2, max_leafs=20, bins=16):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.eps = 1e-15
        self.sum_preds = 0
        self.trees = []
    
    def __str__(self):
        return f'MyBoostClf class: ' + ', '.join(f'{key}={value}' for key, value in self.__dict__.items())

    def fit(self, X, y):
        self.cols = X.columns.values.tolist()
        self.cols_count = len(self.cols)
        self.N = X.shape[0]
        self.pred_0 = np.log(y.mean() / (1 - y.mean()) + self.eps)
        self.sum_preds += np.full(self.N, self.pred_0, dtype=np.float64)
        for i in range(self.n_estimators):
            error_target = y - self.prob_from_odds(self.sum_preds)
            model = MyTreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_leafs=self.max_leafs, bins=self.bins)
            model.fit(X, error_target)
            self.change_leafs_values(y, model.leafs)
            self.sum_preds += self.learning_rate * model.predict(X)
            self.trees.append(model)
    
    def prob_from_odds(self, log_odds):
        return np.exp(log_odds) / (1 + np.exp(log_odds))
    
    def change_leafs_values(self, y, leafs):
        for leaf in leafs:
            leaf_obj = np.array(y.loc[leaf.value[2]])
            leaf_pred = self.prob_from_odds(self.sum_preds[leaf.value[2]])
            new_value_leaf = np.sum(leaf_obj - leaf_pred) / np.sum(leaf_pred * (1 - leaf_pred))
            leaf.value[1] = new_value_leaf

    def predict_proba(self, X):
        res = self.prob_from_odds(self.pred_0)
        for i, tree in enumerate(self.trees):
            res += self.learning_rate * tree.predict(X)
        return res
    
    def predict(self, X):
        res = self.prob_from_odds(self.pred_0)
        for i, tree in enumerate(self.trees):
            res += self.learning_rate * tree.predict(X)
        res = np.where(res > 0.5, 1, 0)
        return res