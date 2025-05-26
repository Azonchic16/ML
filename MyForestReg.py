import pandas as pd
import numpy as np
import random
from MyTreeReg import MyTreeReg
from sklearn.datasets import make_regression


class MyForestReg():

    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5, random_state=42, max_depth=5, min_samples_split=2, max_leafs=20, bins=16):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.random_state = random_state
        self.leafs_cnt = 0
        self.trees = [0] * n_estimators

    def __str__(self):
        return' MyForestReg class: ' + ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
    
    def fit(self, X, y):
        # X['target'] = y
        # y.reset_index(inplace=True)
        X.reset_index(inplace=True)
        del X['index']
        self.cols = X.columns.values.tolist()
        self.cols_count = len(self.cols)
        self.N = X.shape[0]
        random.seed(self.random_state)
        for tree in range(self.n_estimators):
            cols_learn = random.sample(self.cols, round(self.cols_count * self.max_features))
            print(cols_learn)
            rows_idx = random.sample(range(self.N), round(self.N * self.max_samples))
            X_learn = X[cols_learn]
            X_learn = X_learn.iloc[rows_idx]
            y_learn  = y.iloc[rows_idx]
            model = MyTreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_leafs=self.max_leafs, bins=self.bins)
            model.fit(X_learn, y_learn)
            self.trees[tree] = model
            self.leafs_cnt += model.leafs_cnt

if __name__ == '__main__':
    d = {"n_estimators": 5, "max_depth": 4, "max_features": 0.4, "max_samples": 0.3}
    X, y = make_regression(n_samples=150, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X).round(2)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]
    model = MyForestReg(n_estimators=5, max_depth=4, max_features=0.4, max_samples=0.3)
    model.fit(X, y)
    print(f'{model.leafs_cnt} - всего листов')
    print(model.trees[0].tree.print_tree())
    # print([tree.leafs_cnt for tree in model.trees])

