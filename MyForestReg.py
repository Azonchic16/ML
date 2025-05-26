import pandas as pd
import numpy as np
import random
from MyTreeReg import MyTreeReg
from sklearn.datasets import make_regression


class MyForestReg():

    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5, random_state=42, max_depth=5, min_samples_split=2, max_leafs=20, bins=16,
                 oob_score=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.random_state = random_state
        self.oob_score = oob_score
        self.leafs_cnt = 0
        self.trees = [0] * n_estimators
        self.fi = {}
        self.oob_score_ = None
        self.update_metric()

    def __str__(self):
        return' MyForestReg class: ' + ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
    
    def fit(self, X, y):
        X.reset_index(inplace=True)
        del X['index']
        self.cols = X.columns.values.tolist()
        self.cols_count = len(self.cols)
        self.N = X.shape[0]
        random.seed(self.random_state)
        self.fi.update({col: 0 for col in self.cols})
        y_oob_lst = []
        pred_oob_lst = []
        for tree in range(self.n_estimators):
            cols_learn = random.sample(self.cols, round(self.cols_count * self.max_features))
            rows_idx = random.sample(range(self.N), round(self.N * self.max_samples))
            rows_idx_oob = [i for i in range(self.N) if i not in rows_idx]
            X_learn = X[cols_learn]
            X_oob = X[cols_learn]
            X_learn = X_learn.iloc[rows_idx]
            X_oob = X_oob.iloc[rows_idx_oob]
            y_learn = y.iloc[rows_idx]
            y_oob = y.iloc[rows_idx_oob]
            y_oob_lst.append(y_oob.mean())
            model = MyTreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_leafs=self.max_leafs, bins=self.bins)
            model.fit(X_learn, y_learn, N=self.N)
            self.trees[tree] = model
            self.leafs_cnt += model.leafs_cnt
            pred_oob = model.predict(X_oob)
            pred_oob_lst.append(np.array(pred_oob).mean())
        # y_mean_oob = np.array([np.array(y_oob_lst).mean()])
        # pred_mean_oob = np.array([np.array(pred_oob_lst).mean()])
        self.oob_score_ = self.metric(np.array(y_oob_lst), np.array(pred_oob_lst))
        self.result_feature_importance()

    def predict(self, X):
        pred = []
        for ind in X.index.values:
            row = X.loc[ind]
            pred_every_tree = []
            for tree in self.trees:
                pred_every_tree.append(tree.tree.find_proba(row))
            pred.append(np.array(pred_every_tree).mean())
        return pred
    
    def result_feature_importance(self):
        fi_new = pd.DataFrame([tree.fi for tree in self.trees]).sum().to_dict()
        fi_new = {key: value / 2 for key, value in fi_new.items()}
        self.fi.update(fi_new)

    def update_metric(self):
        if not self.oob_score or self.oob_score == 'mae':
            self.metric = self.mae
        elif self.oob_score == 'mse':
            self.metric = self.mse
        elif self.oob_score == 'rmse':
            self.metric = self.rmse
        elif self.oob_score == 'mape':
            self.metric = self.mape
        elif self.oob_score == 'r2':
            self.metric = self.r2

    def mae(self, y, y_pred): 
        return sum(np.abs(y_pred - y)) / len(y)
    
    def mse(self, y, y_pred):
        return sum((y_pred - y) ** 2) / self.N
    
    def rmse(self, y, y_pred):
        return np.sqrt(sum((y_pred - y) ** 2) / self.N)
    
    def mape(self, y, y_pred):
        return (100 / self.N) * sum(np.abs((y_pred - y) / y))
    
    def r2(self, y, y_pred):
        return 1 - (sum((y_pred - y) ** 2)) / (sum((y - np.mean(y))**2))
    

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

