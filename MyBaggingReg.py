import pandas as pd
import numpy as np
import random
from MyKNNReg import MyKNNReg
from MyLineReg import MyLineReg
from MyTreeReg import MyTreeReg
import copy


class MyBaggingReg():

    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0, random_state=42, oob_score=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.oob_score = oob_score
        self.estimators = []
        self.oob_score_ = None
        self.update_metric()

    def __str__(self):
        return f'MyBaggingReg class: ' + ', '.join(f'{key}={value}' for key, value in self.__dict__.items())

    def fit(self, X, y):
        X.reset_index(inplace=True)
        del X['index']
        self.N = X.shape[0]
        rows_num_list = X.index.values.tolist()
        random_num = []
        random.seed(self.random_state)
        y_oob_lst = np.full((self.N, self.n_estimators), np.nan)
        pred_oob_lst = np.full((self.N, self.n_estimators), np.nan)
        for _ in range(self.n_estimators):
            sample_rows_idx = random.choices(rows_num_list, k=round(self.N * self.max_samples))
            random_num.append(sample_rows_idx)
        for i, rows in enumerate(random_num):
            X_learn = X.iloc[rows]
            y_learn = y.iloc[rows]
            X_learn.reset_index(inplace=True, drop=True)
            y_learn.reset_index(inplace=True, drop=True)
            model = copy.deepcopy(self.estimator)
            model.fit(X_learn, y_learn)
            self.estimators.append(model)
            rows_idx_oob = [i for i in range(self.N) if i not in rows]
            X_oob = X.iloc[rows_idx_oob]
            y_oob = np.full(self.N, np.nan)
            y_oob[rows_idx_oob] = y.iloc[rows_idx_oob]
            y_oob_lst[:, i] = y_oob
            X_oob.reset_index(inplace=True, drop=True)
            pred_oob = np.full(self.N, np.nan)
            pred_oob[rows_idx_oob] = model.predict(X_oob)
            pred_oob_lst[:, i] = pred_oob
        pred_oob_lst = pred_oob_lst[~np.all(np.isnan(pred_oob_lst), axis=1)]
        pred_oob_mean = np.nanmean(pred_oob_lst, axis=1)
        pred_oob_mean = pred_oob_mean[~np.isnan(pred_oob_mean)]
        y_oob_lst = y_oob_lst[~np.all(np.isnan(y_oob_lst), axis=1)]
        y_oob_mean = np.nanmean(y_oob_lst, axis=1)
        y_oob_mean = y_oob_mean[~np.isnan(y_oob_mean)]
        self.oob_score_ = self.metric(y_oob_mean, pred_oob_mean)

    def predict(self, X):
        pred = np.full((X.shape[0], self.n_estimators), np.nan)
        for i, model in enumerate(self.estimators):
            pred[:,i] = model.predict(X)
        return pred.mean(axis=1)
    
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
        return sum((y_pred - y) ** 2) / len(y)
    
    def rmse(self, y, y_pred):
        return np.sqrt(sum((y_pred - y) ** 2) / len(y))
    
    def mape(self, y, y_pred):
        return (100 / len(y)) * sum(np.abs((y_pred - y) / y))
    
    def r2(self, y, y_pred):
        return 1 - (sum((y_pred - y) ** 2)) / (sum((y - np.mean(y))**2))
