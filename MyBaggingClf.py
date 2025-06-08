import pandas as pd
import numpy as np
import random
from MyKNNClf import MyKNNClf
from MyLogReg import MyLogReg
from MyTreeClf import MyTreeClf
import copy


class MyBaggingClf():

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
        return f'MyBaggingClf class: ' + ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
    
    def fit(self, X, y):
        X.reset_index(inplace=True, drop=True)
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
            pred_oob[rows_idx_oob] = model.predict_proba(X_oob)
            pred_oob_lst[:, i] = pred_oob
        pred_oob_lst = pred_oob_lst[~np.all(np.isnan(pred_oob_lst), axis=1)]
        y_oob_lst = y_oob_lst[~np.all(np.isnan(y_oob_lst), axis=1)]
        pred_oob_mean = np.nanmean(pred_oob_lst, axis=1)
        y_oob_mean = np.nanmean(y_oob_lst, axis=1)
        if self.metric == self.roc_auc:
            self.oob_score_ = self.metric(pred_oob_mean, y_oob_mean)
        else:
            self.oob_score_ = self.metric(np.where(pred_oob_mean > 0.5, 1, 0), y_oob_mean)
    
    def predict(self, X, type):
        pred = np.full((X.shape[0], self.n_estimators), np.nan)
        for i, model in enumerate(self.estimators):
            pred[:,i] = model.predict_proba(X)
        if type == 'mean':
            res_pred = np.where(pred.mean(axis=1) > 0.5, 1, 0)
        else:
            pred = np.where(pred > 0.5, 1, 0)
            count_ones = np.count_nonzero(pred, axis=1)
            count_zeros = pred.shape[1] - count_ones
            res_pred = np.where(count_ones >= count_zeros, 1, 0)
        return res_pred
    
    def predict_proba(self, X):
        pred = np.full((X.shape[0], self.n_estimators), np.nan)
        for i, model in enumerate(self.estimators):
            pred[:,i] = model.predict_proba(X)
        return pred.mean(axis=1)
    
    def accuracy(self, prb, y):
        TP = np.count_nonzero((prb == 1) & (prb == y))
        TN = np.count_nonzero((prb == 0) & (prb == y))
        FN = np.count_nonzero((prb == 0) & (prb != y))
        FP = np.count_nonzero((prb == 1) & (prb != y))
        return (TP + TN) / (TP + TN + FN + FP)
    
    def precision(self, prb, y):
        TP = np.count_nonzero((prb == 1) & (prb == y))
        FP = np.count_nonzero((prb == 1) & (prb != y))
        return TP / (TP + FP)   
         
    def recall(self, prb, y):
        TP = np.count_nonzero((prb == 1) & (prb == y))
        FN = np.count_nonzero((prb == 0) & (prb != y))
        return TP / (TP + FN)

    def f1(self, prb, y):
        return (2 * self.precision(prb, y) * self.recall(prb, y)) / (self.precision(prb, y) + self.recall(prb, y))
    
    def roc_auc(self, prb, y):
        prb = np.round(prb, 10)
        ind_prob = lambda p1, p2: 1.0 if p1 < p2 else 0.0 if p1 > p2 else 0.5
        ind_labels = lambda l1, l2: 1 if l1 < l2 else 0

        scores, labels = zip(*sorted(zip(prb, y), key=lambda x: x[0], reverse=True))

        l = y.shape[0]
        num_p = sum(y == 1)
        nn = l - num_p

        roc_auc = 0

        for i in range(l):
            for j in range(l):
                indl = ind_labels(labels[i], labels[j])
                indpr = ind_prob(scores[i], scores[j])
                roc_auc += indpr * indl
        return roc_auc / (num_p * nn)
    
    def update_metric(self):
        if not self.oob_score or self.oob_score == 'accuracy':
            self.metric = self.accuracy
        elif self.oob_score == 'precision':
            self.metric = self.precision
        elif self.oob_score == 'recall':
            self.metric = self.recall
        elif self.oob_score == 'f1':
            self.metric = self.f1
        elif self.oob_score == 'roc_auc':
            self.metric = self.roc_auc