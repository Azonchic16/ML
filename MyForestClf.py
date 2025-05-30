import pandas as pd
import numpy as np
import random
from MyTreeClf import MyTreeClf


class MyForestClf():

    def __init__(self, n_estimators=10, max_features=0.5, max_samples=0.5, random_state=42, max_depth=5, min_samples_split=2, max_leafs=20, bins=16,
                 criterion='entropy', oob_score=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion
        self.random_state = random_state
        self.oob_score = oob_score
        self.leafs_cnt = 0
        self.trees = [0] * n_estimators
        self.fi = {}
        self.oob_score_ = None
        self.update_metric()

    def __str__(self):
        return 'MyForestClf class: ' + ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
    
    def fit(self, X, y):
        X.reset_index(inplace=True)
        del X['index']
        self.cols = X.columns.values.tolist()
        self.cols_count = len(self.cols)
        self.N = X.shape[0]
        random.seed(self.random_state)
        self.fi.update({col: 0 for col in self.cols})
        y_oob_lst = np.full((self.N, self.n_estimators), np.nan)
        pred_oob_lst = np.full((self.N, self.n_estimators), np.nan)
        for tree in range(self.n_estimators):
            cols_learn = random.sample(self.cols, round(self.cols_count * self.max_features))
            rows_idx = random.sample(range(self.N), round(self.N * self.max_samples))
            rows_idx_oob = [i for i in range(self.N) if i not in rows_idx]
            X_learn = X[cols_learn]
            X_oob = X[cols_learn]
            X_learn = X_learn.iloc[rows_idx]
            X_oob = X_oob.iloc[rows_idx_oob]
            y_learn = y.iloc[rows_idx]
            y_oob = np.full(self.N, np.nan)
            y_oob[rows_idx_oob] = y.iloc[rows_idx_oob]
            y_oob_lst[:, tree] = y_oob
            model = MyTreeClf(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_leafs=self.max_leafs, bins=self.bins,
                              criterion=self.criterion)
            model.fit(X_learn, y_learn, N=self.N)
            self.trees[tree] = model
            self.leafs_cnt += model.leafs_cnt
            pred_oob = np.full(self.N, np.nan)
            pred_oob[rows_idx_oob] = model.predict_proba(X_oob)
            pred_oob_lst[:, tree] = pred_oob
        pred_oob_lst = pred_oob_lst[~np.all(np.isnan(pred_oob_lst), axis=1)]
        y_oob_lst = y_oob_lst[~np.all(np.isnan(y_oob_lst), axis=1)]
        pred_oob_mean = np.nanmean(pred_oob_lst, axis=1)
        y_oob_mean = np.nanmean(y_oob_lst, axis=1)
        if self.metric == self.roc_auc:
            self.oob_score_ = self.metric(pred_oob_mean, y_oob_mean)
        else:
            self.oob_score_ = self.metric(np.where(pred_oob_mean > 0.5, 1, 0), y_oob_mean)
        self.result_feature_importance()

    def predict(self, X, type):
        pred = []
        for ind in X.index.values:
            row = X.loc[ind]
            pred_every_tree = []
            for tree in self.trees:
                pred_every_tree.append(tree.tree.find_proba(row))
            if type == 'mean':
                mean_every_tree = np.array(pred_every_tree).mean()
                pred.append(1 if mean_every_tree > 0.5 else 0)
            else:
                class_every_tree = np.where(np.array(pred_every_tree) > 0.5, 1, 0)
                num_one = np.count_nonzero(class_every_tree)
                num_zero = len(class_every_tree) - num_one
                pred.append(1 if num_one >= num_zero else 0)
        return np.array(pred)
    
    def predict_proba(self, X):
        pred = []
        for ind in X.index.values:
            row = X.loc[ind]
            pred_every_tree = []
            for tree in self.trees:
                pred_every_tree.append(tree.tree.find_proba(row))
            pred.append(np.array(pred_every_tree).mean())
        return np.array(pred)
    
    def result_feature_importance(self):
        fi_new = pd.DataFrame([tree.fi for tree in self.trees]).sum().to_dict()
        fi_new = {key: value / 2 for key, value in fi_new.items()} # почему-то в курсе надо было в 2 раза меньше, либо ошибка в коде
        self.fi.update(fi_new)

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