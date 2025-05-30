import numpy as np
import pandas as pd
import random

class MyLogReg():
    
    def __init__(self, n_iter=10, learning_rate=0.01, weights=None, metric=None, reg=None,
                l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.dynamic_l_r = not(isinstance(self.learning_rate, float))
        self.weights = weights
        self.metric = metric
        self.metric_value = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        
    def __str__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def fit(self, X, y, verbose=False):
        n = len(X)
        y = np.array(y)
        m = len(X.columns)
        X.insert(0, 'x0', [1] * n)
        self.weights = np.ones(m + 1)
        eps = 1e-15
        alpha = self.dynamic_learning_rate()
        random.seed(self.random_state)
        X = X.reset_index()
        del X['index']
        for i in range(1, self.n_iter + 1):
            prb = 1 / (1 + np.exp(-np.dot(X, self.weights)))
            LogLoss = -(1 / n) * sum(y * np.log(prb + eps) + (1 - y) * np.log(1 - y + eps)) +\
                self.regularization()[0]
            sample_index = self.sample(X)
            if sample_index is None:
                grad = (1 / n) * np.dot((prb - y), X) + self.regularization()[1]
                self.weights = self.weights - alpha(i) * grad
            else:
                self.sgd(sample_index, X, y, alpha, i)
        self.metric_value = self.metrics(X, y)
        
    def get_coef(self):
        return np.mean(self.weights[1:])
    
    def predict_proba(self, X):
        if not(len(self.weights) == len(X.columns)):
            n = len(X)
            X.insert(0, 'x0', [1] * n)
        prb = 1 / (1 + np.exp(-np.dot(X, self.weights)))
        return np.mean(prb)
    
    def predict(self, X):
        if not(len(self.weights) == len(X.columns)):
            n = len(X)
            X.insert(0, 'x0', [1] * n)
        prb = 1 / (1 + np.exp(-np.dot(X, self.weights)))
        y_pred = self.classification(prb, 0.5)
        return int(sum(prb))
    
    def metrics(self, X: pd.DataFrame, y: pd.Series):
        y = np.array(y)
        n = len(X)
        if not(len(self.weights) == len(X.columns)):
            n = len(X)
            X.insert(0, 'x0', [1] * n)
        prb = 1 / (1 + np.exp(-np.dot(X, self.weights)))  # probabylyties
        y_pred = self.classification(prb, 0.5)
        if self.metric is None:
            return None
        elif self.metric  == 'roc_auc':
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
            return roc_auc / (nn * num_p)
        else:
            TP = np.count_nonzero((prb == 1) & (prb == y))
            TN = np.count_nonzero((prb == 0) & (prb == y))
            FN = np.count_nonzero((prb == 0) & (prb != y))
            FP = np.count_nonzero((prb == 1) & (prb != y))
            accuracy = (TP + TN) / (TP + TN + FN + FP)
            precision = TP / (TP + FP)
            TP = np.count_nonzero((prb == 1) & (prb == y))
            TN = np.count_nonzero((prb == 0) & (prb == y))
            FN = np.count_nonzero((prb == 0) & (prb != y))
            FP = np.count_nonzero((prb == 1) & (prb != y))
            accuracy = (TP + TN) / (TP + TN + FN + FP)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = (2 * precision * recall) / (precision + recall)
            metrics = {'accuracy': accuracy, 'precision': precision,
                   'recall': recall, 'f1': f1}
            return metrics[self.metric]
    
    def classification(self, prb, level):
        y_pred = np.copy(prb)
        y_pred[y_pred > level] = 1
        y_pred[y_pred <= level] = 0
        return y_pred
    
    def get_best_score(self):
        return self.metric_value
    
    def regularization(self):
        l1 = self.l1_coef * sum(np.abs(self.weights))
        l2 = self.l2_coef * sum((self.weights)**2)
        gr1 = self.l1_coef * np.sign(self.weights)
        gr2 = 2 * self.l2_coef * self.weights
        regul = {'l1': l1,
                 'l2': l2,
                 'elasticnet': l1 + l2,
                 None: 0}
        regul_grad = {'l1': gr1,
                      'l2': gr2,
                      'elasticnet': gr1 + gr2,
                      None: 0}
        return (regul[self.reg], regul_grad[self.reg])
    
    def dynamic_learning_rate(self):
        if self.dynamic_l_r:
            return self.learning_rate
        else:
            a = lambda i: self.learning_rate
            return a
        
    def sample(self, X):
        if self.sgd_sample is None:
            return None
        elif isinstance(self.sgd_sample, int):
            sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
            return sample_rows_idx
        else:
            n = len(X)
            number = int(self.sgd_sample * n)
            sample_rows_idx = random.sample(range(X.shape[0]), number)
            return sample_rows_idx
        
    def sgd(self, sample_index, X, y, alpha, i):
        X_sample = X.iloc[sample_index]
        n1 = len(X_sample)
        prb = 1 / (1 + np.exp(-np.dot(X_sample, self.weights)))
        y1 = [y[k] for k in sample_index]
        grad = (1 / n1) * np.dot((prb - np.array(y1)), X_sample) + self.regularization()[1]
        self.weights = self.weights - alpha(i) * grad