import numpy as np
import pandas as pd
import random


class MyLineReg():
    
    def __init__(self, weights=None, n_iter=1000, learning_rate=0.000001, metric=None, reg=None,
                l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):            
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.dynamic_l_r = not(isinstance(self.learning_rate, float))
        self.weights = weights
        self.pred = None
        self.metric = metric
        self.metric_value = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        y = np.array(y)
        n = len(X)
        m = len(X.columns)
        X.insert(0, 'x0', [1] * len(X))
        self.weights = np.ones(m + 1)
        alpha = self.dynamic_learning_rate()
#         random.seed(self.random_state)
#         X = X.reset_index()
#         del X['index']
        for i in range(1, self.n_iter + 1):
            y_pred = np.dot(X, self.weights)
            MSE = (1 / n) * sum((y_pred - y) ** 2) + self.regularization()[0]
            sample_index = self.sample(X)
            if sample_index is None:
                grad = (2 / n) * np.dot((y_pred - y), X) + self.regularization()[1]
                self.weights = self.weights - alpha(i) * grad
            else:
                self.sgd(sample_index, X, y, alpha, i)
        self.metric_value = self.metrics(X, y)
        
            
    def get_coef(self):
        return self.weights
            
    def __str__(self):
        return np.sum(self.weights[1:])
    
    def predict(self, X):
        X.insert(0, 'x0', [1] * len(X))
        self.pred = np.dot(X, self.weights)
        return self.pred
    
    def metrics(self, X: pd.DataFrame, y: pd.Series):
        n = len(X)
        y = np.array(y)
        y_pred = np.dot(X, self.weights)
        sum1 = sum(np.abs(y_pred - y))
        sum2 = sum((y_pred - y) ** 2)
        mae = (1 / n) * sum1
        mse = (1 / n) * sum2
        rmse = np.sqrt((1 / n) * sum2)
        mape = (100 / n) * sum(np.abs((y_pred - y) / y))
        r2 = 1 - \
        (sum2) / (sum((y - np.mean(y))**2))
        metrics = {'mae': mae, 'mse': mse, 'rmse': rmse, 
                   'mape': mape, 
                   'r2': r2, 
                   None: None}
        return metrics[self.metric]
    
    def get_best_score(self):
        return self.metric_value
    
    def dynamic_learning_rate(self):
        if self.dynamic_l_r:
            return self.learning_rate
        else:
            a = lambda i: self.learning_rate
            return a
        
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
        y_pred = np.dot(X_sample, self.weights)
        y1 = [y[k] for k in sample_index]
        grad = (2 / n1) * np.dot((y_pred - np.array(y1)), X_sample) + self.regularization()[1]
        self.weights = self.weights - alpha(i) * grad