import pandas as pd
import numpy as np
import random


class MySVM():

    def __init__(self, n_iter=10, learning_rate=0.001, C=1, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.weights = None
        self.b = None

    def __str__(self):
        return f'MySVM class: ' + ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
    
    def fit(self, X, y):
        y = np.where(y == 0, -1, 1)
        self.N = X.shape[0]
        self.weights = np.ones(X.shape[1])
        self.b = 1
        random.seed(self.random_state)
        for epoch in range(self.n_iter):
            sample_index = self.sample(X)
            if sample_index:
                X_learn = X.iloc[sample_index]
            else:
                X_learn = X
            for ind, row in X_learn.iterrows():
                flag_for_grad = int(y[ind] * (self.weights @ np.array(row) + self.b) < 1)
                grad_w =  2 * self.weights - y[ind] * np.array(row) * self.C * flag_for_grad
                grad_b = -y[ind] * flag_for_grad  * self.C
                self.weights -= self.learning_rate * grad_w
                self.b -= self.learning_rate * grad_b

    def loss(self, y, x):
        cond = 1 - y(self.weights @ x + self.b)
        for_sum = np.where(cond > 0, cond, 0)
        return self.weights ** 2 + np.sum(for_sum) / self.N

    def get_coef(self):
        return self.weights, self.b
    
    def predict(self, X):
        pred = np.sign(X @ self.weights + self.b)
        pred = np.where(pred == -1, 0, pred)
        return [int(i) for i in pred]

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
