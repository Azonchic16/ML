import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from Tree import Node, Tree


class MyTreeReg():
    
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.depth = 0
        self.leafs_sum = 0
        self.potential_leafs = 1
        self.tree = Tree()
        self.curr = None
        self.bins = bins
        self.split_bins = None
        self.fi = {}

    def __str__(self):
        attrs = vars(self)
        return f'MyTreeClf class: ' + ', '.join(f'{key}={value}' for key, value in attrs.items())
    
    def loss(self, N, y):
        return  np.sum((y - y.mean()) ** 2) / N
    
    def get_best_split_feature_mean(self, N, S0, X_numpy, y):
        split_value = None
        ig = None
        for i in range(len(self.names)):
            feature = pd.DataFrame({i: X_numpy[:,i], 'y': y})
            unique = feature[i].unique()
            unique = np.sort(unique)
            if len(unique) == 1:
                col_name = self.names[i]
                split_value = unique[0]
                return col_name, split_value
            feature_unique_mean = (unique[:-1] + unique[1:]) / 2
            for k in feature_unique_mean:
                if split_value is None: 
                    split_value = k
                    col_name = self.names[i]
                left_df = feature.loc[feature[i] <= k]
                right_df = feature.loc[feature[i] > k]
                left = left_df['y']
                right = right_df['y']
                N1 = len(left)
                N2 = len(right)
                S1 = self.loss(N1, left)
                S2 = self.loss(N2, right)
                ig_new = S0 - (N1 / N) * S1 - (N2 / N) * S2
                if ig is None:
                    ig = ig_new
                elif ig_new > ig:
                    split_value = k
                    col_name = self.names[i]
                    ig = ig_new
                # elif ig == ig_new:
                #     split_value = k
                #     col_name = max(col_name, self.names[i])
        return col_name, split_value
    
    def get_best_split_bins(self, N, S0, X_numpy, y):
        col_name = None
        split_value = None
        ig = None
        for i in range(len(self.names)):
            feature = pd.DataFrame({i: X_numpy[:,i], 'y': y})
            min_feature, max_feature = min(feature[i]), max(feature[i])
            splits = self.split_bins[self.names[i]]
            needed_splits = splits[(splits >= min_feature) & (splits < max_feature)]
            if not len(needed_splits) and not col_name:
                col_name = None
                split_value = None
            for k in needed_splits:
                if split_value is None: 
                    split_value = k
                    col_name = self.names[i]
                left_df = feature.loc[feature[i] <= k]
                right_df = feature.loc[feature[i] > k]
                left = left_df['y']
                right = right_df['y']
                N1 = len(left)
                N2 = len(right)
                S1 = self.loss(N1, left)
                S2 = self.loss(N2, right)
                ig_new = S0 - (N1 / N) * S1 - (N2 / N) * S2
                if ig is None:
                    ig = ig_new
                elif ig_new > ig:
                    split_value = k
                    col_name = self.names[i]
                    ig = ig_new
                # elif ig == ig_new:
                #     split_value = k
                #     col_name = max(col_name, self.names[i])
        return col_name, split_value
    
    def get_best_split(self, X, y):
        N = len(y)
        S0 = self.loss(N, y)
        X_numpy = X.to_numpy()
        if not self.split_bins:
            col_name, split_value = self.get_best_split_feature_mean(N, S0, X_numpy, y)
        else:
            col_name, split_value = self.get_best_split_bins(N, S0, X_numpy, y)
        return col_name, split_value
    
    def get_split_values_or_bins(self, X):
        if self.bins:
            self.split_bins = {name: None for name in X.columns.values}
            for col in self.split_bins:
                feature = X[col]
                unique = np.sort(feature.unique())
                split_mean = (unique[:-1] + unique[1:]) / 2
                split_hist = np.histogram(feature, bins=self.bins)[1][1:-1]
                if len(split_mean) <= len(split_hist) - 1:
                    self.split_bins[col] = split_mean
                else:
                    self.split_bins[col] = split_hist

    def build_tree(self, X):
        if self.leafs_cnt >= self.max_leafs or self.depth > self.max_depth:
            return
        if not self.tree.root:
            y = self.y.loc[X.index]
            col_name, split_value = self.get_best_split(X, y)
            self.tree.root = Node((col_name, split_value))
            self.tree.root.df = X
            self.curr = self.tree.root
            X_left = X.loc[X[col_name] <= split_value]
            X_right = X.loc[X[col_name] > split_value]
            self.depth += 1
            self.potential_leafs += 1
            self.fi[col_name] += self.compute_feature_importance(X, X_left, X_right)
            self.build_tree(X_left)
        if not self.curr.left:
            y = self.y.loc[X.index]
            col_name, split_value = self.get_best_split(X, y)
            if not self.is_leaf(y) and col_name: # узел
                self.curr.left = Node((col_name, split_value))
                self.curr.left.parent = self.curr
                self.curr.left.df = X
                self.curr = self.curr.left
                X_left = X.loc[X[col_name] <= split_value]
                X_right = X.loc[X[col_name] > split_value]
                self.fi[col_name] += self.compute_feature_importance(X, X_left, X_right)
                self.depth += 1
                self.potential_leafs += 1
                self.build_tree(X_left)
            else: #  лист
                self.curr.left = Node(('leaf_left', y.mean()))
                self.leafs_sum += self.curr.left.value[1]
                self.curr.left.is_leaf = True
                self.curr.left.parent = self.curr
                self.leafs_cnt += 1
                self.potential_leafs -= 1
                self.build_tree(self.curr.df)
        elif self.curr.left and not self.curr.right:
            X_curr = self.curr.df.loc[self.curr.df[self.curr.value[0]] > self.curr.value[1]]
            y_r = self.y.loc[X_curr.index]
            col_name, split_value = self.get_best_split(X_curr, y_r)
            if not self.is_leaf(y_r) and col_name:
                self.curr.right = Node((col_name, split_value))
                self.curr.right.df = X_curr
                self.curr.right.parent = self.curr
                self.curr = self.curr.right
                self.depth += 1
                self.potential_leafs += 1
                X_left = X_curr.loc[X_curr[col_name] <= split_value]
                X_right = X_curr.loc[X_curr[col_name] > split_value]
                self.fi[col_name] += self.compute_feature_importance(X_curr, X_left, X_right)
                self.build_tree(X_left)
            else:
                self.curr.right = Node(('leaf_right', y_r.mean()))
                self.leafs_sum += self.curr.right.value[1]
                self.curr.right.is_leaf = True
                self.curr.right.parent = self.curr
                if self.tree.root != self.curr:
                    self.curr = self.curr.parent
                self.depth -= 1
                self.leafs_cnt += 1
                self.potential_leafs -= 1
                self.build_tree(self.curr.df)
        elif self.curr == self.tree.root and self.curr.left and self.curr.right:
            return
        elif self.curr.left and self.curr.right:
            self.curr = self.curr.parent
            self.depth -= 1
            self.build_tree(self.curr.df)

    def fit(self, X, y, N=0):
        '''N - параметр передается при обучениие случайного леса (количество строк в исходном датасете)'''
        if N:
            self.N = N
        self.y = y
        self.N = len(X)
        self.names = X.columns.values.tolist()
        self.get_split_values_or_bins(X)
        self.fi.update({col: 0 for col in self.names})
        self.build_tree(X)

    def is_leaf(self, y):
        '''True - лист, False - узел'''
        if len(y) == 1 or self.max_depth == self.depth or self.max_leafs <= 2 or \
            self.min_samples_split > len(y) or self.leafs_cnt + self.potential_leafs >= self.max_leafs:
            return True
        return False
    
    def predict(self, X):
        pred = []
        for ind, row in X.iterrows():
            # s = X.loc[ind]
            pred.append(self.tree.find_proba(row))
        return pred
    
    def feature_importance(self, N_p, N, N_l, N_r, I, I_l, I_r):
        return N_p * (I - N_l * I_l / N_p - N_r * I_r / N_p) / N
    
    def compute_feature_importance(self, X, X_left, X_right):
        left = self.y.loc[X_left.index]
        right = self.y.loc[X_right.index]
        y = self.y.loc[X.index]
        N = len(X)
        N_l = len(left)
        N_r = len(right)
        S = self.loss(N, y)
        S1 = self.loss(N_l, left)
        S2 = self.loss(N_r, right)
        return self.feature_importance(N, self.N, N_l, N_r, S, S1, S2)