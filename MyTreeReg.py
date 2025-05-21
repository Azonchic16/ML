import pandas as pd
import numpy as np


class Node():

    def __init__(self, value=None):
        self.df = None
        self.parent = None
        self.value = value
        self.left = None
        self.right = None
        self.is_leaf = False


class Tree():

    def __init__(self):
        self.root = None

    def print_tree(self):
        """Печатает дерево в консоли, начиная с корня"""
        if not self.root:
            print("(пустое дерево)")
            return
        
        def _print(node, prefix="", is_left=True):
            if not node:
                return
            
            if node.is_leaf:
                label = f"{node.value[0]} = {node.value[1]}" if is_left else f"{node.value[0]} = {node.value[1]}"
            else:
                label = f"{node.value[0]} > {node.value[1]}"
            
            print(prefix + ("└── " if not prefix else "├── ") + label)
            
            new_prefix = prefix + ("    " if not prefix else "│   ")
            
            _print(node.left, new_prefix, True)
            _print(node.right, new_prefix, False)
        
        _print(self.root, "", False)
        
    def find_proba(self, s):
        def tree_traversal(node, s):
            if node.is_leaf:
                return node.value[1]
            else:
                feature, split_val = node.value[0], node.value[1]
                if s[feature] < split_val:
                    return tree_traversal(node.left, s)
                else:
                    return tree_traversal(node.right, s)
        return tree_traversal(self.root, s)
        

class MyTreeReg():
    
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, criterion='entropy'):
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
        self.criterion = criterion
        self.fi = {}

    def __str__(self):
        attrs = vars(self)
        return f'MyTreeReg class: ' + ', '.join(f'{key}={value}' for key, value in attrs.items())

    @staticmethod
    def custom_log(value):
        if value == 0.0:
            return 0
        return np.log2(value)
    
    def entropy_gini(self, N, count_nonzero):
        if self.criterion == 'entropy':
            return -(count_nonzero * MyTreeReg.custom_log(count_nonzero / N) +
                                (N - count_nonzero) * MyTreeReg.custom_log((N - count_nonzero) / N)) / N
        return 1 - (count_nonzero / N) ** 2 - ((N - count_nonzero) / N) ** 2
    
    def get_best_split(self, X, y):
        names = X.columns.values.tolist()
        N = len(y)
        count_nonzero = np.count_nonzero(y)
        d = pd.DataFrame(columns=['col_name', 'split_value', 'ig'])
        S0 = self.entropy_gini(N, count_nonzero)
        X_numpy = X.to_numpy()
        if not self.split_bins:
            for i in range(len(X.columns)):
                feature = pd.DataFrame({i: X_numpy[:,i], 'y': y})
                unique = feature[i].unique()
                unique = np.sort(unique)
                feature_unique_mean = (unique[:-1] + unique[1:]) / 2
                for k in range(len(feature_unique_mean)):
                    left_df = feature.loc[feature[i] <= feature_unique_mean[k]]
                    right_df = feature.loc[feature[i] > feature_unique_mean[k]]
                    left = left_df['y']
                    right = right_df['y']
                    N1 = len(left)
                    N2 = len(right)
                    c_n_l = int(np.count_nonzero(left))
                    c_n_r = int(np.count_nonzero(right))
                    S1 = self.entropy_gini(N1, c_n_l)
                    S2 = self.entropy_gini(N2, c_n_r)
                    ig = S0 - (N1 / N) * S1 - (N2 / N) * S2
                    d.loc[len(d.index)] = [names[i], feature_unique_mean[k], ig]
        else:
            for name in names:
                min_feature, max_feature = min(X[name]), max(X[name])
                splits = self.split_bins[name]
                needed_splits = splits[(splits >= min_feature) & (splits < max_feature)]
                for k in needed_splits:
                    left_df = X.loc[X[name] <= k]
                    right_df = X.loc[X[name] > k]
                    left = self.y[left_df.index]
                    right = self.y[right_df.index]
                    N1 = len(left)
                    N2 = len(right)
                    c_n_l = int(np.count_nonzero(left))
                    c_n_r = int(np.count_nonzero(right))
                    S1 = self.entropy_gini(N1, c_n_l)
                    S2 = self.entropy_gini(N2, c_n_r)
                    ig = S0 - (N1 / N) * S1 - (N2 / N) * S2
                    d.loc[len(d.index)] = [name, k, ig]
        d = d.sort_values(by=['ig', 'col_name'], ascending=[False, True])
        if len(d):
            col_name, split_value, ig = d.iloc[0, 0], d.iloc[0, 1], d.iloc[0, 2]
            return col_name, split_value
        return None, None
    
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
        # self.tree.print_tree()
        # print('')
        if self.leafs_cnt >= self.max_leafs or self.depth > self.max_depth:
            return
        if not self.tree.root:
            y = self.y[X.index]
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
            y = self.y[X.index]
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
                # print('Зашел в левый лист')
                self.curr.left = Node(('leaf_left', np.count_nonzero(y == 1) / len(y)))
                self.leafs_sum += self.curr.left.value[1]
                self.curr.left.is_leaf = True
                self.curr.left.parent = self.curr
                self.leafs_cnt += 1
                self.potential_leafs -= 1
                self.build_tree(self.curr.df)
        elif self.curr.left and not self.curr.right:
            X_curr = self.curr.df.loc[self.curr.df[self.curr.value[0]] > self.curr.value[1]]
            y_r = self.y[X_curr.index]
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
                self.curr.right = Node(('leaf_right', np.count_nonzero(y_r == 1) / len(y_r)))
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

    def fit(self, X, y):
        self.y = y
        self.N = len(X)
        self.get_split_values_or_bins(X)
        self.fi.update({col: 0 for col in X.columns.values})
        self.build_tree(X)

    def is_leaf(self, y):
        '''True - лист, False - узел'''
        if len(y) == 1 or np.count_nonzero(y == 1) == 0 or np.count_nonzero(y == 0) == 0 or \
                self.max_depth == self.depth or self.max_leafs <= 2 or self.min_samples_split > len(y) or \
                self.leafs_cnt + self.potential_leafs >= self.max_leafs:
            return True
        return False

    def predict_proba(self, X):
        proba = []
        for ind in X.index.values:
            s = X.loc[ind]
            proba.append(self.tree.find_proba(s))
        return proba
    
    def predict(self, X):
        proba = []
        for ind in X.index.values:
            s = X.loc[ind]
            proba.append(self.tree.find_proba(s))
        pred = [1 if val > 0.5 else 0 for val in proba]
        return pred
    
    def feature_importance(self, N_p, N, N_l, N_r, I, I_l, I_r):
        return N_p * (I - N_l * I_l / N_p - N_r * I_r / N_p) / N
    
    def compute_feature_importance(self, X, X_left, X_right):
        left = self.y[X_left.index]
        right = self.y[X_right.index]
        y = self.y[X.index]
        N = len(X)
        N_l = len(left)
        N_r = len(right)
        c_n_l = int(np.count_nonzero(left))
        c_n_r = int(np.count_nonzero(right))
        c_n = int(np.count_nonzero(y))
        S = self.entropy_gini(N, c_n)
        S1 = self.entropy_gini(N_l, c_n_l)
        S2 = self.entropy_gini(N_r, c_n_r)
        return self.feature_importance(N, self.N, N_l, N_r, S, S1, S2)