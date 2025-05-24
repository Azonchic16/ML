import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

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
    
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.depth = 0
        self.leafs_sum = 0
        self.potential_leafs = 1
        self.tree = Tree()
        self.curr = None
        # self.bins = bins
        self.split_bins = None
        # self.criterion = criterion
        self.fi = {}

    def __str__(self):
        attrs = vars(self)
        return f'MyTreeClf class: ' + ', '.join(f'{key}={value}' for key, value in attrs.items())
    
    def loss(self, N, y):
        return  np.sum((y - y.mean()) ** 2) / N
    
    def get_best_split(self, X, y):
        # print(self.tree.print_tree())
        N = len(y)
        S0 = self.loss(N, y)
        X_numpy = X.to_numpy()
        split_value = None
        ig = None
        # if not self.split_bins:
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
        # else:
        #     for name in names:
        #         min_feature, max_feature = min(X[name]), max(X[name])
        #         splits = self.split_bins[name]
        #         needed_splits = splits[(splits >= min_feature) & (splits < max_feature)]
        #         for k in needed_splits:
        #             left_df = X.loc[X[name] <= k]
        #             right_df = X.loc[X[name] > k]
        #             left = self.y[left_df.index]
        #             right = self.y[right_df.index]
        #             N1 = len(left)
        #             N2 = len(right)
        #             c_n_l = int(np.count_nonzero(left))
        #             c_n_r = int(np.count_nonzero(right))
        #             S1 = self.entropy_gini(N1, c_n_l)
        #             S2 = self.entropy_gini(N2, c_n_r)
        #             ig = S0 - (N1 / N) * S1 - (N2 / N) * S2
        #             d.loc[len(d.index)] = [name, k, ig]
        # d = d.sort_values(by=['ig', 'col_name'], ascending=[False, True])
        return col_name, split_value
        # return None, None
    
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
            # self.fi[col_name] += self.compute_feature_importance(X, X_left, X_right)
            self.build_tree(X_left)
        if not self.curr.left:
            y = self.y[X.index]
            col_name, split_value = self.get_best_split(X, y)
            # print(col_name, split_value)
            if not self.is_leaf(y) and col_name: # узел
                self.curr.left = Node((col_name, split_value))
                self.curr.left.parent = self.curr
                self.curr.left.df = X
                self.curr = self.curr.left
                X_left = X.loc[X[col_name] <= split_value]
                X_right = X.loc[X[col_name] > split_value]
                # self.fi[col_name] += self.compute_feature_importance(X, X_left, X_right)
                self.depth += 1
                self.potential_leafs += 1
                self.build_tree(X_left)
            else: #  лист
                # print('Зашел в левый лист')
                self.curr.left = Node(('leaf_left', y.mean()))
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
                # self.fi[col_name] += self.compute_feature_importance(X_curr, X_left, X_right)
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

    def fit(self, X, y):
        self.y = y
        self.N = len(X)
        self.names = X.columns.values.tolist()
        # self.get_split_values_or_bins(X)
        # self.fi.update({col: 0 for col in X.columns.values})
        self.build_tree(X)

    def is_leaf(self, y):
        '''True - лист, False - узел'''
        if len(y) == 1 or self.max_depth == self.depth or self.max_leafs <= 2 or \
            self.min_samples_split > len(y) or self.leafs_cnt + self.potential_leafs >= self.max_leafs:
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
    

# if __name__ == '__main__':
#     X = load_diabetes()
#     df = pd.DataFrame(X['data'], columns=X['feature_names'])
#     y = X['target']
#     lst = [(1,1,2), (3,2,5), (5,100,10), (4,50,17), (10,40,21), (15,35,30)]
#     lst = [lst[3]]
#     for max_depth, min_samples_split, max_leafs in lst:
#         model = MyTreeReg(max_depth=max_depth, min_samples_split=min_samples_split, max_leafs=max_leafs)
#         model.fit(df, y)
#         print(round(model.leafs_sum, 6))
#         print(model.leafs_cnt)