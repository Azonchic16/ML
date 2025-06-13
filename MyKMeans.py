import numpy as np
import pandas as pd
import random


class MyKMeans():

    def __init__(self, n_clusters=3, max_iter=10, n_init=3, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.inertia_ = None
        self.cluster_centers_ = None

    def __str__(self):
        return f'MyKMeans class: ' + ', '.join(f'{key}={value}' for key, value in self.__dict__.items())

    def fit(self, X):
        np.random.seed(seed=self.random_state)
        points_cluster = []
        for i in range(self.n_init):
            centers = {}
            features = X.columns.values
            # X_learn = X.copy()
            for _ in range(self.n_clusters):
                center = X.apply(lambda col: np.random.uniform(col.min(), col.max()))
                centers[_] = center
            # print(centers)
            for epoch in range(self.max_iter):
                # print(centers)
                X_learn = X.copy()
                for cluster_num, center in centers.items():
                    X_learn[cluster_num] = X_learn[features].apply(lambda x: np.sqrt(np.sum((x - center) ** 2)), axis=1)
                # print(X_learn)
                cluster_num = [i for i in range(self.n_clusters)]
                X_learn['min_dist'] = X_learn[cluster_num].min(axis=1)
                X_learn['cluster'] = X_learn[cluster_num].idxmin(axis=1)
                X_learn['obj_ind'] = X_learn.index
                X_learn = X_learn[['cluster', 'obj_ind']]
                X_learn = X_learn.groupby('cluster', as_index=False).agg(list)
                # print(X_learn)
                # X_learn = (X_learn.groupby('cluster').apply(lambda x: x.index.tolist()).reset_index(name='obj_ind'))
                # print(X_learn)
                center_points = []
                # print(X_learn)
                for ind, row in X_learn.iterrows():
                    center_points.append(centers[row['cluster']])
                # print(center_points)
                X_learn['center'] = center_points
                points_cluster.append(X_learn)
                new_centers = {}
                stop = 0
                for ind, row in X_learn.iterrows():
                    cluster_obj = X.loc[row['obj_ind']]
                    new_center = cluster_obj.sum() / cluster_obj.shape[0]
                    if (row['center'] == new_center).all():
                        stop += 1
                    new_centers[ind] = new_center
                if stop == self.n_clusters:
                    break
                else:
                    centers.update(new_centers)
        wcss = None
        number_df = None
        for number, df in enumerate(points_cluster):
            res_sum = 0
            for ind, row in df.iterrows():
                center =  row['center']
                points = X.loc[row['obj_ind']]
                points['dist'] = points.apply(lambda x: np.sum((x - center) ** 2), axis=1)
                res_sum += points['dist'].sum()
            wcss_new = res_sum
            print(w)
            if wcss is None or wcss_new < wcss:
                wcss = wcss_new
                number_df = number
        self.inertia_ = wcss
        # print(points_cluster[number_df])
        self.cluster_centers_ = [elem for elem in list(points_cluster[number_df]['center'])]

                