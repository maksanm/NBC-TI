from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math

class NBC(BaseEstimator, ClusterMixin):
    def __init__(self, k=None, n_clusters=None):
        self.n_clusters = n_clusters
        self.k = k

    def fit_predict(self, X):
        # Set k=sqrt(n) if k wasn't provided
        if self.k is None: self.k = int(math.sqrt(X.shape[0]))

        # Brute k-NN calculates distances to all points
        self.nbrs_ = NearestNeighbors(n_neighbors=self.k, algorithm="brute").fit(X)
        _, self.knn_ = self.nbrs_.kneighbors(X)
        self.n_ = self.knn_.shape[0]

        self.calculate_rknn()
        self.calculate_ndf()
        self.assign_labels()

        return self.labels_

    def calculate_rknn(self):
        self.rknn_ = []
        mask = np.zeros((self.n_, self.n_), dtype=bool)
        for i, a in enumerate(self.knn_):
            mask[a, i] = True
        mask = np.logical_and(mask, np.not_equal.outer(np.arange(self.n_), np.arange(self.n_)))
        self.rknn_ += [np.where(mask[i])[0].tolist() for i in range(self.n_)]

    def calculate_ndf(self):
        self.ndf_ = [len(self.rknn_[i])/len(self.knn_[i]) for i in range(self.n_)]

    def assign_labels(self):
        labels = [-1 for _ in range(self.n_)]
        cluster_id = 1
        for i, nbrs in enumerate(self.knn_):
            if labels[i] != -1:
                continue
            if self.ndf_[i] < 1:
                labels[i] = 0
            else:
                labels[i] = cluster_id
                seeds = nbrs.tolist()
                while len(seeds) > 0:
                    p = seeds.pop()
                    if labels[p] == -1:
                        labels[p] = cluster_id
                        if self.ndf_[p] >= 1:
                            seeds += self.knn_[p].tolist()
                    elif labels[p] == 0:
                        labels[p] = cluster_id
                cluster_id += 1
        self.labels_ = np.array(labels)