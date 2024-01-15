from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import minkowski
import numpy as np
import math

class NBC(BaseEstimator, ClusterMixin):
    def __init__(self, k=None, n_clusters=None):
        self.n_clusters = n_clusters
        self.k = k

    def fit_predict(self, X):
        self.n_ = X.shape[0]
        self.dim_ = X.shape[1]
        self.X_ = X

        # Set k=sqrt(n) if wasn't provided
        if self.k is None: self.k = int(math.sqrt(self.n_))

        if self.k > self.n_:
            raise Exception("Error: The number of neighbors ‘k’ in should be less than the number of elements in the dataset")

        self.knn()
        self.rknn()
        self.ndf()
        self.labels()

        return self.labels_

    def knn(self):
        distances = np.empty((self.n_, self.n_))
        for i in range(self.n_):
            for j in range(i + 1, self.n_):
                dist = self.distance(self.X_[i], self.X_[j])
                distances[i, j] = dist
                distances[j, i] = dist
        self.knn_ = np.empty((self.n_, self.k), dtype=int)
        for p in range(self.n_):
            self.knn_[p] = distances[p].argsort()[:self.k]

    def rknn(self):
        self.rknn_ = []
        mask = np.zeros((self.n_, self.n_), dtype=bool)
        for i, a in enumerate(self.knn_):
            mask[a, i] = True
        mask = np.logical_and(mask, np.not_equal.outer(np.arange(self.n_), np.arange(self.n_)))
        self.rknn_ += [np.where(mask[i])[0].tolist() for i in range(self.n_)]

    def ndf(self):
        self.ndf_ = [len(self.rknn_[i])/len(self.knn_[i]) for i in range(self.n_)]

    def labels(self):
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

    def distance(self, x, y):
        return minkowski(x, y, p=self.dim_)