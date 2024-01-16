from src.models.NBC import NBC
from sklearn.neighbors import NearestNeighbors
import math

class sklearn_NBC(NBC):
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
        self.nbrs_ = NearestNeighbors(n_neighbors=self.k, algorithm="auto").fit(self.X_)
        _, self.knn_ = self.nbrs_.kneighbors(self.X_)