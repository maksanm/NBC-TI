from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np

class NBC_TI(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def fit_predict(self, X):
        self.cluster_centers_ = np.random.rand(self.n_clusters, X.shape[0]) * self.n_clusters
        return self.cluster_centers_.astype(int)[0]