from src.models.NBC import NBC
from sklearn.utils.validation import check_is_fitted
import numpy as np

class NBC_TI(NBC):
    def __init__(self, k=None, n_clusters=3):
        self.n_clusters = n_clusters
        self.k = k