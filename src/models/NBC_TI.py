from src.models.NBC import NBC
from scipy.spatial.distance import minkowski
import numpy as np
import math

class Point:
    def __init__(self, orig_id, coords):
        self.coords = coords
        self.dist = None
        self.eps = None
        self.id = None
        self.orig_id = orig_id

class NBC_TI(NBC):
    def __init__(self, k=None, n_clusters=3):
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

        self.ti_knn()
        self.rknn()
        self.ndf()
        self.labels()

        return self.labels_

    def ti_knn(self):
        points = [Point(i, p) for i, p in enumerate(self.X_)]

        min_coord = np.min(self.X_)
        ref_point = Point(-1, np.full(self.dim_, min_coord))
        for p in points:
            p.dist = self.distance(p, ref_point)

        points.sort(key=lambda p: p.dist)
        for i, p in enumerate(points):
            p.id = i


        self.knn_ = np.empty((self.n_, self.k), dtype=int)
        for p in points:
            self.knn_[p.orig_id] = self.ti_k_neighborhood(points, p)


    def ti_k_neighborhood(self, points, p):
        '''
        Returns indicies of k nearest neighbours of point p in points
        '''
        k_neighborhood = []
        i = 0
        prev_p = p
        next_p = p
        backward, prev_p = self.preceding(points, prev_p)
        forward, next_p = self.following(points, next_p)
        p, prev_p, next_p, backward, forward, k_neighborhood, i = self.find_first_k_candidates_fab(points, p, prev_p, next_p, backward, forward, k_neighborhood, i)
        p, prev_p, backward, k_neighborhood, i = self.find_first_k_candidates_b(points, p, prev_p, backward, k_neighborhood, i)
        p, next_p, forward, k_neighborhood, i = self.find_first_k_candidates_f(points, p, next_p, forward, k_neighborhood, i)
        p.eps = self.eps(k_neighborhood)
        p, prev_p, backward, k_neighborhood = self.verify_k_neighbours_b(points, p, prev_p, backward, k_neighborhood)
        p, next_p, forward, k_neighborhood = self.verify_k_neighbours_f(points, p, next_p, forward, k_neighborhood)

        return [e[0].orig_id for e in k_neighborhood]


    def preceding(self, points, p):
        if p.id > 0:
            p = points[p.id - 1]
            backward = True
        else:
            backward = False
        return backward, p


    def following(self, points, p):
        if p.id < self.n_ - 1:
            p = points[p.id + 1]
            forward = True
        else:
            forward = False
        return forward, p

    def distance(self, first: Point, second: Point):
        return minkowski(first.coords, second.coords, p=self.dim_)

    def eps(self, k_neighborhood):
        return max([n[1] for n in k_neighborhood])

    def insert_in_sorted(self, k_neighborhood, point, dist):
        for i in reversed(range(len(k_neighborhood))):
            if dist > k_neighborhood[i][1]:
                k_neighborhood.insert(i + 1, (point, dist))
                return k_neighborhood
        k_neighborhood.insert(0, (point, dist))
        return k_neighborhood


    def find_first_k_candidates_fab(self, points, p, prev_p, next_p, backward, forward, k_neighborhood, i):
        while backward and forward and i < self.k:
            if p.dist - prev_p.dist < next_p.dist - p.dist:
                dist = self.distance(prev_p, p)
                self.insert_in_sorted(k_neighborhood, prev_p, dist)
                backward, prev_p = self.preceding(points, prev_p)
            else:
                dist = self.distance(next_p, p)
                self.insert_in_sorted(k_neighborhood, next_p, dist)
                forward, next_p = self.following(points, next_p)
            i += 1

        return p, prev_p, next_p, backward, forward, k_neighborhood, i


    def find_first_k_candidates_b(self, points, p, prev_p, backward, k_neighborhood, i):
        while backward and i < self.k:
            dist = self.distance(prev_p, p)
            self.insert_in_sorted(k_neighborhood, prev_p, dist)
            backward, prev_p = self.preceding(points, prev_p)
            i += 1
        return p, prev_p, backward, k_neighborhood, i


    def find_first_k_candidates_f(self, points, p, next_p, forward, k_neighborhood, i):
        while forward and i < self.k:
            dist = self.distance(next_p, p)
            self.insert_in_sorted(k_neighborhood, next_p, dist)
            forward, next_p = self.following(points, next_p)
            i += 1
        return p, next_p, forward, k_neighborhood, i


    def verify_k_neighbours_b(self, points, p, prev_p, backward, k_neighborhood):
        while backward and (p.dist - prev_p.dist) <= p.eps:
            dist = self.distance(prev_p, p)
            if dist < p.eps:
                edge_len = len([r for r in k_neighborhood if r[1] == p.eps])
                if len(k_neighborhood) - edge_len >= self.k - 1:
                    k_neighborhood = [r for r in k_neighborhood if r[1] != p.eps]
                    self.insert_in_sorted(k_neighborhood, prev_p, dist)
                    p.eps = self.eps(k_neighborhood)
                else:
                    self.insert_in_sorted(k_neighborhood, prev_p, dist)
            elif dist == p.eps:
                self.insert_in_sorted(k_neighborhood, prev_p, dist)
            backward, prev_p = self.preceding(points, prev_p)
        return p, prev_p, backward, k_neighborhood


    def verify_k_neighbours_f(self, points, p, next_p, forward, k_neighborhood):
        while forward and (next_p.dist - p.dist) <= p.eps:
            dist = self.distance(next_p, p)
            if dist < p.eps:
                i = len([r for r in k_neighborhood if r[1] == p.eps])
                if len(k_neighborhood) - i >= self.k - 1:
                    k_neighborhood = [r for r in k_neighborhood if r[1] != p.eps]
                    self.insert_in_sorted(k_neighborhood, next_p, dist)
                    p.eps = self.eps(k_neighborhood)
                else:
                    self.insert_in_sorted(k_neighborhood, next_p, dist)
            elif dist == p.eps:
                self.insert_in_sorted(k_neighborhood, next_p, dist)
            forward, next_p = self.following(points, next_p)
        return p, next_p, forward, k_neighborhood