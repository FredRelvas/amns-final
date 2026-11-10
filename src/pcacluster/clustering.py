"""Clustering algorithms with a common interface and estimation of K."""

from typing import Protocol, Self

import numpy as np


class ClusterAlgo(Protocol):
    name: str

    def fit(self, Z: np.ndarray, K: int) -> Self: ...

    def predict(self, Z: np.ndarray) -> np.ndarray: ...


class KMeansAlgo:
    """K-means with several initializations (n_init >= 5)."""

    name = "kmeans"

    def __init__(self, n_init: int = 5, seed: int = 0):
        self.n_init = n_init
        self.seed = seed

    def fit(self, Z: np.ndarray, K: int) -> Self:
        raise NotImplementedError

    def predict(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GMMAlgo:
    """Gaussian mixture with diagonal covariance."""

    name = "gmm"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, Z: np.ndarray, K: int) -> Self:
        raise NotImplementedError

    def predict(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def estimate_K(
    Z: np.ndarray, algo_name: str, K_grid: range, criterion: str = "bic"
) -> int:
    """Fit algo_name for each K in K_grid and pick K by criterion.

    criterion: 'bic' or 'silhouette'.
    """
    raise NotImplementedError
