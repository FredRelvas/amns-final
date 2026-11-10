"""Dataset loading, subsampling, train/test split and preprocessing."""

from typing import Self

import numpy as np


def load_dataset(
    name: str, *, n: int | None = None, seed: int = 0, **kw
) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset as (X, y).

    name: 'digits', 'mnist', 'fashion_mnist' or 'synthetic'. MNIST and
    Fashion-MNIST come from sklearn's fetch_openml, cached in data/.
    'synthetic' accepts n_clusters, p, noise_std and separation in kw.
    If n is given, return a subsample of n rows.
    """
    raise NotImplementedError


def subsample(
    X: np.ndarray, y: np.ndarray, n: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Random subsample of n rows, used to vary p/n."""
    raise NotImplementedError


def split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split. Returns X_train, X_test, y_train, y_test."""
    raise NotImplementedError


class Preprocessor:
    """Drops constant columns and standardizes, fitted on training data only."""

    def fit(self, X_train: np.ndarray) -> Self:
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
