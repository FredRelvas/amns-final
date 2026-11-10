"""Label-free rules to choose the number of PCA components d.

Every rule receives X_train already preprocessed (see data.Preprocessor).
"""

import numpy as np


def d_marchenko_pastur(X_train: np.ndarray, sigma2: float | str = "median") -> int:
    """Count correlation eigenvalues above the edge sigma2 * (1 + sqrt(p/n))**2.

    sigma2='median' estimates the noise level as median(eigenvalues) divided by
    the expected median eigenvalue of N(0, 1) noise of the same shape (~30
    simulations).
    """
    raise NotImplementedError


def d_parallel_analysis(
    X_train: np.ndarray, n_perm: int = 100, q: float = 95, seed: int = 0
) -> int:
    """Count eigenvalues above the q-th percentile, per index, of the
    eigenvalues obtained after shuffling each column independently."""
    raise NotImplementedError


def d_from_K(K: int, minus_one: bool = False) -> int:
    """d = K, or K - 1 (Ding and He)."""
    return K - 1 if minus_one else K


def d_variance_threshold(X_train: np.ndarray, frac: float = 0.90) -> int:
    """Smallest d whose cumulative explained variance is >= frac."""
    raise NotImplementedError


def d_kaiser(X_train: np.ndarray) -> int:
    """Count eigenvalues > 1."""
    raise NotImplementedError


def d_elbow(X_train: np.ndarray) -> int:
    """d at the elbow of the eigenvalue (scree) curve."""
    raise NotImplementedError
