"""Label-free rules to choose the number of PCA components d.

Every rule receives X_train already preprocessed (see data.Preprocessor):
standardized or only centered, depending on the dataset. Rules work on the
eigenvalues of its covariance matrix; for standardized data this is the
correlation matrix and the rules reduce to their usual definitions.
"""

from functools import lru_cache

import numpy as np


def _eigenvalues(X: np.ndarray) -> np.ndarray:
    """Eigenvalues of the covariance matrix X^T X / n, in descending order.

    Only the min(n - 1, p) possibly nonzero ones are returned (X is centered
    here, so the rank is at most n - 1). The eigenvalues come from the smaller
    of the two Gram matrices, so p > n (e.g. gene expression) never builds a
    p x p matrix.
    """
    n, p = X.shape
    X = X - X.mean(axis=0)
    gram = X.T @ X if p <= n else X @ X.T
    ev = np.linalg.eigvalsh(gram)[::-1] / n
    return np.clip(ev[: min(n - 1, p)], 0.0, None)


@lru_cache(maxsize=None)
def _median_noise_eigenvalue(n: int, p: int, n_sim: int, seed: int) -> float:
    """Expected median eigenvalue of the covariance of n x p N(0, 1) noise,
    averaged over n_sim simulations."""
    rng = np.random.default_rng(seed)
    return float(
        np.mean(
            [np.median(_eigenvalues(rng.standard_normal((n, p)))) for _ in range(n_sim)]
        )
    )


def d_marchenko_pastur(
    X_train: np.ndarray,
    sigma2: float | str = "median",
    n_sim: int = 30,
    seed: int = 0,
) -> int:
    """Count eigenvalues above the edge sigma2 * (1 + sqrt(p/n))**2.

    sigma2='median' estimates the noise level as median(eigenvalues) divided by
    the expected median eigenvalue of N(0, 1) noise of the same shape (~30
    simulations); it works with or without standardization. A fixed sigma2=1
    only makes sense for standardized data or synthetic data with noise_std=1.
    """
    n, p = X_train.shape
    ev = _eigenvalues(X_train)
    if sigma2 == "median":
        sigma2 = np.median(ev) / _median_noise_eigenvalue(n, p, n_sim, seed)
    elif isinstance(sigma2, str):
        raise ValueError(f"sigma2 must be a number or 'median', got {sigma2!r}")
    edge = sigma2 * (1 + np.sqrt(p / n)) ** 2
    return int(np.sum(ev > edge))


def d_parallel_analysis(
    X_train: np.ndarray, n_perm: int = 100, q: float = 95, seed: int = 0
) -> int:
    """Count eigenvalues above the q-th percentile, per index, of the
    eigenvalues obtained after shuffling each column independently.

    Counting stops at the first eigenvalue that does not exceed its threshold
    (Horn's stopping rule). Counting every exceedance instead would return
    about (100 - q)% of the indices even on pure noise (2.5 of 50 at q=95).
    """
    rng = np.random.default_rng(seed)
    X = X_train - X_train.mean(axis=0)
    null = np.array(
        [_eigenvalues(rng.permuted(X, axis=0)) for _ in range(n_perm)]
    )
    threshold = np.percentile(null, q, axis=0)
    below = _eigenvalues(X) <= threshold
    return int(np.argmax(below)) if below.any() else len(below)


def d_from_K(K: int, minus_one: bool = False) -> int:
    """d = K, or K - 1 (Ding and He)."""
    return K - 1 if minus_one else K


def d_variance_threshold(X_train: np.ndarray, frac: float = 0.90) -> int:
    """Smallest d whose cumulative explained variance is >= frac."""
    ev = _eigenvalues(X_train)
    explained = np.cumsum(ev) / ev.sum()
    # Tolerance so that a cumulative sum of exactly frac is not missed by
    # floating point error.
    return int(np.searchsorted(explained, frac - 1e-12) + 1)


def d_kaiser(X_train: np.ndarray) -> int:
    """Count eigenvalues above their mean (the usual > 1 for standardized data)."""
    p = X_train.shape[1]
    ev = _eigenvalues(X_train)
    # The mean is over all p eigenvalues (trace / p), not only the ones
    # returned, which are fewer than p when p > n - 1.
    return int(np.sum(ev > ev.sum() / p))


def d_elbow(X_train: np.ndarray) -> int:
    """d at the elbow of the eigenvalue (scree) curve.

    The elbow is the point of the scree curve farthest from the straight line
    joining its first and last points, after rescaling both axes to [0, 1].
    Following Cattell's scree test, the components before the elbow are kept
    and the elbow point itself is not (at least one component is returned).
    """
    ev = _eigenvalues(X_train)
    if len(ev) < 3 or ev[0] == ev[-1]:
        return min(len(ev), 1)
    x = np.linspace(0.0, 1.0, len(ev))
    y = (ev - ev[-1]) / (ev[0] - ev[-1])
    # Distance below the chord from (0, 1) to (1, 0), proportional to 1 - x - y.
    return max(int(np.argmax(1.0 - x - y)), 1)
