"""Dataset loading, subsampling, train/test split and preprocessing."""

from pathlib import Path
from typing import Self

import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OPENML_NAMES = {
    "mnist": "mnist_784",
    "fashion_mnist": "Fashion-MNIST",
    "tumors": "11_Tumors",
}
# Standardize only datasets whose features have different units or scales
# (genes). Pixels and synthetic coordinates share one unit, so they are only
# centered. Criterion and evidence in docs/DECISOES.md.
STANDARDIZE = {"tumors"}


def load_dataset(
    name: str, *, n: int | None = None, seed: int = 0, **kw
) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset as (X, y).

    name: 'digits', 'mnist', 'fashion_mnist', 'tumors' or 'synthetic'.
    'tumors' is the 11_Tumors gene expression dataset (174 samples x 12533
    genes, 11 cancer types), a natural p > n case. MNIST, Fashion-MNIST and
    11_Tumors come from OpenML and are cached in data/.
    'synthetic' generates n samples (default 1000) and accepts n_clusters, p,
    noise_std and separation in kw. For the other datasets, if n is given,
    return a random subsample of n rows.
    """
    if name == "synthetic":
        return _synthetic(n or 1000, seed, **kw)
    if name == "digits":
        X, y = load_digits(return_X_y=True)
    elif name in OPENML_NAMES:
        X, y = fetch_openml(
            OPENML_NAMES[name],
            version=1,
            return_X_y=True,
            as_frame=True,  # 11_Tumors stores its labels as strings
            data_home=DATA_DIR,
            parser="liac-arff",  # sklearn's pandas parser breaks with pandas 3
        )
    else:
        raise ValueError(f"Unknown dataset: {name!r}")
    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
    return (X, y) if n is None else subsample(X, y, n, seed)


def _synthetic(
    n: int,
    seed: int,
    n_clusters: int = 10,
    p: int = 50,
    noise_std: float = 1.0,
    separation: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian clusters plus isotropic noise N(0, noise_std^2) in all p
    dimensions (p >= n_clusters).

    Centers are separation times orthonormal vectors of a random subspace:
    every pair of centers is separation * sqrt(2) apart and the signal spans
    exactly n_clusters - 1 dimensions. Extra dimensions only add noise.
    """
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((p, n_clusters)))
    centers = separation * basis.T
    y = rng.integers(n_clusters, size=n)
    X = centers[y] + noise_std * rng.standard_normal((n, p))
    return X, y


def subsample(
    X: np.ndarray, y: np.ndarray, n: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Random subsample of n rows, used to vary p/n."""
    idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split. Returns X_train, X_test, y_train, y_test."""
    return tuple(
        train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    )


class Preprocessor:
    """Drops constant columns, centers and, if scale, divides by the standard
    deviation. Fitted on training data only."""

    def __init__(self, scale: bool = True):
        self.scale = scale

    def fit(self, X_train: np.ndarray) -> Self:
        std = X_train.std(axis=0)
        self.keep_ = std > 0
        self.mean_ = X_train.mean(axis=0)[self.keep_]
        self.std_ = std[self.keep_] if self.scale else 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X[:, self.keep_] - self.mean_) / self.std_
