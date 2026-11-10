"""Evaluation metrics. Labels are used only here, never by the algorithms."""

import numpy as np


def ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Adjusted Rand index: 1 = identical partitions, ~0 = chance."""
    raise NotImplementedError


def nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mutual information."""
    raise NotImplementedError


def knn_accuracy(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 5,
) -> float:
    """Test accuracy of a k-NN classifier fitted on the training split."""
    raise NotImplementedError


def reconstruction_error(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Mean squared Euclidean distance between rows of X and X_hat."""
    raise NotImplementedError
