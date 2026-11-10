"""Evaluation metrics. Labels are used only here, never by the algorithms."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


def ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Adjusted Rand index: 1 = identical partitions, ~0 = chance."""
    return float(adjusted_rand_score(y_true, y_pred))


def nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized mutual information."""
    return float(normalized_mutual_info_score(y_true, y_pred))


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clustering accuracy: fraction of points in the right group after the
    best one-to-one matching of clusters to classes (Hungarian algorithm).
    Clusters left without a class count as errors."""
    C = contingency_matrix(y_true, y_pred)
    rows, cols = linear_sum_assignment(C, maximize=True)
    return float(C[rows, cols].sum() / C.sum())


def reconstruction_error(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Mean squared Euclidean distance between rows of X and X_hat."""
    return float(np.mean(np.sum((X - X_hat) ** 2, axis=1)))
