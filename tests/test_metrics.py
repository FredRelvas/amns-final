import numpy as np
import pytest

from pcacluster.metrics import acc, ari, reconstruction_error

Y = np.array([0, 0, 1, 1, 2, 2])


def test_ari_is_invariant_to_label_names():
    assert ari(Y, np.array([2, 2, 0, 0, 1, 1])) == pytest.approx(1.0)


def test_ari_single_cluster_is_zero():
    assert ari(Y, np.zeros_like(Y)) == pytest.approx(0.0)


def test_acc_is_invariant_to_label_names():
    assert acc(Y, np.array([2, 2, 0, 0, 1, 1])) == pytest.approx(1.0)


def test_acc_uses_best_matching():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 0, 0])  # best: class 0 -> 1, class 1 -> 0
    assert acc(y_true, y_pred) == pytest.approx(5 / 6)


def test_reconstruction_error():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert reconstruction_error(X, np.zeros_like(X)) == pytest.approx(1.0)
