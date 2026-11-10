import numpy as np
import pytest

from pcacluster.data import Preprocessor, load_dataset
from pcacluster.rules import (
    d_elbow,
    d_from_K,
    d_kaiser,
    d_marchenko_pastur,
    d_parallel_analysis,
    d_variance_threshold,
)


def with_eigenvalues(ev, n=200, seed=0):
    """Centered n x len(ev) matrix whose covariance has exactly these
    eigenvalues (orthogonal columns scaled to the requested variances)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, len(ev)))
    Q, _ = np.linalg.qr(Z - Z.mean(axis=0))
    return np.sqrt(n * np.asarray(ev)) * Q


def synthetic_train(n=1000, p=50, K=10, seed=0, **kw):
    X, y = load_dataset("synthetic", n=n, seed=seed, n_clusters=K, p=p, **kw)
    return Preprocessor(scale=False).fit(X).transform(X)


def test_d_from_K():
    assert d_from_K(10) == 10
    assert d_from_K(10, minus_one=True) == 9


def test_variance_threshold_and_kaiser_on_known_eigenvalues():
    # Total variance 20: cumulative 0.5, 0.75, 0.9, 0.95, 1.0 and mean 4.
    X = with_eigenvalues([10, 5, 3, 1, 1])
    assert d_variance_threshold(X, 0.90) == 3
    assert d_variance_threshold(X, 0.50) == 1  # exactly on the boundary
    assert d_variance_threshold(X, 1.0) == 5
    assert d_kaiser(X) == 2  # 10 and 5 are above the mean 4


def test_kaiser_mean_uses_all_p_dimensions_when_p_exceeds_n():
    # n=20 < p=100: only 19 eigenvalues exist, the other 81 are zeros.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 100))
    ev_sum = np.sum(np.var(X, axis=0))
    assert d_kaiser(X) == np.sum(
        np.linalg.svd(X - X.mean(axis=0), compute_uv=False) ** 2 / 20 > ev_sum / 100
    )


def test_elbow_finds_the_knee():
    # Three large eigenvalues, then a flat tail.
    X = with_eigenvalues([10, 8, 6] + [0.3] * 17, n=300)
    assert d_elbow(X) == 3


@pytest.mark.parametrize("seed", range(5))
def test_pure_noise_has_no_spikes(seed):
    X = np.random.default_rng(seed).standard_normal((500, 50))
    assert d_marchenko_pastur(X, sigma2=1) <= 1
    assert d_marchenko_pastur(X) <= 1
    assert d_parallel_analysis(X, n_perm=50) <= 1


def test_noise_level_estimate_absorbs_the_scale():
    X = 7.0 * np.random.default_rng(1).standard_normal((500, 50))
    assert d_marchenko_pastur(X) <= 1
    assert d_marchenko_pastur(X, sigma2=1) == 50  # wrong sigma2: everything is "signal"


def test_marchenko_pastur_counts_spikes_in_noise():
    # 5 strong spikes on top of unit noise, n=500, p=50.
    rng = np.random.default_rng(0)
    n, p, k = 500, 50, 5
    signal = rng.standard_normal((n, k)) * 3.0 @ np.linalg.qr(
        rng.standard_normal((p, k))
    )[0].T
    X = signal + rng.standard_normal((n, p))
    assert d_marchenko_pastur(X, sigma2=1) == k
    assert d_marchenko_pastur(X) == k
    assert d_parallel_analysis(X, n_perm=50) == k


def test_synthetic_signal_has_K_minus_1_dimensions():
    X = synthetic_train()
    assert d_marchenko_pastur(X, sigma2=1) == 9
    assert d_marchenko_pastur(X) == 9
    assert d_parallel_analysis(X, n_perm=50) == 9


def test_rules_work_when_p_exceeds_n():
    X = np.random.default_rng(0).standard_normal((60, 300))
    for d in (
        d_marchenko_pastur(X),
        d_parallel_analysis(X, n_perm=10),
        d_variance_threshold(X),
        d_kaiser(X),
        d_elbow(X),
    ):
        assert 0 <= d <= 59


def test_parallel_analysis_is_reproducible_by_seed():
    X = synthetic_train(n=300, p=30, K=5)
    assert d_parallel_analysis(X, n_perm=20, seed=3) == d_parallel_analysis(
        X, n_perm=20, seed=3
    )


def test_unknown_sigma2_string_is_rejected():
    with pytest.raises(ValueError):
        d_marchenko_pastur(np.zeros((10, 3)), sigma2="mean")
