"""Tests for the sweep over d and the comparison with the rules."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from pcacluster.data import load_dataset, split
from pcacluster.metrics import ari, reconstruction_error
from pcacluster.sweep import (
    _classify,
    best_d,
    compare_rules,
    default_score,
    effect_vs_full,
    oracle_cv,
    prepare,
    rule_ds,
    run_sweep,
    summarize,
)

COLUMNS = [
    "dataset",
    "n",
    "p",
    "p_over_n",
    "seed",
    "algo",
    "d",
    "K",
    "ari_train",
    "ari_test",
    "nmi_test",
    "recon_err_test",
    "agreement_full",
]


def synthetic(n=300, p=20, K=4, seed=0, **kw):
    return load_dataset("synthetic", n=n, seed=seed, n_clusters=K, p=p, **kw)


@pytest.fixture(scope="module")
def small_sweep():
    X, y = synthetic()
    return run_sweep(
        X, y, "kmeans", range(1, 21), 4, dataset="synthetic", seeds=[0, 1, 2]
    )


def test_run_sweep_has_the_agreed_columns_and_one_row_per_seed_and_d(small_sweep):
    df = small_sweep
    assert set(COLUMNS) <= set(df.columns)
    pca = df[df["space"] == "pca"]
    assert len(pca) == 3 * 20
    assert not pca.duplicated(["seed", "d"]).any()
    full = df[df["space"] == "full"]
    assert list(full["seed"]) == [0, 1, 2]
    assert (full["agreement_full"] == 1.0).all()
    assert (df["p_over_n"] == 20 / 300).all()


def test_run_sweep_matches_an_independent_pca_per_d():
    """Cross-check: slicing one PCA fitted with the largest d gives the same
    clustering as fitting a new PCA with d components for every d."""
    X, y = synthetic(seed=3)
    df = run_sweep(X, y, "kmeans", [1, 2, 3, 5, 8], 4, dataset="synthetic", seeds=[3])
    X_train, X_test, y_train, y_test = split(X, y, seed=3)
    mean = X_train.mean(axis=0)
    for d in [1, 2, 3, 5, 8]:
        pca = PCA(n_components=d).fit(X_train - mean)
        Z_train, Z_test = pca.transform(X_train - mean), pca.transform(X_test - mean)
        km = KMeans(n_clusters=4, n_init=5, random_state=3).fit(Z_train)
        row = df[(df["space"] == "pca") & (df["d"] == d)].iloc[0]
        assert row["ari_train"] == pytest.approx(ari(y_train, km.predict(Z_train)))
        assert row["ari_test"] == pytest.approx(ari(y_test, km.predict(Z_test)))
        X_hat = pca.inverse_transform(Z_test) + mean
        assert row["recon_err_test"] == pytest.approx(
            reconstruction_error(X_test, X_hat), rel=1e-9
        )


def test_reconstruction_error_falls_with_d_and_vanishes_at_full_rank(small_sweep):
    # p = 20 < n_train, so the last d of the grid is the full rank.
    for _, g in small_sweep[small_sweep["space"] == "pca"].groupby("seed"):
        err = g.sort_values("d")["recon_err_train"].to_numpy()
        assert np.all(np.diff(err) <= 1e-9)
        assert err[-1] == pytest.approx(0.0, abs=1e-8)


def test_train_metrics_do_not_depend_on_the_test_rows():
    """No leakage: changing the test rows leaves every train column equal."""
    X, y = synthetic(seed=1)
    _, X_test, _, _ = split(X, y, seed=1)
    test_rows = np.isin(X, X_test).all(axis=1)
    X_changed = X.copy()
    X_changed[test_rows] += 100.0
    cols = ["ari_train", "nmi_train", "acc_train", "recon_err_train"]
    a = run_sweep(X, y, "gmm", [2, 3, 5], 4, dataset="synthetic", seeds=[1])
    b = run_sweep(X_changed, y, "gmm", [2, 3, 5], 4, dataset="synthetic", seeds=[1])
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert not np.allclose(a["recon_err_test"], b["recon_err_test"])


def test_without_split_test_columns_are_nan_and_agreement_uses_all_data():
    X, y = synthetic(n=150)
    df = run_sweep(
        X, y, "kmeans", [1, 3], 4, dataset="synthetic", seeds=[0], test_size=0
    )
    assert df["ari_test"].isna().all() and df["recon_err_test"].isna().all()
    assert (df["n_train"] == 150).all()
    assert df["agreement_full"].notna().all()
    assert default_score(df) == "ari_train"


def test_d_above_the_pca_limit_is_skipped_with_a_warning():
    X, y = synthetic(n=30, p=50)  # n_train = 21, so d_max = 20
    with pytest.warns(UserWarning, match="skipped"):
        df = run_sweep(
            X, y, "kmeans", [5, 20, 21, 40], 4, dataset="synthetic", seeds=[0]
        )
    assert sorted(df.loc[df["space"] == "pca", "d"]) == [5, 20]


def test_invalid_arguments_are_rejected():
    X, y = synthetic(n=60)
    with pytest.raises(ValueError):
        run_sweep(X, y, "dbscan", [1], 4, dataset="synthetic", seeds=[0])
    with pytest.raises(ValueError):
        run_sweep(X, y, "kmeans", [0, 1], 4, dataset="synthetic", seeds=[0])
    with pytest.raises(ValueError):
        prepare(X, y, "synthetic", 0, test_size=1.0)


def test_estimate_K_flag_records_the_fitted_K():
    X, y = synthetic(n=300, separation=10.0)
    df = run_sweep(
        X,
        y,
        "kmeans",
        [1, 3],
        4,
        dataset="synthetic",
        seeds=[0],
        estimate_K_flag=True,
        K_grid=range(2, 7),
    )
    assert (df["K"] == 4).all()
    k_fit = df.set_index(["space", "d"])["K_fit"]
    assert k_fit[("full", 20)] == 4 and k_fit[("pca", 3)] == 4
    assert k_fit[("pca", 1)] < 4  # one dimension cannot separate 4 clusters


def test_kmeans_on_synthetic_needs_K_minus_1_dimensions():
    """Ding and He: K-1 components hold the K centers, so the curve reaches
    the no-PCA level at d = K-1 and the best d is there or after."""
    X, y = synthetic(n=600, p=30, K=5, separation=6.0)
    df = run_sweep(
        X, y, "kmeans", range(1, 31), 5, dataset="synthetic", seeds=[0, 1, 2]
    )
    curve = summarize(df, "ari_test").set_index("d")["mean"]
    best = best_d(summarize(df, "ari_test"))
    assert best["d_plateau"] <= 4 <= best["d_best"]
    assert curve[1] < curve[4] - 0.3
    eff = effect_vs_full(df, "ari_test").set_index("d")["effect"]
    assert eff[1] == "piora" and eff[4] == "indiferente"


def test_summarize_and_best_d_on_a_known_curve():
    rows = [
        {"space": "pca", "seed": s, "d": d, "ari_test": v + 0.01 * s}
        for s in range(3)
        for d, v in [(1, 0.2), (2, 0.6), (3, 0.97), (4, 1.0), (5, 0.99)]
    ]
    df = pd.DataFrame(rows)
    curve = summarize(df, "ari_test")
    assert list(curve["n_seeds"]) == [3] * 5
    assert curve["std"].iloc[0] == pytest.approx(0.01)
    best = best_d(curve)
    assert best["d_best"] == 4 and best["d_plateau"] == 3


def test_summarize_ignores_d_missing_in_some_seed():
    df = pd.DataFrame(
        [
            {"space": "pca", "seed": 0, "d": 1, "s": 0.1},
            {"space": "pca", "seed": 1, "d": 1, "s": 0.2},
            {"space": "pca", "seed": 0, "d": 2, "s": 0.9},
        ]
    )
    assert list(summarize(df, "s")["d"]) == [1]


def test_oracle_cv_picks_d_on_the_other_seeds():
    # Seed 2 alone prefers d=2; the other seeds prefer d=1.
    values = {0: {1: 0.9, 2: 0.5}, 1: {1: 0.8, 2: 0.6}, 2: {1: 0.1, 2: 0.95}}
    df = pd.DataFrame(
        [
            {"space": "pca", "seed": s, "d": d, "s": v}
            for s, dv in values.items()
            for d, v in dv.items()
        ]
    )
    cv = oracle_cv(df, "s").set_index("seed")
    assert cv.loc[2, "d"] == 1 and cv.loc[2, "score"] == pytest.approx(0.1)
    assert cv.loc[0, "d"] == 2  # mean of seeds 1 and 2: d=1 0.45, d=2 0.775


@pytest.mark.parametrize(
    "diff, margin, effect",
    [
        ([0.10, 0.12, 0.11, 0.09], 0.05, "melhora"),
        ([-0.10, -0.12, -0.11, -0.09], 0.05, "piora"),
        ([0.01, 0.02, 0.015, 0.012], 0.05, "indiferente"),  # within the margin
        ([0.30, -0.10, 0.25, -0.05], 0.05, "indiferente"),  # interval includes 0
    ],
)
def test_classify(diff, margin, effect):
    assert _classify(np.array(diff), margin)["effect"] == effect


def test_compare_rules_loss_is_zero_at_the_best_d_and_sem_pca_is_reported():
    X, y = synthetic(n=400, p=20, K=4, separation=6.0)
    df = run_sweep(
        X, y, "kmeans", range(1, 20), 4, dataset="synthetic", seeds=[0, 1, 2]
    )
    rules = rule_ds(X, y, "synthetic", [0, 1, 2], 4, n_perm=20)
    best = best_d(summarize(df, "ari_test"))["d_best"]
    rules = pd.concat(
        [
            rules,
            pd.DataFrame(
                {"seed": [0, 1, 2], "rule": "oracle", "d": best, "d_used": best}
            ),
        ]
    )
    out = compare_rules(df, rules, "ari_test").set_index("rule")
    assert out.loc["oracle", "loss_mean"] == pytest.approx(0.0)
    assert "sem_pca" in out.index
    assert out.loc["mp_median", "d_median"] == 3  # signal in K-1 = 3 dimensions
    assert (out["n_missing"] == 0).all()
    assert out.attrs["d_best"] == best


def test_rule_ds_clips_d_to_at_least_one():
    X = np.random.default_rng(0).standard_normal((200, 10))
    y = np.repeat(np.arange(2), 100)
    rules = rule_ds(X, y, "synthetic", range(5), 2, n_perm=20)
    assert (rules["d_used"] == rules["d"].clip(lower=1)).all()
    assert (rules["d"] == 0).any()  # pure noise: some rule finds no component


def test_rule_ds_adds_the_estimated_K():
    with_K = rule_ds(
        *synthetic(separation=10.0),
        "synthetic",
        [0],
        4,
        n_perm=10,
        algo_name="kmeans",
        K_grid=range(2, 7),
    ).set_index("rule")
    assert with_K.loc["K_est", "d"] == 4 and with_K.loc["K_est-1", "d"] == 3


def test_compare_rules_with_a_single_seed():
    X, y = synthetic(n=200)
    df = run_sweep(X, y, "kmeans", range(1, 6), 4, dataset="synthetic", seeds=[0])
    rules = rule_ds(X, y, "synthetic", [0], 4, n_perm=10)
    out = compare_rules(df, rules, "ari_test")
    assert np.isnan(out.attrs["score_oracle_cv"])
    found = out[out["n_missing"] == 0]
    assert set(found["effect"]) == {"indiferente"}  # no interval with one seed
    # Rules whose d is outside the grid have no score and no effect.
    assert out.loc[out["n_missing"] == 1, "effect"].isna().all()
    with pytest.raises(ValueError):
        oracle_cv(df, "ari_test")
