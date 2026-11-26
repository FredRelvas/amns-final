"""Tests for the analysis across p/n (issue #6)."""

import pandas as pd
import pytest

from pcacluster.data import load_dataset
from pcacluster.pn import breaking_point, has_signal, rule_vs_oracle, train_test_gap
from pcacluster.sweep import rule_ds, run_sweep


def bp(hits):
    pn = pd.Series([0.01, 0.1, 1.0, 10.0])
    return breaking_point(pn, pd.Series(hits))


def test_breaking_point_never_breaks():
    assert bp([True, True, True, True]) == {"last_hit": 10.0, "first_miss": None}


def test_breaking_point_breaks_between_last_hit_and_first_miss():
    assert bp([True, True, False, False]) == {"last_hit": 0.1, "first_miss": 1.0}


def test_breaking_point_ignores_isolated_misses_before_the_break():
    # A miss at 0.1 followed by a hit at 1.0 is noise, not the break.
    assert bp([True, False, True, False]) == {"last_hit": 1.0, "first_miss": 10.0}


def test_breaking_point_misses_everywhere():
    assert bp([False, False, False, False]) == {"last_hit": None, "first_miss": 0.01}


def test_breaking_point_does_not_depend_on_order():
    pn = pd.Series([10.0, 0.01, 1.0, 0.1])
    hits = pd.Series([False, True, False, True])
    assert breaking_point(pn, hits) == {"last_hit": 0.1, "first_miss": 1.0}


def test_breaking_point_ignores_points_without_signal():
    # At 1.0 and 10.0 nothing clusters: the "hits" there are trivial.
    pn = pd.Series([0.01, 0.1, 1.0, 10.0])
    hits = pd.Series([True, False, True, True])
    signal = has_signal(pn, pd.Series([0.9, 0.8, 0.2, 0.05]))
    assert signal.tolist() == [True, True, False, False]
    assert breaking_point(pn, hits, signal) == {"last_hit": 0.01, "first_miss": 0.1}


def fake_sweep(scores: dict[int, list[float]], full: list[float]) -> pd.DataFrame:
    """Sweep table with ari_test by d (one value per seed) and no PCA."""
    rows = []
    for seed, f in enumerate(full):
        rows.append(
            {
                "seed": seed,
                "space": "full",
                "d": 50,
                "p_eff": 50,
                "ari_test": f,
                "ari_train": f,
            }
        )
        for d, values in scores.items():
            rows.append(
                {
                    "seed": seed,
                    "space": "pca",
                    "d": d,
                    "p_eff": 50,
                    "ari_test": values[seed],
                    "ari_train": values[seed] + 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_rule_vs_oracle_flags_a_rule_far_below_the_best_d():
    df = fake_sweep(
        {2: [0.30, 0.31, 0.29, 0.30], 9: [0.90, 0.91, 0.89, 0.90]},
        full=[0.80, 0.82, 0.78, 0.80],
    )
    rules = pd.DataFrame(
        {
            "seed": [0, 1, 2, 3] * 2,
            "rule": ["good"] * 4 + ["bad"] * 4,
            "d_used": [9] * 4 + [2] * 4,
        }
    )
    out = rule_vs_oracle(df, rules, "ari_test").set_index("rule")
    assert out.loc["good", "hit"]
    assert out.loc["good", "diff_mean"] == pytest.approx(0.0)
    assert not out.loc["bad", "hit"]
    assert not out.loc["sem_pca", "hit"]


def test_train_test_gap_needs_split():
    df = fake_sweep({9: [0.9, 0.9]}, full=[0.8, 0.8]).assign(ari_test=float("nan"))
    with pytest.raises(ValueError):
        train_test_gap(df)


def test_train_test_gap_on_fake_sweep():
    df = fake_sweep({2: [0.3, 0.3, 0.3], 9: [0.9, 0.91, 0.89]}, full=[0.8, 0.8, 0.8])
    by_d, summary = train_test_gap(df)
    assert set(by_d["d"]) == {"2", "9", "full"}
    assert summary["gap_full"] == pytest.approx(0.0)
    assert summary["gap_cv"] == pytest.approx(0.1)


def test_pn_analysis_on_a_small_real_sweep():
    X, y = load_dataset("synthetic", n=300, seed=0, p=20, n_clusters=4)
    seeds = [0, 1, 2]
    df = run_sweep(X, y, "kmeans", range(1, 20), 4, dataset="synthetic", seeds=seeds)
    rules = rule_ds(X, y, "synthetic", seeds, 4, n_perm=10)
    out = rule_vs_oracle(df, rules, "ari_test")
    assert set(out["rule"]) == set(rules["rule"]) | {"sem_pca"}
    # The signal spans K - 1 = 3 dimensions, which K-means separates easily.
    assert out.set_index("rule").loc["K-1", "hit"]
    _, summary = train_test_gap(df)
    assert summary["effect"] in {"melhora", "piora", "indiferente"}
