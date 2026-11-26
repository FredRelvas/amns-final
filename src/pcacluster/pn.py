"""Analysis across p/n (issue #6): overfitting (H4) and breaking point (H5).

Every function reads the tables of one sweep (one dataset and one n) written by
scripts/run_sweep.py, and reuses the criteria of sweep.py: the score of
default_score, the fair reference oracle_cv and the classification _classify
(margin of k standard deviations of no PCA plus a paired 95% interval).
scripts/run_pn.py stacks the results of every n and finds the breaking point.
"""

import numpy as np
import pandas as pd

from pcacluster.sweep import _classify, _complete_seeds, oracle_cv


def rule_vs_oracle(
    df: pd.DataFrame, rules: pd.DataFrame, score: str, k: float = 1.0
) -> pd.DataFrame:
    """Does each rule reach the best d of the sweep? (H5)

    For each seed, score at the rule's d minus the score of oracle_cv on that
    seed (d chosen with labels on the other seeds, the unbiased reference of
    #5). The paired differences are classified as in effect_vs_full, with the
    margin k times the seed-to-seed std of no PCA: 'piora' means the rule is
    clearly below the best d, so hit = False; 'indiferente' or 'melhora' mean
    hit = True. A row 'sem_pca' (no PCA) is added for reference.
    """
    pca = df[df["space"] == "pca"].set_index(["seed", "d"])[score]
    full = df[df["space"] == "full"].set_index("seed")[score]
    margin = k * float(full.std(ddof=1))
    best = oracle_cv(df, score).set_index("seed")["score"]

    def entry(name: str, d: pd.Series, s: pd.Series) -> dict:
        diff = (s - best.reindex(s.index)).to_numpy(dtype=float)
        c = _classify(diff, margin)
        return {
            "rule": name,
            "d_median": float(d.median()),
            "score_mean": float(s.mean()),
            "oracle_cv": float(best.mean()),
            "diff_mean": c["diff_mean"],
            "ci_low": c["ci_low"],
            "ci_high": c["ci_high"],
            "margin": margin,
            "hit": c["effect"] is not None and c["effect"] != "piora",
        }

    rows = []
    for name, g in rules.groupby("rule", sort=False):
        d = g.set_index("seed")["d_used"]
        s = pd.Series(
            [pca.get((seed, x), np.nan) for seed, x in d.items()],
            index=d.index,
            dtype=float,
        )
        rows.append(entry(name, d, s))
    p_eff = df[df["space"] == "full"].set_index("seed")["p_eff"]
    rows.append(entry("sem_pca", p_eff, full.astype(float)))
    return pd.DataFrame(rows)


def train_test_gap(df: pd.DataFrame, k: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Overfitting as ari_train - ari_test (H4). Needs a sweep with split.

    Returns the gap by d (mean and std between seeds, d = 'full' for no PCA)
    and the classification of gap(no PCA) - gap(PCA at the d of oracle_cv),
    paired by seed: 'melhora' means PCA clearly reduces the gap. The margin is
    k times the seed-to-seed std of the no-PCA gap.
    """
    if df["ari_test"].isna().all():
        raise ValueError("train_test_gap needs a sweep with a train/test split.")
    gap = df.assign(gap=df["ari_train"] - df["ari_test"])
    full = gap[gap["space"] == "full"].set_index("seed")["gap"]
    by_d = (
        _complete_seeds(gap)
        .groupby("d")["gap"]
        .agg(mean="mean", std="std")
        .reset_index()
        .astype({"d": str})
    )
    by_d = pd.concat(
        [
            by_d,
            pd.DataFrame([{"d": "full", "mean": full.mean(), "std": full.std(ddof=1)}]),
        ],
        ignore_index=True,
    )

    cv = oracle_cv(df, "ari_test").set_index("seed")["d"]
    pca = gap[gap["space"] == "pca"].set_index(["seed", "d"])["gap"]
    at_cv = pd.Series([pca[(s, d)] for s, d in cv.items()], index=cv.index)
    diff = (full.reindex(cv.index) - at_cv).to_numpy(dtype=float)
    summary = {"gap_full": float(full.mean()), "gap_cv": float(at_cv.mean())}
    summary |= _classify(diff, k * float(full.std(ddof=1)))
    return by_d, summary


def has_signal(pn: pd.Series, oracle: pd.Series, frac: float = 0.5) -> pd.Series:
    """Whether the problem itself still has structure at each p/n.

    When the best d no longer clusters (oracle_cv near zero), every rule
    "hits" trivially, because every d scores about the same. A p/n counts as
    having signal while oracle_cv keeps at least frac of its value at the
    smallest p/n. frac = 0.5 is our choice, with no source.
    """
    ref = float(oracle[pn.idxmin()])
    return oracle >= frac * ref


def breaking_point(
    pn: pd.Series, hit: pd.Series, signal: pd.Series | None = None
) -> dict:
    """Breaking point of a rule along p/n.

    The rule breaks at the smallest p/n from which it misses at every larger
    p/n. Points without signal (see has_signal) are left out, since hitting
    there is trivial. Returned as the interval (last_hit, first_miss) in p/n:
    - first_miss is None if the rule hits at the largest p/n (never breaks);
    - last_hit is None if it misses already at the smallest p/n.
    """
    if signal is not None:
        keep = signal.to_numpy(dtype=bool)
        pn, hit = pn[keep], hit[keep]
    if len(pn) == 0:
        return {"last_hit": None, "first_miss": None}
    order = np.argsort(pn.to_numpy())
    x = pn.to_numpy(dtype=float)[order]
    h = hit.to_numpy(dtype=bool)[order]
    if h[-1]:
        return {"last_hit": float(x[-1]), "first_miss": None}
    hits = np.flatnonzero(h)
    if len(hits) == 0:
        return {"last_hit": None, "first_miss": float(x[0])}
    return {"last_hit": float(x[hits[-1]]), "first_miss": float(x[hits[-1] + 1])}
