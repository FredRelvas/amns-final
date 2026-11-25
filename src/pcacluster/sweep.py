"""Sweep over the number of PCA components d and comparison with the rules.

run_sweep produces one row per (seed, d). The other functions read that table:
summarize (curve with error bars), best_d (argmax and plateau), effect_vs_full
(better, worse or indifferent against clustering without PCA), rule_ds (d chosen
by each label-free rule on the same training data) and compare_rules (loss of
each rule against the best d).
"""

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

from pcacluster.clustering import ALGOS, estimate_K
from pcacluster.data import STANDARDIZE, Preprocessor, split
from pcacluster.metrics import acc, ari, nmi
from pcacluster.rules import (
    d_elbow,
    d_from_K,
    d_gavish_donoho,
    d_kaiser,
    d_marchenko_pastur,
    d_parallel_analysis,
    d_variance_threshold,
)

# Offset added to the seed to fit a second clustering in the full space that
# only differs in initialization (stability baseline of agreement_full).
STABILITY_SEED_OFFSET = 10_000


def prepare(
    X: np.ndarray,
    y: np.ndarray,
    dataset: str,
    seed: int,
    test_size: float = 0.3,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Split (unless test_size=0) and preprocess with statistics of the
    training part only. Returns X_train, X_test, y_train, y_test; the test
    arrays are None when test_size=0.

    run_sweep and rule_ds both call this, so the rules see exactly the
    training data the sweep clusters.
    """
    if not 0 <= test_size < 1:
        raise ValueError(f"test_size must be in [0, 1), got {test_size}")
    if test_size == 0:
        X_train, X_test, y_train, y_test = X, None, y, None
    else:
        X_train, X_test, y_train, y_test = split(X, y, test_size=test_size, seed=seed)
    pre = Preprocessor(scale=dataset in STANDARDIZE).fit(X_train)
    X_test = None if X_test is None else pre.transform(X_test)
    return pre.transform(X_train), X_test, y_train, y_test


def _scores(y_true: np.ndarray | None, labels: np.ndarray | None) -> dict[str, float]:
    if y_true is None or labels is None:
        return {"ari": np.nan, "nmi": np.nan, "acc": np.nan}
    return {
        "ari": ari(y_true, labels),
        "nmi": nmi(y_true, labels),
        "acc": acc(y_true, labels),
    }


def _evaluate(
    model,
    Z_train: np.ndarray,
    Z_test: np.ndarray | None,
    y_train: np.ndarray,
    y_test: np.ndarray | None,
    reference: np.ndarray,
) -> dict[str, float]:
    """Metrics of a fitted clustering against the labels on train and test,
    and its agreement (ARI) with the reference labels of the full-space
    clustering, on test or, without split, on train."""
    pred_train = model.predict(Z_train)
    pred_test = None if Z_test is None else model.predict(Z_test)
    tr, te = _scores(y_train, pred_train), _scores(y_test, pred_test)
    return (
        {f"{m}_train": v for m, v in tr.items()}
        | {f"{m}_test": v for m, v in te.items()}
        | {
            "agreement_full": ari(
                reference, pred_train if pred_test is None else pred_test
            )
        }
    )


def _recon_errors(X: np.ndarray, Z: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Reconstruction error of X with the first 1, 2, ..., m components.

    Z holds the m PCA scores of X. The components are orthonormal, so by
    Pythagoras ||x - x_hat_d||^2 = ||x - mean||^2 - sum_{j<=d} z_j^2, which
    gives every d at once without reconstructing X. Entry d-1 is the error
    with d components (same value as metrics.reconstruction_error).
    """
    total = np.sum((X - center) ** 2, axis=1)
    kept = np.cumsum(Z**2, axis=1)
    return np.clip(np.mean(total[:, None] - kept, axis=0), 0.0, None)


def run_sweep(
    X: np.ndarray,
    y: np.ndarray,
    algo_name: str,
    d_grid: Sequence[int],
    K: int,
    *,
    dataset: str,
    seeds: Sequence[int],
    test_size: float = 0.3,
    estimate_K_flag: bool = False,
    K_grid: Sequence[int] = range(2, 21),
    criterion: str = "bic",
) -> pd.DataFrame:
    """Cluster in the PCA space for every d in d_grid and every seed.

    Preprocessor and PCA are fitted on the training split only; the clustering
    is fitted on train and predicted on test. test_size=0 skips the split:
    everything is fitted and evaluated on all data and the test columns are
    NaN (used where n is small, see docs/DECISOES.md). Returns one row per (seed, d)
    with columns: dataset, n, p, p_over_n, seed, algo, d, K, ari_train,
    ari_test, nmi_test, recon_err_test, agreement_full (ARI against the
    clustering in the full dimension).

    Besides the PCA rows (space='pca'), each seed gets one row with the
    clustering of the preprocessed data without PCA (space='full', d = p_eff),
    the baseline for "better, worse or indifferent". Extra columns:
    - n_train, p_eff: rows used to fit and columns left after preprocessing;
      the sweep goes up to d = min(n_train - 1, p_eff), and larger values of
      d_grid are skipped for that seed with a warning;
    - K_fit: number of clusters actually fitted. Equal to K, or, with
      estimate_K_flag=True, estimated by estimate_K(K_grid, criterion) in the
      space being clustered (full space or PCA with d components);
    - nmi_train, acc_train, acc_test, recon_err_train;
    - agreement_baseline: ARI between two clusterings of the full space that
      only differ in initialization (seed and seed + STABILITY_SEED_OFFSET).
      agreement_full is only meaningful against it. Both are measured on the
      test split, or on all data when test_size=0.
    """
    if algo_name not in ALGOS:
        raise ValueError(f"Unknown algo_name: {algo_name!r}")
    grid = sorted({int(d) for d in d_grid})
    if not grid or grid[0] < 1:
        raise ValueError("d_grid must contain positive integers only.")

    def fit_K(Z: np.ndarray, seed: int) -> int:
        if not estimate_K_flag:
            return K
        return estimate_K(Z, algo_name, K_grid, criterion=criterion, seed=seed)

    n, p = X.shape
    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = prepare(X, y, dataset, seed, test_size)
        n_train, p_eff = X_train.shape
        X_eval = X_train if X_test is None else X_test
        d_max = min(n_train - 1, p_eff)
        ds = [d for d in grid if d <= d_max]
        if len(ds) < len(grid):
            warnings.warn(
                f"seed {seed}: d > {d_max} = min(n_train - 1, p_eff) skipped",
                stacklevel=2,
            )
        base = {
            "dataset": dataset,
            "n": n,
            "p": p,
            "p_over_n": p / n,
            "n_train": n_train,
            "p_eff": p_eff,
            "seed": seed,
            "algo": algo_name,
            "K": K,
        }

        K_full = fit_K(X_train, seed)
        full = ALGOS[algo_name](seed=seed).fit(X_train, K_full)
        full_eval = full.predict(X_eval)
        other = ALGOS[algo_name](seed=seed + STABILITY_SEED_OFFSET).fit(X_train, K_full)
        base["agreement_baseline"] = ari(full_eval, other.predict(X_eval))
        rows.append(
            base
            | {
                "space": "full",
                "d": p_eff,
                "K_fit": K_full,
                "recon_err_train": 0.0,
                "recon_err_test": np.nan if X_test is None else 0.0,
            }
            | _evaluate(full, X_train, X_test, y_train, y_test, full_eval)
        )
        if not ds:
            continue

        # The components are nested: the PCA with d components is the first
        # d columns of the PCA with the largest d, so it is fitted once.
        pca = PCA(n_components=ds[-1], svd_solver="full").fit(X_train)
        Z_train = pca.transform(X_train)
        rec_train = _recon_errors(X_train, Z_train, pca.mean_)
        Z_test = None if X_test is None else pca.transform(X_test)
        rec_test = None if X_test is None else _recon_errors(X_test, Z_test, pca.mean_)

        for d in ds:
            K_d = fit_K(Z_train[:, :d], seed)
            model = ALGOS[algo_name](seed=seed).fit(Z_train[:, :d], K_d)
            rows.append(
                base
                | {
                    "space": "pca",
                    "d": d,
                    "K_fit": K_d,
                    "recon_err_train": rec_train[d - 1],
                    "recon_err_test": np.nan if rec_test is None else rec_test[d - 1],
                }
                | _evaluate(
                    model,
                    Z_train[:, :d],
                    None if Z_test is None else Z_test[:, :d],
                    y_train,
                    y_test,
                    full_eval,
                )
            )
    return pd.DataFrame(rows)


def default_score(df: pd.DataFrame, metric: str = "ari") -> str:
    """Column to compare d values: the test metric, or the train metric when
    the sweep ran without split (test_size=0, test columns all NaN)."""
    return f"{metric}_test" if df[f"{metric}_test"].notna().any() else f"{metric}_train"


def _complete_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the d values present for every seed, so that means over d
    are always taken over the same repetitions."""
    pca = df[df["space"] == "pca"]
    n_seeds = df["seed"].nunique()
    counts = pca.groupby("d")["seed"].nunique()
    return pca[pca["d"].isin(counts.index[counts == n_seeds])]


def summarize(df: pd.DataFrame, score: str) -> pd.DataFrame:
    """Curve of score by d: mean, standard deviation between seeds (the error
    bar) and number of seeds. Only d values present for every seed."""
    pca = _complete_seeds(df)
    out = pca.groupby("d")[score].agg(mean="mean", std="std", n_seeds="count")
    return out.reset_index()


def best_d(curve: pd.DataFrame, frac: float = 0.95) -> dict[str, float]:
    """Best d of a curve from summarize.

    d_best is the argmax of the mean. d_plateau is the smallest d whose mean
    reaches frac of the maximum: the curves usually form a plateau and the
    argmax inside it is noisy (section 4.5 of docs/CONTEXTO_PROJETO.md).
    """
    i = int(curve["mean"].to_numpy().argmax())
    best = float(curve["mean"].iloc[i])
    reach = curve[curve["mean"] >= frac * best]
    return {
        "d_best": int(curve["d"].iloc[i]),
        "score_best": best,
        "d_plateau": int(reach["d"].min()),
        "score_plateau": float(reach.sort_values("d")["mean"].iloc[0]),
    }


def oracle_cv(df: pd.DataFrame, score: str) -> pd.DataFrame:
    """Leave-one-seed-out estimate of the score of the best d.

    The maximum of a mean curve over many d is biased upwards: noise that
    happens to be high is selected. For each seed s, d is chosen as the argmax
    of the mean curve of the other seeds and scored on seed s. The mean of the
    returned score is an unbiased estimate of what choosing d with labels
    achieves on a new repetition. Returns one row per seed: seed, d, score.
    """
    pca = _complete_seeds(df)
    table = pca.pivot_table(index="d", columns="seed", values=score)
    if table.shape[1] < 2:
        raise ValueError("oracle_cv needs at least two seeds.")
    rows = []
    for s in table.columns:
        d = int(table.drop(columns=s).mean(axis=1).idxmax())
        rows.append({"seed": s, "d": d, "score": float(table.loc[d, s])})
    return pd.DataFrame(rows)


def _classify(diff: np.ndarray, margin: float) -> dict[str, float | str]:
    """Paired differences (one per seed) to melhora, piora or indiferente.

    melhora needs the mean difference above the margin (relevant) and the 95%
    t interval of the mean above zero (not explained by seed-to-seed noise);
    piora is the mirror case. Anything else is indiferente: within the margin,
    or with an interval that includes zero.
    """
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        keys = ("diff_mean", "diff_std", "ci_low", "ci_high")
        return dict.fromkeys(keys, np.nan) | {"margin": margin, "effect": None}
    mean = float(np.mean(diff))
    half = np.nan
    if len(diff) > 1:
        sem = float(stats.sem(diff))
        half = float(stats.t.ppf(0.975, len(diff) - 1) * sem) if sem > 0 else 0.0
    low, high = mean - half, mean + half
    if mean > margin and low > 0:
        effect = "melhora"
    elif mean < -margin and high < 0:
        effect = "piora"
    else:
        effect = "indiferente"
    return {
        "diff_mean": mean,
        "diff_std": float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan,
        "ci_low": low,
        "ci_high": high,
        "margin": margin,
        "effect": effect,
    }


def effect_vs_full(df: pd.DataFrame, score: str, k: float = 1.0) -> pd.DataFrame:
    """Classify each d as melhora, piora or indiferente against no PCA.

    For each seed the difference score(PCA with d) - score(no PCA) is paired:
    same split and same initialization seed, so variation that both share
    cancels out. The margin is k times the standard deviation of the no-PCA
    score between seeds (k = 1 by default, the group's proposal in section
    4.3 of docs/CONTEXTO_PROJETO.md); see _classify for the rule, which also
    requires the paired 95% interval to exclude zero.
    """
    full = df[df["space"] == "full"].set_index("seed")[score]
    margin = k * float(full.std(ddof=1))
    pca = _complete_seeds(df)
    rows = []
    for d, g in pca.groupby("d"):
        diff = (g.set_index("seed")[score] - full).to_numpy(dtype=float)
        rows.append({"d": int(d)} | _classify(diff, margin))
    return pd.DataFrame(rows)


def rule_ds(
    X: np.ndarray,
    y: np.ndarray,
    dataset: str,
    seeds: Sequence[int],
    K: int,
    *,
    test_size: float = 0.3,
    algo_name: str | None = None,
    K_grid: Sequence[int] = range(2, 21),
    criterion: str = "bic",
    n_perm: int = 100,
) -> pd.DataFrame:
    """d chosen by every label-free rule on the training data of each seed
    (the same X_train the sweep uses, via prepare).

    With algo_name, K is also estimated in the full space (estimate_K with
    K_grid and criterion) and the rules d = K_est and d = K_est - 1 are added:
    the professor's procedure, PCA with the K computed by the algorithm.

    A rule can return 0 (no component above the noise) but the clustering
    needs at least one dimension, so d_used = max(d, 1). Returns one row per
    (seed, rule) with columns seed, rule, d, d_used.
    """
    rows = []
    for seed in seeds:
        X_train, *_ = prepare(X, y, dataset, seed, test_size)
        found = {
            "mp_median": d_marchenko_pastur(X_train, seed=seed),
            "mp_sigma1": d_marchenko_pastur(X_train, sigma2=1),
            "parallel": d_parallel_analysis(X_train, n_perm=n_perm, seed=seed),
            "gavish_donoho": d_gavish_donoho(X_train),
            "K": d_from_K(K),
            "K-1": d_from_K(K, minus_one=True),
            "var90": d_variance_threshold(X_train),
            "kaiser": d_kaiser(X_train),
            "elbow": d_elbow(X_train),
        }
        if algo_name is not None:
            K_est = estimate_K(
                X_train, algo_name, K_grid, criterion=criterion, seed=seed
            )
            found["K_est"] = d_from_K(K_est)
            found["K_est-1"] = d_from_K(K_est, minus_one=True)
        d_max = min(X_train.shape[0] - 1, X_train.shape[1])
        for rule, d in found.items():
            rows.append(
                {
                    "seed": seed,
                    "rule": rule,
                    "d": d,
                    "d_used": int(np.clip(d, 1, d_max)),
                }
            )
    return pd.DataFrame(rows)


def compare_rules(
    df: pd.DataFrame,
    rules: pd.DataFrame,
    score: str,
    frac: float = 0.95,
    k: float = 1.0,
) -> pd.DataFrame:
    """Loss of each rule against the best d of the sweep, with no PCA as a
    reference row.

    For each seed, the rule is scored at its own d (d_used of that seed) and
    compared, paired, with the same seed at d_best (argmax of the mean curve).
    Columns:
    - d_median, d_min, d_max: d chosen by the rule across seeds;
    - score_mean, score_std: score at the rule's d;
    - loss_mean, loss_std: score(d_best) - score(rule's d), paired by seed;
    - loss_cv: mean score of oracle_cv minus score_mean, the same loss
      against a reference that is not biased upwards;
    - in_plateau: fraction of seeds whose d reaches frac of the best mean;
    - effect, diff_mean, ci_low, ci_high: classification against no PCA as in
      effect_vs_full (margin k times the seed-to-seed std of no PCA);
    - n_missing: seeds whose d is not in the sweep (excluded from the means).
    """
    pca = df[df["space"] == "pca"].set_index(["seed", "d"])[score]
    full = df[df["space"] == "full"].set_index("seed")[score]
    margin = k * float(full.std(ddof=1))
    curve = summarize(df, score)
    best = best_d(curve, frac)
    threshold = frac * best["score_best"]
    at_best = pca.xs(best["d_best"], level="d")
    n_seeds = df["seed"].nunique()
    cv_mean = float(oracle_cv(df, score)["score"].mean()) if n_seeds > 1 else np.nan

    def entry(name: str, chosen: pd.Series, s: pd.Series) -> dict:
        found = s.dropna()
        loss = (at_best.reindex(found.index) - found).to_numpy(dtype=float)
        diff = (found - full.reindex(found.index)).to_numpy(dtype=float)
        return {
            "rule": name,
            "d_median": float(chosen.median()),
            "d_min": int(chosen.min()),
            "d_max": int(chosen.max()),
            "score_mean": float(found.mean()),
            "score_std": float(found.std(ddof=1)),
            "loss_mean": float(loss.mean()) if len(loss) else np.nan,
            "loss_std": float(loss.std(ddof=1)) if len(loss) > 1 else np.nan,
            "loss_cv": cv_mean - float(found.mean()),
            "in_plateau": float(np.mean(found >= threshold)),
            "n_missing": int(s.isna().sum()),
        } | {
            key: value
            for key, value in _classify(diff, margin).items()
            if key in ("effect", "diff_mean", "ci_low", "ci_high")
        }

    rows = []
    for name, g in rules.groupby("rule", sort=False):
        chosen = g.set_index("seed")["d_used"]
        s = pd.Series(
            [pca.get((seed, d), np.nan) for seed, d in chosen.items()],
            index=chosen.index,
            dtype=float,
        )
        rows.append(entry(name, chosen, s))
    p_eff = df[df["space"] == "full"].set_index("seed")["p_eff"]
    rows.append(entry("sem_pca", p_eff, full.astype(float)))
    out = pd.DataFrame(rows)
    out.attrs |= best | {"score_oracle_cv": cv_mean, "score": score, "margin": margin}
    return out
