"""Sweep over d for one dataset and comparison with the rules (issue #5).

Example:
    uv run python scripts/run_sweep.py --dataset digits
    uv run python scripts/run_sweep.py --dataset mnist --n 10000 --grid 1-30,35-100:5,120-600:40
    uv run python scripts/run_sweep.py --dataset tumors --test-size 0

Writes to results/<tag>/: sweep.csv (one row per algorithm, seed and d),
rules.csv (d of each rule per seed), compare.csv (loss of each rule against
the best d), effect.csv (melhora/piora/indiferente by d), best.csv and the
figures curves.png and effect.png. With --reuse, sweep.csv and rules.csv are
read back and only the analysis and the figures are redone.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pcacluster import SEEDS
from pcacluster.data import load_dataset, subsample
from pcacluster.plots import plot_curves, plot_effect
from pcacluster.sweep import (
    best_d,
    compare_rules,
    default_score,
    effect_at_oracle_cv,
    effect_vs_full,
    oracle_cv,
    prepare,
    rule_ds,
    run_sweep,
    summarize,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def parse_grid(spec: str) -> list[int] | None:
    """'all' -> None (every d up to the PCA limit); otherwise comma-separated
    ranges 'a-b' or 'a-b:step' and single values, e.g. '1-30,35-100:5'."""
    if spec == "all":
        return None
    grid = set()
    for part in spec.split(","):
        rng, _, step = part.partition(":")
        lo, _, hi = rng.partition("-")
        grid.update(range(int(lo), int(hi or lo) + 1, int(step or 1)))
    return sorted(grid)


def parse_seeds(spec: str) -> list[int]:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def compute(args, seeds: list[int]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Rules for every seed, then the sweep over the grid plus every d the
    rules chose. Returns the sweep and the rules of each algorithm."""
    syn = {}
    if args.synthetic:
        p, noise, sep = args.synthetic.split(",")
        syn = {"p": int(p), "noise_std": float(noise), "separation": float(sep)}

    # The seed also draws the data when it is generated or subsampled.
    base = None if args.dataset == "synthetic" else load_dataset(args.dataset)

    def data(seed: int) -> tuple[np.ndarray, np.ndarray]:
        if base is None:
            return load_dataset(
                "synthetic", n=args.n or 1000, seed=seed, n_clusters=args.K, **syn
            )
        return base if args.n is None else subsample(*base, args.n, seed)

    t0 = time.time()
    rules = {}
    for algo in args.algos:
        parts = [
            rule_ds(
                *data(s),
                args.dataset,
                [s],
                args.K,
                test_size=args.test_size,
                algo_name=algo if args.k_est else None,
                criterion=args.k_est or "bic",
                n_perm=args.n_perm,
            )
            for s in seeds
        ]
        rules[algo] = pd.concat(parts, ignore_index=True)
        if not args.k_est:
            # Without K estimation the rules do not depend on the algorithm.
            rules |= {a: rules[algo] for a in args.algos}
            break
    print(f"rules: {time.time() - t0:.0f} s")

    grid = parse_grid(args.grid)
    if grid is None:
        # Every d up to the largest PCA limit among the seeds; run_sweep skips
        # the values above the limit of each seed.
        d_max = 0
        for s in seeds:
            X_train, *_ = prepare(*data(s), args.dataset, s, args.test_size)
            d_max = max(d_max, min(X_train.shape[0] - 1, X_train.shape[1]))
        grid = range(1, d_max + 1)
    used = {int(d) for r in rules.values() for d in r["d_used"]}

    frames = []
    for algo in args.algos:
        for s in seeds:
            t = time.time()
            frames.append(
                run_sweep(
                    *data(s),
                    algo,
                    sorted(set(grid) | used),
                    args.K,
                    dataset=args.dataset,
                    seeds=[s],
                    test_size=args.test_size,
                )
            )
            print(f"{algo} seed {s}: {time.time() - t:.0f} s", flush=True)
    return pd.concat(frames, ignore_index=True), rules


def analyze(df: pd.DataFrame, rules: dict[str, pd.DataFrame], out: Path) -> None:
    """Tables and figures of a finished sweep."""
    score = default_score(df)
    compares, effects, bests, curve_panels, effect_panels = [], [], [], {}, {}
    for algo in df["algo"].unique():
        sub = df[df["algo"] == algo]
        full = sub[sub["space"] == "full"]
        curve = summarize(sub, score)
        cmp = compare_rules(sub, rules[algo], score)
        eff = effect_vs_full(sub, score)
        cv = oracle_cv(sub, score)
        ecv = effect_at_oracle_cv(sub, score)
        compares.append(cmp.assign(algo=algo))
        effects.append(eff.assign(algo=algo))
        bests.append(
            best_d(curve)
            | {
                "algo": algo,
                "score": score,
                "oracle_cv": cv["score"].mean(),
                "oracle_cv_d": ",".join(map(str, cv["d"])),
                "effect_cv": ecv["effect"],
                "diff_cv": ecv["diff_mean"],
                "ci_cv_low": ecv["ci_low"],
                "ci_cv_high": ecv["ci_high"],
                "margin": cmp.attrs["margin"],
                # One value per seed, taken from the no-PCA rows.
                "agreement_baseline": full["agreement_baseline"].mean(),
            }
        )
        curve_panels[algo] = (curve, full, cmp)
        effect_panels[algo] = eff

    pd.concat(compares).to_csv(out / "compare.csv", index=False)
    pd.concat(effects).to_csv(out / "effect.csv", index=False)
    pd.DataFrame(bests).to_csv(out / "best.csv", index=False)

    row = df.iloc[0]
    split = (
        "sem split"
        if df["ari_test"].isna().all()
        else f"teste de {1 - row['n_train'] / row['n']:.0%}"
    )
    title = (
        f"{row['dataset']}: n = {row['n']}, p = {row['p']}, "
        f"p/n = {row['p_over_n']:.3g}, {df['seed'].nunique()} sementes, {split}"
    )
    plot_curves(curve_panels, score, title, out / "curves.png")
    plot_effect(effect_panels, score, title, out / "effect.png")

    cols = ["algo", "rule", "d_median", "score_mean", "loss_mean", "loss_cv"]
    cols += ["in_plateau", "effect"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(bests).round(3).to_string(index=False))
        print(pd.concat(compares)[cols].round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, help="subsample (or synthetic) size per seed")
    ap.add_argument("--K", type=int, default=10, help="true number of classes")
    ap.add_argument("--algos", nargs="+", default=["kmeans", "gmm"])
    ap.add_argument("--seeds", default=f"{SEEDS[0]}-{SEEDS[-1]}")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--grid", default="all")
    ap.add_argument("--n-perm", type=int, default=100, help="parallel analysis")
    ap.add_argument(
        "--k-est",
        choices=["bic", "silhouette"],
        help="also test d = K estimated in the full space with this criterion",
    )
    ap.add_argument("--synthetic", default="", help="p,noise_std,separation")
    ap.add_argument("--tag")
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="read sweep.csv and rules.csv of the tag and redo only the analysis",
    )
    args = ap.parse_args()

    tag = args.tag or "_".join(
        [args.dataset]
        + ([f"n{args.n}"] if args.n else [])
        + (["sem-split"] if args.test_size == 0 else [])
        + ([f"kest-{args.k_est}"] if args.k_est else [])
    )
    out = RESULTS / tag
    t0 = time.time()
    if args.reuse:
        df = pd.read_csv(out / "sweep.csv")
        table = pd.read_csv(out / "rules.csv")
        rules = {a: g.drop(columns="algo") for a, g in table.groupby("algo")}
    else:
        out.mkdir(parents=True, exist_ok=True)
        df, rules = compute(args, parse_seeds(args.seeds))
        df.to_csv(out / "sweep.csv", index=False)
        pd.concat([r.assign(algo=a) for a, r in rules.items()]).to_csv(
            out / "rules.csv", index=False
        )
    analyze(df, rules, out)
    print(f"total: {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
