"""Sweep along p/n (issue #6): runs scripts/run_sweep.py for every n and stacks
the results.

Examples:
    uv run python scripts/run_pn.py --name mnist --dataset mnist \
        --ns 70000,35000,10000,3500,1400,700,500,350 --no-split-max 3500
    uv run python scripts/run_pn.py --name synthetic_s5 --dataset synthetic \
        --synthetic 1000,1,5 --ns 4000,2000,1000,500,250,125 --no-split-max 4000

Each (n, split) is one call to run_sweep.py with its own tag; a run whose
sweep.csv already exists is skipped, so an interrupted sweep resumes. With
--analyze-only nothing is run. Writes to results/pn_<name>/:
- rules.csv: per n, split, algorithm and rule, the difference to oracle_cv and
  whether the rule hits the best d (pn.rule_vs_oracle);
- breaks.csv: breaking point of each rule (pn.breaking_point);
- gap.csv and gap_by_d.csv: train-test difference (H4, runs with split only);
- rules.png and gap.png.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from pcacluster.plots import plot_pn_gap, plot_pn_rules
from pcacluster.pn import breaking_point, has_signal, rule_vs_oracle, train_test_gap
from pcacluster.sweep import default_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def runs(args) -> list[tuple[int, bool, str]]:
    """(n, split, tag) of every run, from the largest n to the smallest."""
    out = []
    for n in sorted(args.ns, reverse=True):
        out.append((n, True, f"pn_{args.name}/n{n}"))
        if n <= args.no_split_max:
            out.append((n, False, f"pn_{args.name}/n{n}_sem-split"))
    return out


def run_one(args, n: int, split: bool, tag: str) -> None:
    if (RESULTS / tag / "sweep.csv").exists():
        print(f"{tag}: já existe, pulando", flush=True)
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_sweep.py"),
        "--dataset",
        args.dataset,
        "--n",
        str(n),
        "--grid",
        args.grid,
        "--seeds",
        args.seeds,
        "--tag",
        tag,
        "--algos",
        *args.algos,
    ]
    if args.synthetic:
        cmd += ["--synthetic", args.synthetic]
    if not split:
        cmd += ["--test-size", "0"]
    print(f"{tag}: rodando", flush=True)
    subprocess.run(cmd, check=True)


def analyze(args) -> None:
    out = RESULTS / f"pn_{args.name}"
    rule_rows, gap_rows, gap_by_d = [], [], []
    for n, split, tag in runs(args):
        d = RESULTS / tag
        if not (d / "sweep.csv").exists():
            continue
        df = pd.read_csv(d / "sweep.csv")
        table = pd.read_csv(d / "rules.csv")
        score = default_score(df)
        p_over_n = float(df["p_over_n"].iloc[0])
        info = {"n": n, "p_over_n": p_over_n, "split": split, "score": score}
        for algo, sub in df.groupby("algo"):
            rules = table[table["algo"] == algo].drop(columns="algo")
            rule_rows.append(
                rule_vs_oracle(sub, rules, score).assign(algo=algo, **info)
            )
            if split:
                by_d, summary = train_test_gap(sub)
                gap_by_d.append(by_d.assign(algo=algo, **info))
                gap_rows.append(summary | {"algo": algo} | info)

    rules = pd.concat(rule_rows, ignore_index=True)
    rules["signal"] = False
    for _, g in rules.groupby(["split", "algo"]):
        ref = g[g["rule"] == "sem_pca"]
        sig = has_signal(ref["p_over_n"], ref["oracle_cv"])
        rules.loc[g.index, "signal"] = g["n"].map(dict(zip(ref["n"], sig))).astype(bool)
    rules.to_csv(out / "rules.csv", index=False)
    breaks = []
    for (split, algo, rule), g in rules.groupby(["split", "algo", "rule"]):
        no_signal = g.loc[~g["signal"], "p_over_n"]
        breaks.append(
            {"split": split, "algo": algo, "rule": rule}
            | breaking_point(g["p_over_n"], g["hit"], g["signal"])
            | {"collapse": no_signal.min() if len(no_signal) else None}
        )
    breaks = pd.DataFrame(breaks)
    breaks.to_csv(out / "breaks.csv", index=False)
    plot_pn_rules(rules, args.name, out / "rules.png")

    if gap_rows:
        gap = pd.DataFrame(gap_rows)
        gap.to_csv(out / "gap.csv", index=False)
        pd.concat(gap_by_d, ignore_index=True).to_csv(out / "gap_by_d.csv", index=False)
        plot_pn_gap(gap, args.name, out / "gap.png")
        with pd.option_context("display.width", 200):
            print(
                gap[["algo", "n", "p_over_n", "gap_full", "gap_cv", "effect"]]
                .round(3)
                .to_string(index=False)
            )
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(breaks.to_string(index=False))
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="results go to results/pn_<name>")
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--ns", required=True, type=lambda s: [int(x) for x in s.split(",")]
    )
    ap.add_argument(
        "--no-split-max",
        type=int,
        default=0,
        help="also run without split for n <= this value",
    )
    ap.add_argument(
        "--grid",
        default="1-30,35-100:5,120-600:40",
        help="d grid, as in run_sweep.py (default: the #5 MNIST grid)",
    )
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--algos", nargs="+", default=["kmeans", "gmm"])
    ap.add_argument("--synthetic", default="", help="p,noise_std,separation")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    (RESULTS / f"pn_{args.name}").mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        for n, split, tag in runs(args):
            run_one(args, n, split, tag)
    analyze(args)


if __name__ == "__main__":
    main()
