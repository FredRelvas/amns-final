"""Figures of the sweep: score by d and difference against no PCA."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter

SERIES = "#2a78d6"
BASELINE = "#52514e"
INK = "#0b0b0b"
MUTED = "#8a8985"

RULE_LABELS = {
    "mp_median": "MP σ² est.",
    "mp_sigma1": "MP σ²=1",
    "parallel": "análise paralela",
    "gavish_donoho": "Gavish-Donoho",
    "K": "d = K",
    "K-1": "d = K−1",
    "K_est": "d = K est.",
    "K_est-1": "d = K est.−1",
    "var90": "90% var.",
    "kaiser": "Kaiser",
    "elbow": "cotovelo",
}


def _style(ax, log_x: bool) -> None:
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors="#52514e", labelsize=8)


def plot_curves(
    panels: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    score: str,
    title: str,
    path,
) -> None:
    """One column per algorithm.

    Top: mean score by d with a band of +- 1 standard deviation between
    seeds, and the no-PCA mean (dashed, with its band). Bottom, on the same d
    axis: one row per rule, with the median d chosen across seeds (dot) and
    the range from the smallest to the largest (line).

    panels maps algorithm name to (curve from summarize, rows of the sweep
    with space == 'full', comparison from compare_rules).
    """
    first_cmp = next(iter(panels.values()))[2]
    n_rules = int((first_cmp["rule"] != "sem_pca").sum())
    fig, axes = plt.subplots(
        2,
        len(panels),
        figsize=(5.2 * len(panels), 4.0 + 0.17 * n_rules),
        sharex="col",
        sharey="row",
        squeeze=False,
        gridspec_kw={"height_ratios": [3.0, 0.3 + 0.17 * n_rules]},
    )
    for col, (algo, (curve, full, cmp)) in enumerate(panels.items()):
        ax, ax_r = axes[0][col], axes[1][col]
        log_x = curve["d"].max() > 80
        _style(ax, log_x)
        _style(ax_r, log_x)
        d, m, s = curve["d"], curve["mean"], curve["std"].fillna(0)
        ax.fill_between(d, m - s, m + s, color=SERIES, alpha=0.18, linewidth=0)
        ax.plot(d, m, color=SERIES, linewidth=2, label="PCA com d componentes")
        fm, fs = full[score].mean(), full[score].std()
        ax.axhspan(fm - fs, fm + fs, color=BASELINE, alpha=0.10, linewidth=0)
        ax.axhline(fm, color=BASELINE, linewidth=1.2, linestyle="--", label="sem PCA")
        ax.set_title(algo, fontsize=10, color=INK, loc="left")

        rules = cmp[cmp["rule"] != "sem_pca"].reset_index(drop=True)
        y = range(len(rules))[::-1]
        ax_r.hlines(y, rules["d_min"], rules["d_max"], color=INK, linewidth=1.2)
        ax_r.plot(rules["d_median"], y, "o", color=INK, markersize=4)
        ax_r.set_yticks(list(y), [RULE_LABELS.get(r, r) for r in rules["rule"]])
        ax_r.set_ylim(-0.7, len(rules) - 0.3)
        ax_r.grid(False, axis="y")
        ax_r.set_xlabel("d (número de componentes do PCA)", fontsize=8.5)
    axes[0][0].set_ylabel(f"{score}\n(média ± DP entre sementes)", fontsize=8.5)
    axes[1][0].set_ylabel("d da regra", fontsize=8.5)
    axes[0][0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle(title, fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_effect(panels: dict[str, pd.DataFrame], score: str, title: str, path) -> None:
    """One panel per algorithm: mean paired difference score(d) - score(no
    PCA) with its 95% interval, and the margin of indifference shaded.
    panels maps algorithm name to the output of effect_vs_full."""
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.2 * len(panels), 3.4), sharey=True, squeeze=False
    )
    for ax, (algo, eff) in zip(axes[0], panels.items()):
        _style(ax, eff["d"].max() > 80)
        margin = eff["margin"].iloc[0]
        ax.axhspan(
            -margin,
            margin,
            color=BASELINE,
            alpha=0.12,
            linewidth=0,
            label="margem (±1 DP sem PCA)",
        )
        ax.axhline(0, color=BASELINE, linewidth=0.8)
        ax.fill_between(
            eff["d"],
            eff["ci_low"],
            eff["ci_high"],
            color=SERIES,
            alpha=0.18,
            linewidth=0,
            label="IC 95% pareado",
        )
        ax.plot(
            eff["d"],
            eff["diff_mean"],
            color=SERIES,
            linewidth=2,
            label="diferença média",
        )
        ax.set_title(algo, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("d (número de componentes do PCA)", fontsize=8.5)
    axes[0][0].set_ylabel(f"{score}: PCA − sem PCA", fontsize=8.5)
    axes[0][0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle(title, fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pn_rules(rules: pd.DataFrame, title: str, path) -> None:
    """Issue #6. One panel per (split, algorithm): for each rule, the mean
    difference score(rule's d) - score(oracle_cv) along p/n (log axis).
    Filled markers: the rule hits the best d (pn.rule_vs_oracle); hollow: it
    misses. The shaded band is the margin, which changes with n."""
    groups = list(rules.groupby(["split", "algo"]))
    fig, axes = plt.subplots(
        1, len(groups), figsize=(5.2 * len(groups), 3.8), sharey=True, squeeze=False
    )
    colors = plt.get_cmap("tab10").colors
    for ax, ((split, algo), g) in zip(axes[0], groups):
        _style(ax, True)
        _pn_axis(ax, g["p_over_n"])
        if "signal" in g and not g["signal"].all():
            lost = g.loc[~g["signal"].astype(bool), "p_over_n"].min()
            ax.axvspan(lost, g["p_over_n"].max() * 1.15, color=MUTED, alpha=0.10,
                       linewidth=0, label="sem sinal")
        band = g.groupby("p_over_n")["margin"].first()
        ax.fill_between(band.index, -band, band, color=BASELINE, alpha=0.12, linewidth=0)
        ax.axhline(0, color=BASELINE, linewidth=0.8)
        for i, (rule, r) in enumerate(g.groupby("rule", sort=False)):
            r = r.sort_values("p_over_n")
            c = INK if rule == "sem_pca" else colors[i % len(colors)]
            ls = "--" if rule == "sem_pca" else "-"
            label = "sem PCA" if rule == "sem_pca" else RULE_LABELS.get(rule, rule)
            ax.plot(r["p_over_n"], r["diff_mean"], color=c, linestyle=ls,
                    linewidth=1.3, label=label)
            hit = r["hit"].astype(bool)
            ax.plot(r["p_over_n"][hit], r["diff_mean"][hit], "o", color=c, markersize=4)
            ax.plot(r["p_over_n"][~hit], r["diff_mean"][~hit], "o", markersize=4,
                    markerfacecolor="white", markeredgecolor=c)
        ax.set_title(f"{algo}, {'com split' if split else 'sem split'}",
                     fontsize=10, color=INK, loc="left")
        ax.set_xlabel("p/n", fontsize=8.5)
    axes[0][0].set_ylabel("score no d da regra − oracle_cv", fontsize=8.5)
    axes[0][-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(f"{title}: regras contra o melhor d (vazado = erra)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pn_gap(gap: pd.DataFrame, title: str, path) -> None:
    """Issue #6 (H4). One panel per algorithm: mean ari_train - ari_test along
    p/n, without PCA and with PCA at the d of oracle_cv."""
    algos = list(gap.groupby("algo"))
    fig, axes = plt.subplots(
        1, len(algos), figsize=(5.2 * len(algos), 3.4), sharey=True, squeeze=False
    )
    for ax, (algo, g) in zip(axes[0], algos):
        _style(ax, True)
        _pn_axis(ax, g["p_over_n"])
        g = g.sort_values("p_over_n")
        ax.axhline(0, color=BASELINE, linewidth=0.8)
        ax.plot(g["p_over_n"], g["gap_full"], "o--", color=BASELINE, linewidth=1.3,
                markersize=4, label="sem PCA")
        ax.plot(g["p_over_n"], g["gap_cv"], "o-", color=SERIES, linewidth=2,
                markersize=4, label="PCA no d do oracle_cv")
        ax.set_title(algo, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("p/n", fontsize=8.5)
    axes[0][0].set_ylabel("ari_train − ari_test", fontsize=8.5)
    axes[0][0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"{title}: diferença treino–teste (H4)", fontsize=10.5, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _pn_axis(ax, values) -> None:
    """Log p/n axis with one labelled tick per p/n of the sweep."""
    ticks = sorted(set(float(v) for v in values))
    ax.set_xticks(ticks, [f"{t:.3g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
