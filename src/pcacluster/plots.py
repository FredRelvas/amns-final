"""Figures of the sweep: score by d and difference against no PCA."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

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
