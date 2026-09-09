"""
Visualizations for the hierarchical-counterfactuals results CSV (e.g. data/iris.csv).

Two plots, one figure each:

1. Validity/proximity trade-off: for each method, sort its valid counterfactuals
   by proximity (ascending L2) and plot the running mean L2 against the fraction
   of ALL pairs (valid + invalid) covered so far. A method with validity v% has
   no curve beyond x=v%, so low-validity methods visibly stop early.

2. Per-pair run time: pair_idx on the x-axis, seconds on the y-axis, one series
   per method, markers only (no connecting line) so a method that failed to
   produce a result for a pair simply leaves a gap.

3. Sparsity: mean fraction of features changed (over valid counterfactuals
   only), one bar per method, with the spread across pairs shown as an
   error bar.

Usage:
    python visualizations.py --csv data/iris.csv --out-dir figures
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Validated categorical palette (see dataviz skill, references/palette.md).
# Fixed hue order, assigned by method identity (not by rank/performance).
PALETTE = {
    "method": "#2a78d6",    # blue
    "baseline": "#eb6834",  # orange
    "baseline2": "#e34948", # red
    "dice": "#1baf7a",      # aqua
    "baycon": "#eda100",    # yellow
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e3e2dc"

METHOD_ORDER = ["method", "baseline", "baseline2", "dice", "baycon"]
METHOD_LABEL = {
    "method": "HiCE",
    "baseline": "Baseline",
    "baseline2": "Baseline 2",
    "dice": "DiCE",
    "baycon": "BayCon",
}


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT_PRIMARY)
    ax.yaxis.label.set_color(TEXT_PRIMARY)


def plot_validity_proximity(df, out_path, prox_col="l2"):
    """
    For each method: rank ALL its rows for a pair-set by proximity (only valid
    rows have a defined proximity; invalid rows count toward the denominator
    but never toward the numerator/curve). At coverage fraction x, plot the
    mean proximity of the x-fraction-of-all-pairs most-proximal VALID
    counterfactuals seen so far. The curve for a method stops at its validity
    rate, since beyond that there are no more valid points left to average in.
    """
    methods = [m for m in METHOD_ORDER if m in df["which"].unique()]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    for m in methods:
        sub = df[df["which"] == m]
        n_total = len(sub)
        if n_total == 0:
            continue
        valid_sorted = sub.loc[sub["valid"] == 1, prox_col].dropna().sort_values()
        n_valid = len(valid_sorted)
        if n_valid == 0:
            continue

        # running mean proximity of the k most-proximal valid CFs, at
        # x = k / n_total (fraction of ALL pairs for this method)
        k = np.arange(1, n_valid + 1)
        running_mean = valid_sorted.to_numpy().cumsum() / k
        x_frac = k / n_total

        color = PALETTE.get(m, "#4a3aa7")
        ax.plot(
            x_frac * 100, running_mean,
            color=color, linewidth=2, solid_capstyle="round",
            label=f"{METHOD_LABEL.get(m, m)} (validity {n_valid/n_total:.0%})",
        )
        # mark the endpoint (= method's validity rate) with a dot
        ax.plot(x_frac[-1] * 100, running_mean[-1], "o",
                color=color, markersize=6, markeredgecolor="white",
                markeredgewidth=1, zorder=5)

    ax.set_xlabel("Most-proximal fraction of pairs (%)")
    ax.set_ylabel(f"Mean proximity ({prox_col.upper()}) of the top-x% valid CFs")
    ax.set_title("Validity vs. proximity trade-off", color=TEXT_PRIMARY, fontsize=13, pad=12)
    ax.set_xlim(0, 100)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[out] wrote {out_path}")


def plot_runtime_per_pair(df, out_path):
    """
    pair_idx on x, seconds on y, one marker series per method. A pair for
    which a method has no row (or only an errored row with no timing) simply
    has no marker there, producing a visible gap.
    """
    methods = [m for m in METHOD_ORDER if m in df["which"].unique()]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    for i, m in enumerate(methods):
        sub = df[(df["which"] == m) & df["seconds"].notna()]
        color = PALETTE.get(m, "#4a3aa7")
        ax.scatter(
            sub["pair_idx"], sub["seconds"],
            color=color, s=28, alpha=0.85, edgecolors="white", linewidths=0.5,
            label=METHOD_LABEL.get(m, m), zorder=3 + i,
        )

    ax.set_xlabel("Pair index")
    ax.set_ylabel("Run time (seconds)")
    ax.set_yscale("log")
    ax.set_title("Per-pair run time by method", color=TEXT_PRIMARY, fontsize=13, pad=12)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[out] wrote {out_path}")


def plot_sparsity(df, out_path):
    """
    Bar chart of mean sparsity (fraction of features changed) per method,
    computed over VALID counterfactuals only. Error bars show +/- 1 std
    across pairs. A method with no valid rows (or no sparsity data) is
    skipped, leaving a gap in the bar sequence.
    """
    if "sparsity" not in df.columns:
        print(f"[skip] no 'sparsity' column in data; re-run runHierarchical.py "
              f"to populate it before plotting {out_path}")
        return

    methods = [m for m in METHOD_ORDER if m in df["which"].unique()]

    labels, means, stds, colors = [], [], [], []
    for m in methods:
        sub = df[(df["which"] == m) & (df["valid"] == 1)]["sparsity"].dropna()
        if len(sub) == 0:
            continue
        labels.append(METHOD_LABEL.get(m, m))
        means.append(sub.mean() * 100)
        stds.append(sub.std() * 100)
        colors.append(PALETTE.get(m, "#4a3aa7"))

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, width=0.6,
           capsize=4, error_kw=dict(ecolor=TEXT_SECONDARY, linewidth=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean sparsity (% features changed)")
    ax.set_title("Sparsity by method (valid counterfactuals)",
                 color=TEXT_PRIMARY, fontsize=13, pad=12)
    ax.set_ylim(bottom=0)
    _style_axes(ax)
    ax.xaxis.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[out] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/iris.csv")
    ap.add_argument("--out-dir", type=str, default="data/images")
    ap.add_argument("--prox-col", type=str, default="l2",
                     choices=["l2", "l2_sq", "l1", "linf"])
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    os.makedirs(a.out_dir, exist_ok=True)
    dataset = df["dataset"].iloc[0] if "dataset" in df.columns and len(df) else \
        os.path.splitext(os.path.basename(a.csv))[0]

    plot_validity_proximity(
        df, os.path.join(a.out_dir, f"{dataset}_validity_proximity.png"), prox_col=a.prox_col)
    plot_runtime_per_pair(
        df, os.path.join(a.out_dir, f"{dataset}_runtime_per_pair.png"))
    plot_sparsity(
        df, os.path.join(a.out_dir, f"{dataset}_sparsity.png"))


if __name__ == "__main__":
    main()
