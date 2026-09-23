"""
Visualizations for the hierarchical-counterfactuals results CSVs
(e.g. data/iris.csv, produced by runHierarchical.py or runHiceSparsity.py).

Accepts one or more --csv files. Every distinct "method" found across all
of them is plotted together, so you can e.g. combine a runHierarchical.py
results file (method/baseline/baseline2/dice/baycon) with a
runHiceSparsity.py file (HiCE under different sparsity_mode settings) into
one comparison.

Two schemas are understood and normalized into a single 'which' column:
  - runHierarchical.py output: already has a 'which' column
    (method/baseline/baseline2/dice/baycon).
  - runHiceSparsity.py output: has no 'which' column, instead a
    'sparsity_mode' column (l2/l1/adaptive/baycon_mask) -- each mode is
    treated as its own method, labeled "HiCE (<mode>)".

Three plots, one figure each, produced PER DATASET found in the combined
data (a dataset present in multiple input files has its rows merged):

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

Additionally, the following summary plots are produced ONCE (not per
dataset):

- summary_validity_proximity.png: the same validity/proximity trade-off
  curve as above, but pooling every pair from every dataset into one
  curve per method, instead of one figure per dataset.

- summary_{validity,proximity,sparsity,runtime}.png: four bar charts, one
  per metric, each a bar per method: mean validity rate, mean proximity,
  mean sparsity, and mean run time, all averaged ACROSS DATASETS -- i.e.
  each dataset present contributes one number per method, and the error
  bar shows the spread of that number across datasets (zero-width with a
  single dataset).

- cd_{validity,proximity,sparsity,runtime}.png: critical-difference plots
  produced by the `autorank` package (Nemenyi post-hoc test, forced
  nonparametric mode), ranking methods per dataset and testing whether
  their average ranks differ significantly (alpha=0.05); skipped if there
  are fewer than 2 datasets with complete data across every method.

Usage:
    python visualizations.py --csv data/iris.csv --out-dir figures
    python visualizations.py --csv data/iris.csv data/iris_sparsity.csv --out-dir figures
"""

import argparse
import os
from modules.constants import PALETTE, TEXT_PRIMARY, TEXT_SECONDARY, GRID_COLOR, METHOD_ORDER, METHOD_LABEL, EXTRA_COLORS
import autorank
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def _load_and_normalize(csv_path):
    """
    Read one results CSV and make sure it has a 'which' column identifying
    the method, regardless of which runner produced it.
    """
    df = pd.read_csv(csv_path)
    if "which" not in df.columns:
        if "sparsity_mode" in df.columns:
            df = df.copy()
            df["which"] = "hice_" + df["sparsity_mode"].astype(str)
        else:
            raise ValueError(
                f"'{csv_path}' has neither a 'which' nor a 'sparsity_mode' "
                f"column -- don't know how to identify its method(s).")
    df["_source_csv"] = csv_path
    return df


def _build_style(methods):
    """
    Return (order, label_map, color_map) covering exactly `methods`,
    known methods first (in METHOD_ORDER), then any unrecognized ones
    (e.g. hice_<sparsity_mode>) sorted alphabetically, assigned colors
    from EXTRA_COLORS.
    """
    known = [m for m in METHOD_ORDER if m in methods]
    unknown = sorted(m for m in methods if m not in METHOD_ORDER)

    order = known + unknown
    labels = dict(METHOD_LABEL)
    colors = dict(PALETTE)
    for i, m in enumerate(unknown):
        if m not in labels:
            if m.startswith("hice_"):
                labels[m] = f"HiCE ({m[len('hice_'):]})"
            else:
                labels[m] = m
        if m not in colors:
            colors[m] = EXTRA_COLORS[i % len(EXTRA_COLORS)]
    return order, labels, colors


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


def plot_validity_proximity(df, out_path, order, labels, colors, prox_col="l2",
                             title="Validity vs. proximity trade-off"):
    """
    For each method: rank ALL its rows for a pair-set by proximity (only valid
    rows have a defined proximity; invalid rows count toward the denominator
    but never toward the numerator/curve). At coverage fraction x, plot the
    mean proximity of the x-fraction-of-all-pairs most-proximal VALID
    counterfactuals seen so far. The curve for a method stops at its validity
    rate, since beyond that there are no more valid points left to average in.

    Every method is measured against the SAME denominator: the number of
    distinct pairs present anywhere in `df` (not each method's own row
    count) -- distinct (dataset, pair_idx) combinations if `df` spans
    multiple datasets (e.g. a cross-dataset summary plot), else just
    distinct pair_idx. This matters because some methods only have rows
    for a strict subset of pairs -- e.g. runHiceSparsity.py's baycon_mask
    mode writes no row at all for a pair when BayCon found no
    counterfactual to derive a mask from, rather than an explicit invalid
    row. Using that method's own (smaller, easier) row count as its
    denominator would silently inflate its apparent validity rate and
    misalign its curve against methods evaluated on every pair; a pair
    missing from a method's rows is instead counted here as coverage the
    method failed to reach, same as an explicit invalid row would be.
    """
    methods = [m for m in order if m in df["which"].unique()]
    if "pair_idx" not in df.columns:
        n_universe = None
    elif "dataset" in df.columns and df["dataset"].nunique() > 1:
        n_universe = len(df[["dataset", "pair_idx"]].drop_duplicates())
    else:
        n_universe = df["pair_idx"].nunique()

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    for m in methods:
        sub = df[df["which"] == m]
        n_total = n_universe if n_universe else len(sub)
        if n_total == 0:
            continue
        valid_sorted = sub.loc[sub["valid"] == 1, prox_col].dropna().sort_values()
        n_valid = len(valid_sorted)
        if n_valid == 0:
            continue

        # running mean proximity of the k most-proximal valid CFs, at
        # x = k / n_total (fraction of ALL pairs in the dataset, shared
        # across methods)
        k = np.arange(1, n_valid + 1)
        running_mean = valid_sorted.to_numpy().cumsum() / k
        x_frac = k / n_total

        color = colors.get(m, "#4a3aa7")
        ax.plot(
            x_frac * 100, running_mean,
            color=color, linewidth=2, solid_capstyle="round",
            label=f"{labels.get(m, m)} (validity {n_valid/n_total:.0%})",
        )
        # mark the endpoint (= method's validity rate) with a dot
        ax.plot(x_frac[-1] * 100, running_mean[-1], "o",
                color=color, markersize=6, markeredgecolor="white",
                markeredgewidth=1, zorder=5)

    ax.set_xlabel("Most-proximal fraction of pairs (%)")
    ax.set_ylabel(f"Mean proximity ({prox_col.upper()}) of the top-x% valid CFs")
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, pad=12)
    ax.set_xlim(0, 100)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[out] wrote {out_path}")


def plot_runtime_per_pair(df, out_path, order, labels, colors):
    """
    pair_idx on x, seconds on y, one marker series per method. A pair for
    which a method has no row (or only an errored row with no timing) simply
    has no marker there, producing a visible gap.
    """
    methods = [m for m in order if m in df["which"].unique()]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    for i, m in enumerate(methods):
        sub = df[(df["which"] == m) & df["seconds"].notna()]
        color = colors.get(m, "#4a3aa7")
        ax.scatter(
            sub["pair_idx"], sub["seconds"],
            color=color, s=28, alpha=0.85, edgecolors="white", linewidths=0.5,
            label=labels.get(m, m), zorder=3 + i,
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


def plot_sparsity(df, out_path, order, labels, colors):
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

    methods = [m for m in order if m in df["which"].unique()]

    bar_labels, means, stds, bar_colors = [], [], [], []
    for m in methods:
        sub = df[(df["which"] == m) & (df["valid"] == 1)]["sparsity"].dropna()
        if len(sub) == 0:
            continue
        bar_labels.append(labels.get(m, m))
        means.append(sub.mean() * 100)
        stds.append(sub.std() * 100)
        bar_colors.append(colors.get(m, "#4a3aa7"))

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    x = np.arange(len(bar_labels))
    ax.bar(x, means, yerr=stds, color=bar_colors, width=0.6,
           capsize=4, error_kw=dict(ecolor=TEXT_SECONDARY, linewidth=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
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


def _per_dataset_method_stats(df, prox_col="l2"):
    """
    Collapse the combined multi-dataset dataframe to one row per
    (dataset, method), with the same summary statistics the per-dataset
    plots use. This is the input to the cross-dataset summary plots below:
    each dataset contributes one data point per method, so a method's
    spread across datasets becomes an error bar.
    """
    has_sparsity = "sparsity" in df.columns
    rows = []
    for (dataset, which), g in df.groupby(["dataset", "which"]):
        n_total = len(g)
        valid = g[g["valid"] == 1]
        n_valid = len(valid)
        rows.append(dict(
            dataset=dataset, which=which,
            validity_rate=(n_valid / n_total) if n_total else np.nan,
            mean_prox=valid[prox_col].mean() if n_valid else np.nan,
            mean_sparsity=(valid["sparsity"].mean()
                           if has_sparsity and n_valid else np.nan),
            mean_seconds=g["seconds"].mean() if "seconds" in g.columns else np.nan,
        ))
    return pd.DataFrame(rows)


def plot_cross_dataset_summary(stats, metric, out_path, order, labels, colors,
                                ylabel, title, pct=False):
    """
    Bar chart with one bar per method, averaged ACROSS DATASETS (each
    dataset that has a defined value for `metric` and this method
    contributes one point). Error bars show +/- 1 std across datasets, so
    they collapse to zero with a single dataset. A method missing the
    metric everywhere is skipped, leaving a gap in the bar sequence.
    """
    methods = [m for m in order if m in stats["which"].unique()]

    bar_labels, means, stds, bar_colors, ns = [], [], [], [], []
    for m in methods:
        vals = stats.loc[stats["which"] == m, metric].dropna()
        if len(vals) == 0:
            continue
        v = vals * 100 if pct else vals
        bar_labels.append(labels.get(m, m))
        means.append(v.mean())
        stds.append(v.std() if len(v) > 1 else 0)
        bar_colors.append(colors.get(m, "#4a3aa7"))
        ns.append(len(v))

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    x = np.arange(len(bar_labels))
    ax.bar(x, means, yerr=stds, color=bar_colors, width=0.6,
           capsize=4, error_kw=dict(ecolor=TEXT_SECONDARY, linewidth=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(bar_labels, ns)])
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, pad=12)
    ax.set_ylim(bottom=0)
    _style_axes(ax)
    ax.xaxis.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[out] wrote {out_path}")


def plot_critical_difference(stats, metric, out_path, order, labels, colors,
                              ascending, title, alpha=0.05):
    """
    Critical-difference plot via the `autorank` package (forced
    nonparametric mode -> Friedman test + Nemenyi post-hoc), ranking
    methods per dataset by `metric` ('ascending'==lower raw value is
    better, i.e. proximity/sparsity/runtime; 'descending'==higher raw
    value is better, i.e. validity) and testing whether their average
    ranks differ significantly at level `alpha`.

    `colors` is accepted for signature symmetry with the other summary
    plots but unused: autorank owns the figure's styling internally.

    Requires COMPLETE data: a dataset is only used if every method in
    `order` that appears anywhere in `stats` has a non-NaN value for it
    there. autorank hard-requires at least 5 such complete datasets (and
    at least 2 methods); otherwise the plot is skipped.
    """
    methods = [m for m in order if m in stats["which"].unique()]
    pivot = stats.pivot(index="dataset", columns="which", values=metric)
    pivot = pivot.reindex(columns=methods).dropna(axis=0, how="any")
    n, k = len(pivot), len(methods)

    if k < 2 or n < 5:
        print(f"[skip] not enough complete data for {out_path} "
              f"(methods={k}, complete datasets={n}; autorank needs >=2 "
              f"methods and >=5 datasets with every method present)")
        return

    mat = pivot.rename(columns=labels)
    order_dir = "ascending" if ascending else "descending"
    result = autorank.autorank(mat, alpha=alpha, order=order_dir,
                                force_mode="nonparametric", verbose=False)

    plt.close("all")
    autorank.plot_stats(result, allow_insignificant=True)
    fig = plt.gcf()
    fig.set_dpi(150)
    fig.suptitle(f"{title}\n(N={n} datasets, k={k} methods)",
                 fontsize=13, y=1.22)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[out] wrote {out_path} (N={n}, k={k})")
    print(result.rankdf.sort_values("meanrank"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, nargs="+", default=["data/iris.csv"],
                     help="one or more results CSVs (runHierarchical.py and/or "
                          "runHiceSparsity.py output); all methods found across "
                          "them are combined into the same plots, per dataset")
    ap.add_argument("--out-dir", type=str, default="data/images")
    ap.add_argument("--prox-col", type=str, default="l2",
                     choices=["l2", "l2_sq", "l1", "linf"])
    a = ap.parse_args()

    dfs = [_load_and_normalize(p) for p in a.csv]
    df = pd.concat(dfs, ignore_index=True)
    os.makedirs(a.out_dir, exist_ok=True)

    datasets = df["dataset"].dropna().unique() if "dataset" in df.columns else [None]

    for dataset in datasets:
        sub = df if dataset is None else df[df["dataset"] == dataset]
        if len(sub) == 0:
            continue
        order, labels, colors = _build_style(sub["which"].unique())
        label = dataset if dataset is not None else "combined"

        plot_validity_proximity(
            sub, os.path.join(a.out_dir, f"{label}_validity_proximity.png"),
            order, labels, colors, prox_col=a.prox_col)
        plot_runtime_per_pair(
            sub, os.path.join(a.out_dir, f"{label}_runtime_per_pair.png"),
            order, labels, colors)
        plot_sparsity(
            sub, os.path.join(a.out_dir, f"{label}_sparsity.png"),
            order, labels, colors)

    # cross-dataset summary: only meaningful with results from >1 dataset,
    # but still produced (with zero-width error bars) for a single one.
    if len(datasets) >= 1 and "dataset" in df.columns:
        order, labels, colors = _build_style(df["which"].unique())
        stats = _per_dataset_method_stats(df, prox_col=a.prox_col)

        # validity/proximity trade-off pooling every pair from every
        # dataset into one curve per method (vs. the per-dataset plots
        # above, which are one dataset each)
        plot_validity_proximity(
            df, os.path.join(a.out_dir, "summary_validity_proximity.png"),
            order, labels, colors, prox_col=a.prox_col,
            title=f"Validity vs. proximity trade-off, all pairs pooled "
                  f"across {len(datasets)} dataset(s)")

        plot_cross_dataset_summary(
            stats, "validity_rate",
            os.path.join(a.out_dir, "summary_validity.png"),
            order, labels, colors, pct=True,
            ylabel="Mean validity rate (%)",
            title=f"Validity by method, averaged across {len(datasets)} dataset(s)")
        plot_cross_dataset_summary(
            stats, "mean_prox",
            os.path.join(a.out_dir, "summary_proximity.png"),
            order, labels, colors,
            ylabel=f"Mean proximity ({a.prox_col.upper()}, valid CFs)",
            title=f"Proximity by method, averaged across {len(datasets)} dataset(s)")
        plot_cross_dataset_summary(
            stats, "mean_sparsity",
            os.path.join(a.out_dir, "summary_sparsity.png"),
            order, labels, colors, pct=True,
            ylabel="Mean sparsity (% features changed, valid CFs)",
            title=f"Sparsity by method, averaged across {len(datasets)} dataset(s)")
        plot_cross_dataset_summary(
            stats, "mean_seconds",
            os.path.join(a.out_dir, "summary_runtime.png"),
            order, labels, colors,
            ylabel="Mean run time (seconds)",
            title=f"Run time by method, averaged across {len(datasets)} dataset(s)")

        # Nemenyi critical-difference diagrams (need complete, >=3-dataset
        # data across methods -- plot_critical_difference skips otherwise)
        plot_critical_difference(
            stats, "validity_rate", os.path.join(a.out_dir, "cd_validity.png"),
            order, labels, colors, ascending=False,
            title="Critical difference — validity rate")
        plot_critical_difference(
            stats, "mean_prox", os.path.join(a.out_dir, "cd_proximity.png"),
            order, labels, colors, ascending=True,
            title=f"Critical difference — proximity ({a.prox_col.upper()})")
        plot_critical_difference(
            stats, "mean_sparsity", os.path.join(a.out_dir, "cd_sparsity.png"),
            order, labels, colors, ascending=True,
            title="Critical difference — sparsity")
        plot_critical_difference(
            stats, "mean_seconds", os.path.join(a.out_dir, "cd_runtime.png"),
            order, labels, colors, ascending=True,
            title="Critical difference — run time")


if __name__ == "__main__":
    main()
