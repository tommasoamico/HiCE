"""
Pooled (across-dataset) visualizations for a diversity-summary CSV like
data/hiceSparsityCoverage.csv (the DIVERSITY_SUMMARY_FIELDNAMES schema
produced by runHierarchical.py, one row per (dataset, pair_idx, which) with
k_requested=10 counterfactuals requested per pair).

Three figures, each pooling every (dataset, pair) row into one curve/bar
per method ('which'):

1. Proximity coverage: sort each method's pairs by avg_proximity_norm
   ascending and plot the running mean as a function of the fraction of
   ALL pairs covered so far -- i.e. "if you only trust the x% closest
   counterfactuals a method produced, how proximal are they on average."
   A method stops short of x=100% wherever it failed to return a usable
   (>=1 valid CF) result for some pairs.

2. Diversity coverage: same construction, ranked by avg_pairwise_distance_norm
   instead -- "the x% most mutually-diverse pairs' CF sets."

3. Tree coverage: a separate bar chart (mean tree_coverage, i.e. the
   fraction of B's sub-branches represented by >=1 returned CF, +/- 1 std
   across pairs). Tree coverage is a per-pair fraction, not something
   naturally ranked by "closeness", so it's shown on its own rather than
   forced onto the same most-proximal/most-diverse x-axis as 1-2.

Usage
-----
    python plotHiceSparsityCoverage.py
    python plotHiceSparsityCoverage.py --csv data/hiceSparsityCoverage.csv --out-dir data/images
"""

import argparse
import os

import pandas as pd

from visualizations import _build_style
from runHierarchical import plot_coverage_curve, plot_tree_coverage


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=str, default="data/hiceSparsityCoverage.csv")
    ap.add_argument("--out-dir", type=str, default="data/images")
    ap.add_argument("--prefix", type=str, default="hiceSparsityCoverage",
                     help="filename prefix for the three output PNGs")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    os.makedirs(a.out_dir, exist_ok=True)

    n_datasets = df["dataset"].nunique() if "dataset" in df.columns else 1
    order, labels, colors = _build_style(df["which"].unique())

    plot_coverage_curve(
        df, os.path.join(a.out_dir, f"{a.prefix}_proximity.png"),
        order, labels, colors, metric_col="avg_proximity_norm",
        x_label="Most-proximal fraction of pairs (%)",
        y_label="Mean normalized proximity of the top-x% pairs",
        title=f"Proximity coverage, pooled across {n_datasets} dataset(s)")

    plot_coverage_curve(
        df, os.path.join(a.out_dir, f"{a.prefix}_diversity.png"),
        order, labels, colors, metric_col="avg_pairwise_distance_norm",
        x_label="Most-diverse fraction of pairs (%)",
        y_label="Mean normalized pairwise diversity of the top-x% pairs",
        title=f"Diversity coverage, pooled across {n_datasets} dataset(s)")

    plot_tree_coverage(
        df, os.path.join(a.out_dir, f"{a.prefix}_tree_coverage.png"),
        order, labels, colors,
        title=f"Tree coverage by method, pooled across {n_datasets} dataset(s)")


if __name__ == "__main__":
    main()
