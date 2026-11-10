"""
Diversity figures for complete linkage (Euclidean and Hamming).

Same three panels and the same reading conventions as the Ward diversity set:
pairs sorted, ACTUAL value read off at each 5% step, and a SHARED denominator
of every pair in the experiment, so a method that returns nothing valid for a
pair never earns that coverage and its curve stops at its own validity.

  proximity     best (smallest) valid counterfactual per pair, from the RAW
                per-CF file -- the summary stores only the mean. Normalised by
                that pair's `scale`, which the driver already records: the
                centroid distance under Euclidean, the feature count under
                Hamming (so it reads as the fraction of features relabelled).
  tree coverage fraction of B's sub-branches represented, sorted best-first.
                Defined only where B has >= 3 leaves.
  runtime       wall-clock per (pair, method), log y.

Rows are filtered on the METRIC COLUMN, not the filename: the Euclidean run's
dataset list carries the categorical hayes-roth through, so 100 hamming rows
sit in the Euclidean file and would otherwise pollute the normalisation.
"""
import csv, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import NullLocator
import matplotlib.pyplot as plt
from modules.constants import (PALETTE, METHOD_LABEL, TEXT_PRIMARY,
                               TEXT_SECONDARY, GRID_COLOR)

CFG = {
    "euclidean": dict(sum="data/completeDiversityEuclidean.csv",
                      raw="data/completeDiversityEuclidean_raw.csv",
                      metric="euclidean", out="completeDiv_euclidean",
                      ms=["method", "baseline", "dice", "baycon_random", "baycon_proximal"],
                      extra_sum="data/completeDivBaselineEuclidean.csv",
                      extra_raw="data/completeDivBaselineEuclidean_raw.csv",
                      ylab=r"Proximity   $||t||\,/\,||\mu_B - \mu_A||$"),
    "hamming":   dict(sum="data/completeDiversityHamming.csv",
                      raw="data/completeDiversityHamming_raw.csv",
                      metric="hamming", out="completeDiv_hamming",
                      ms=["method", "baseline"],
                      extra_sum="data/completeDivBaselineHamming.csv",
                      extra_raw="data/completeDivBaselineHamming_raw.csv",
                      # The paper normalises hamming proximity by the FEATURE
                      # count, so the figure reads n_feat (features touched,
                      # <= d) rather than l2 (cells changed, <= |A|*d) and
                      # divides by d. Cells cannot be rescaled into features:
                      # cells/n_feat is the mean number of A's points changed
                      # per touched feature, measured between 1 and 35 here.
                      prox_col="n_feat", prox_denom="d",
                      # HiCE's hamming raw file was written before the driver
                      # recorded n_feat, so under this metric it contributes
                      # NOTHING to the proximity panel: every one of its rows
                      # has an empty n_feat and is skipped. `nfeat_raw` is a
                      # re-solve of exactly the pairs that returned something
                      # valid (runCompleteDiversity --only-pairs), with n_feat
                      # recorded, and is what actually draws HiCE's curve.
                      #
                      # It is a re-solve, not a second sample: the driver
                      # seeds each call with call_seed(dataset, linkage, seed,
                      # idx, which), so the same pair_idx reproduces the same
                      # counterfactuals. Verified by joining on (dataset,
                      # pair_idx, cf_idx) and comparing `l2`, which both files
                      # carry.
                      #
                      # Pairs the original run could not answer are absent
                      # from it by construction, so they are absent here --
                      # which is correct, since the shared denominator N is
                      # still every pair in the experiment and HiCE's curve
                      # must stop at its own validity.
                      nfeat_raw="data/completeDivHammingNfeat_raw.csv",
                      ylab=r"Proximity   (# features relabelled $/\ d$)"),
}

def f(x):
    try:
        v = float(x); return None if math.isnan(v) else v
    except Exception: return None

def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR); ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=16)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8); ax.set_axisbelow(True)

def curve(ax, grid, ys, key, label):
    ax.plot(grid, ys, lw=2, marker="o", ms=5, color=PALETTE[key], label=label,
            mec="white", mew=1.2, solid_capstyle="round")

for name, c in CFG.items():
    METRIC = c["metric"]
    # Both diversity legs are complete linkage; the metric is what varies.
    TITLE = "%s metric, complete linkage" % METRIC.capitalize()
    scale, tc, nsub, nvalid, secs, pairs, nA = {}, {}, {}, {}, {}, set(), {}
    for r in csv.DictReader(open(c["sum"])):
        if r.get("metric") != METRIC:
            continue
        k = (r["dataset"], r["pair_idx"])
        pairs.add(k)
        s = f(r.get("scale", ""))
        if s and s > 0: scale[k] = s
        na = f(r.get("n_A", ""))
        if na and na > 0: nA[k] = na
        ns = f(r.get("n_subbranches", ""))
        if ns is not None: nsub.setdefault(k, ns)
        v = f(r.get("tree_coverage", ""))
        if v is not None: tc[(k[0], k[1], r["which"])] = v
        nv = f(r.get("n_valid", ""))
        if nv is not None: nvalid[(k[0], k[1], r["which"])] = nv
        sc = f(r.get("seconds", ""))
        if sc is not None: secs[(k[0], k[1], r["which"])] = sc

    # The per-sub-branch baseline was run by a separate driver, so it lands in
    # its own pair of files with the SAME schema. Fold it in here rather than
    # re-running anything. Pairs are restricted to those the main summary
    # already defines, so the shared denominator N does not move: a baseline
    # that covers fewer pairs simply ends its curve at its own validity.
    _extra_sum = c["extra_sum"] if os.path.exists(c["extra_sum"]) else None
    _extra_raw = c["extra_raw"] if os.path.exists(c["extra_raw"]) else None
    if _extra_sum is None:
        print("  [note] %s not on disk yet -- baseline omitted from %s"
              % (c["extra_sum"], name))
    for r in (csv.DictReader(open(_extra_sum)) if _extra_sum else []):
        if r.get("metric") != METRIC:
            continue
        k = (r["dataset"], r["pair_idx"])
        if k not in pairs:
            continue
        s_ = f(r.get("scale", ""))
        if s_ and s_ > 0: scale.setdefault(k, s_)
        ns = f(r.get("n_subbranches", ""))
        if ns is not None: nsub.setdefault(k, ns)
        v = f(r.get("tree_coverage", ""))
        if v is not None: tc[(k[0], k[1], r["which"])] = v
        nv = f(r.get("n_valid", ""))
        if nv is not None: nvalid[(k[0], k[1], r["which"])] = nv
        sc = f(r.get("seconds", ""))
        if sc is not None: secs[(k[0], k[1], r["which"])] = sc

    PCOL = c.get("prox_col", "l2")

    best = {}
    for r in csv.DictReader(open(c["raw"])):
        if r.get("metric") != METRIC: continue
        if r.get("valid") not in ("1", "True", "true"): continue
        l2 = f(r.get(PCOL, ""))
        if l2 is None: continue
        k = (r["dataset"], r["pair_idx"], r["which"])
        if k not in best or l2 < best[k]: best[k] = l2

    # The per-sub-branch baseline, and (hamming only) the n_feat re-solve that
    # supplies HiCE's proximity. Both carry the full raw schema and are keyed
    # the same way, so they fold in under the same rule as the main file:
    # smallest valid value per (dataset, pair, method). Restricted to `pairs`
    # so neither file can widen the shared denominator.
    _nfeat_raw = c.get("nfeat_raw")
    if _nfeat_raw and not os.path.exists(_nfeat_raw):
        print("  [note] %s not on disk yet -- HiCE omitted from %s proximity"
              % (_nfeat_raw, name))
        _nfeat_raw = None
    for src in (_extra_raw, _nfeat_raw):
        for r in (csv.DictReader(open(src)) if src else []):
            if r.get("metric") != METRIC: continue
            if r.get("valid") not in ("1", "True", "true"): continue
            l2 = f(r.get(PCOL, ""))
            if l2 is None: continue
            if (r["dataset"], r["pair_idx"]) not in pairs: continue
            k = (r["dataset"], r["pair_idx"], r["which"])
            if k not in best or l2 < best[k]: best[k] = l2

    N = len(pairs)
    measurable = sorted({p for p, v in nsub.items() if v and v > 0})
    M = len(measurable)
    print("=== %s: %d pairs, %d datasets | coverage-measurable: %d" % (
        name, N, len({p[0] for p in pairs}), M))

    def grid_of(vals, denom, reverse=False, pct_scale=1.0):
        vals = sorted(vals, reverse=reverse)
        g, ys = [], []
        for pct in range(5, 101, 5):
            i = int(round(pct / 100.0 * denom)) - 1
            if 0 <= i < len(vals):
                g.append(pct); ys.append(vals[i] * pct_scale)
        return g, ys

    # ---- proximity ----
    # Under Hamming, `l2` is the relabelling cost SUMMED over the |A| points the
    # relabelling moves, so it ranges up to |A|*d, not d. Dividing by `scale`
    # (= d) alone yields "mean features changed per point", which exceeds 1 and
    # is not comparable across datasets; the fraction of relabelled CELLS needs
    # |A|*d. Verified on this data: l2/(n_A*d) lands in [0.009, 1.000] with 0
    # violations over 465 valid CFs, while l2 > d in 89 of them.
    def denom(k):
        # `scale` from the drivers is already the full denominator: ||mu_B-mu_A||
        # under euclidean, |A|*d under hamming (changed cells summed over A's
        # points). Do not re-multiply by n_A here.
        if c.get("prox_denom") == "d":
            # Feature-normalised: the numerator is n_feat (<= d), so the
            # denominator must be d, recovered as scale/|A| since the driver
            # stores the cell-scale |A|*d.
            return scale[k] / nA[k]
        return scale[k]

    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=300)
    for m in c["ms"]:
        vals = [best[(d, p, m)] / denom((d, p)) for (d, p) in pairs
                if (d, p, m) in best and (d, p) in scale
                and (c.get("prox_denom") != "d" or nA.get((d, p), 0) > 0)]
        if not vals: continue
        g, ys = grid_of(vals, N)
        curve(ax, g, ys, m, METHOD_LABEL[m])
        print("  prox %-16s validity=%5.1f%% last=%d%% -> %.3f" % (
            METHOD_LABEL[m], 100*len(vals)/N, g[-1] if g else 0, ys[-1] if ys else float("nan")))
    ax.set_xlabel("Validity   (% of valid counterfactuals)", fontsize=18); ax.set_ylabel(c["ylab"], fontsize=18)
    style(ax); ax.set_xlim(0, 100); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=15, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), ncol=3)
    fig.tight_layout(); fig.savefig("data/images/%s_proximity.png" % c["out"], facecolor="white")
    plt.close(fig)

    # ---- tree coverage ----
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=300)
    for m in c["ms"]:
        vals = [tc[(d, p, m)] for (d, p) in measurable
                if (d, p, m) in tc and nvalid.get((d, p, m), 0) > 0]
        if not vals: continue
        g, ys = grid_of(vals, M, reverse=True, pct_scale=100.0)
        curve(ax, g, ys, m, METHOD_LABEL[m])
        print("  tcov %-16s validity=%5.1f%% last=%d%% -> %.1f%%" % (
            METHOD_LABEL[m], 100*len(vals)/M, g[-1] if g else 0, ys[-1] if ys else float("nan")))
    ax.set_xlabel("Validity   (% of valid counterfactuals)", fontsize=18); ax.set_ylabel("Tree coverage", fontsize=18)
    style(ax); ax.set_xlim(0, 100); ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=15, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), ncol=3)
    fig.tight_layout(); fig.savefig("data/images/%s_treecoverage.png" % c["out"], facecolor="white")
    plt.close(fig)

    # ---- runtime ----
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=300)
    for m in c["ms"]:
        vals = [v for (d, p, w), v in secs.items() if w == m]
        if not vals: continue
        g, ys = grid_of([max(v, 1e-4) for v in vals], N)
        curve(ax, g, ys, m, METHOD_LABEL[m])
        sv = sorted(vals)
        print("  time %-16s median=%8.2fs max=%9.1fs total=%6.1f h" % (
            METHOD_LABEL[m], sv[len(sv)//2], max(sv), sum(sv)/3600))
    ax.set_yscale("log"); ax.yaxis.set_minor_locator(NullLocator())
    # Budget reference line, matching tmp_plot_rt_exact.py so all three runtime
    # panels carry the same line at the same height.
    ax.axhline(1200.0, ls=":", lw=1.2, color=TEXT_SECONDARY, zorder=1)
    ax.set_xlabel("% of pairs", fontsize=18); ax.set_ylabel("Runtime per pair (s)", fontsize=18)
    style(ax)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, which="major")
    ax.grid(False, which="minor"); ax.set_xlim(0, 100)
    ax.legend(frameon=False, fontsize=15, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), ncol=3)
    fig.tight_layout(); fig.savefig("data/images/%s_runtime.png" % c["out"], facecolor="white")
    plt.close(fig)
    print("  wrote data/images/%s_{proximity,treecoverage,runtime}.png\n" % c["out"])
