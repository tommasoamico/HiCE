"""
Tree-coverage-vs-validity curve for the Ward diversity experiment.

Companion to the proximity curve, same construction with the sort reversed
because coverage is better when HIGHER:

  * one value per (pair, method): the fraction of B's sub-branches represented
  * pairs sorted best-first, read off at each 5% step of a SHARED denominator
  * a method that returns no valid counterfactual for a pair never earns that
    coverage, so its curve STOPS at its own validity rate

The denominator is the pairs where tree coverage is DEFINED, i.e. where B has
at least 3 leaves. A 2-leaf B has only single-leaf children, and a sub-branch
needs >=2 leaves, so n_subbranches = 0 and there is nothing to cover -- 726 of
1800 pairs (40%). Those pairs are perfectly valid counterfactual problems and
remain in the proximity figure; they simply cannot exercise this metric.
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modules.constants import (PALETTE, METHOD_LABEL, TEXT_PRIMARY,
                               TEXT_SECONDARY, GRID_COLOR)

SUM  = "data/wardExactL2.csv"
BASE = "data/wardBaseline.csv"
OUT  = "data/images/wardExact_treecoverage_coverage.png"
MS   = ["method", "baseline", "dice", "baycon_random", "baycon_proximal"]

def f(x):
    try:
        v = float(x); return None if math.isnan(v) else v
    except Exception: return None

tc, nsub, nvalid = {}, {}, {}
def ingest(path, valid_key=None):
    for r in csv.DictReader(open(path)):
        k = (r["dataset"], r["pair_idx"], r["which"])
        v = f(r.get("tree_coverage", ""))
        if v is not None:
            tc[k] = v
        ns = f(r.get("n_subbranches", ""))
        if ns is not None:
            nsub.setdefault((r["dataset"], r["pair_idx"]), ns)
        nv = f(r.get("n_valid", ""))
        if nv is not None:
            nvalid[k] = nv
        elif valid_key:                      # baseline file: one CF, valid flag
            nvalid[k] = 1.0 if r.get("valid") in ("True", "1", "true") else 0.0
ingest(SUM)
ingest(BASE, valid_key="valid")

measurable = sorted({p for p, v in nsub.items() if v and v > 0})
N = len(measurable)
print("measurable pairs (B has >=3 leaves): %d of %d" % (N, len(nsub)))

fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=300)
for m in MS:
    vals = []
    for (d, p) in measurable:
        k = (d, p, m)
        # only pairs the method actually solved -- a pair with no valid
        # counterfactual is a failure, not a 0% coverage score
        if nvalid.get(k, 0) <= 0:
            continue
        v = tc.get(k)
        if v is not None:
            vals.append(v)
    if not vals:
        continue
    vals.sort(reverse=True)                  # best first: higher is better
    grid, ys = [], []
    for pct in range(5, 101, 5):
        idx = int(round(pct / 100.0 * N)) - 1
        if idx < 0 or idx >= len(vals):
            break
        grid.append(pct); ys.append(100 * vals[idx])
    ax.plot(grid, ys, lw=2, marker="o", ms=5, color=PALETTE[m],
            label=METHOD_LABEL[m], mec="white", mew=1.2, solid_capstyle="round")
    print("%-16s validity=%5.1f%%  last grid=%d%% -> %.1f%%  (n=%d)" % (
        m, 100*len(vals)/N, grid[-1] if grid else 0, ys[-1] if ys else float('nan'), len(vals)))

ax.set_xlabel("Validity   (% of valid counterfactuals)", fontsize=18)
ax.set_ylabel("Tree coverage", fontsize=18)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID_COLOR); ax.spines["bottom"].set_color(GRID_COLOR)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=16)
ax.grid(True, color=GRID_COLOR, linewidth=0.8); ax.set_axisbelow(True)
ax.set_xlim(0, 100); ax.set_ylim(0, 105)
# Above the axes, not inside -- the shared legend style for every figure in
# the paper. At 15pt a legend no longer reliably fits a gap between the
# curves (here HiCE descends across the lower-right corner, the BayCon pair
# fills the lower left, and the baseline sweeps the middle), and placing it
# per-figure made sibling panels inconsistent. Dropping the titles freed the
# strip above the axes, so every figure puts it there, ncol=3.
ax.legend(frameon=False, fontsize=15, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), ncol=3)
os.makedirs("data/images", exist_ok=True)
fig.tight_layout(); fig.savefig(OUT, facecolor="white")
print("wrote", OUT)
