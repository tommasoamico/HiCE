"""
Between-branch diversity figure (Ward, Euclidean).

One panel, same reading convention as the other proximity figures: pairs are
sorted by proximity, the ACTUAL value is read off at each 5% step, and the
denominator is EVERY pair in the experiment, so a curve ends at the method's
validity.

Two deliberate differences from tmp_plot_div.py, both forced by what a
between-branch alternative is:

  * the per-pair value is the MEAN over that pair's valid alternatives
    (`avg_proximity_norm`), not the best one. Each alternative answers a
    DIFFERENT question ("what would make A resemble B' instead?"), so the
    cheapest of ten is not a summary of the set -- the average cost of reaching
    an alternative target is.
  * proximity is normalised per TARGET, by ||mu_{B'} - mu_A|| for that
    alternative's own centroid, since each alternative has its own target. The
    driver already does this per solve before averaging.

Only HiCE and the baseline are drawn. `flatcut` is excluded on purpose: its
coverage is scored against its OWN candidate set (runBetweenBranch.py:187), so
it is not comparable to the other two on any coverage metric, and including it
in a proximity panel would invite the reader to compare the rest of its row.
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modules.constants import (PALETTE, METHOD_LABEL, TEXT_SECONDARY, GRID_COLOR)

SUM = "data/betweenBranch.csv"
OUT = "data/images/betweenBranch_proximity.png"
SERIES = ["method", "baseline"]
TITLE = "Euclidean metric, Ward linkage"


def f(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


vals, pairs = {}, set()
for r in csv.DictReader(open(SUM)):
    k = (r["dataset"], r["pair_idx"])
    pairs.add(k)
    v = f(r.get("avg_proximity_norm", ""))
    if v is not None:
        vals[(k[0], k[1], r["which"])] = v

N = len(pairs)
print("pairs:", N, "datasets:", len({p[0] for p in pairs}))

fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=300)
for m in SERIES:
    v = sorted(x for (d, p, w), x in vals.items() if w == m)
    if not v:
        continue
    grid, ys = [], []
    for pct in range(5, 101, 5):
        i = int(round(pct / 100.0 * N)) - 1
        if 0 <= i < len(v):
            grid.append(pct); ys.append(v[i])
    ax.plot(grid, ys, lw=2, marker="o", ms=5, color=PALETTE[m],
            label=METHOD_LABEL[m], mec="white", mew=1.2, solid_capstyle="round")
    print("  %-10s n=%4d (%.1f%% of pairs)  median=%.3f  p90=%.3f" % (
        METHOD_LABEL[m], len(v), 100.0 * len(v) / N,
        v[len(v) // 2], v[min(len(v) - 1, int(0.9 * len(v)))]))

ax.set_xlabel("Validity   (% of valid counterfactuals)", fontsize=18)
ax.set_ylabel("Normalised Proximity", fontsize=18)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID_COLOR); ax.spines["bottom"].set_color(GRID_COLOR)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=16)
ax.grid(True, color=GRID_COLOR, linewidth=0.8); ax.set_axisbelow(True)
ax.set_xlim(0, 100); ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=15, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), ncol=3)
os.makedirs("data/images", exist_ok=True)
fig.tight_layout(); fig.savefig(OUT, facecolor="white")
print("wrote", OUT)
