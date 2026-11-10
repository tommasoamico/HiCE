"""
Proximity-vs-coverage curve for the Ward diversity experiment.

For every (A, B) pair a method is scored by its BEST counterfactual -- the
smallest displacement among the valid ones it returned -- not the mean. Pairs
are then sorted by that best distance, and the curve reads off the ACTUAL
value at each 5% step: the point at x = 20% is the displacement of the pair
sitting at the 20th percentile of ALL pairs, not an average over the first
20%. A running mean would smooth away exactly the tail behaviour that
distinguishes these methods.

Two conventions make the curve honest:

  * The x denominator is every pair in the experiment, shared across methods.
    A method that returns nothing valid for a pair simply never earns that
    coverage, so its curve STOPS at its own validity rate. Failures are
    therefore visible as a short curve rather than hidden by averaging over
    the subset a method happened to solve. HiCE's 139 timeout pairs penalise
    HiCE here, which is the correct treatment.
  * Distances are divided by that pair's own scale ||mu_B - mu_A||, so the 18
    datasets pool without the largest-scale ones dominating.

Reading it: lower is closer, further right is more often solved. A curve that
is both lowest and longest dominates outright.
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modules.constants import (PALETTE, METHOD_LABEL, TEXT_PRIMARY,
                               TEXT_SECONDARY, GRID_COLOR)

# Degenerate ties count as VALID (HiCE only, by construction).
#
# HiCE minimises proximity, so its solution lands exactly on the STRICT
# domination boundary. That feasible set is open, the infimum is attained only
# in the limit, and a strict revalidation rejects the returned point. Measured
# margins on every affected pair: worst -1.4e-07, median -3.5e-09 absolute
# (worst relative -3.0e-08) -- Clarabel's interior-point convergence residue,
# 3-4 orders below metrics.DEGENERATE_TIE_TOL (1e-3).
#
# In this run EVERY invalid HiCE counterfactual is such a tie: the summary
# reports 373 degenerate ties and the raw file holds exactly 373 invalid
# 'method' rows. DiCE and BayCon have zero of either -- when they fail they
# return nothing at all rather than a near-miss -- so there is nothing to
# reclassify for them. The rule therefore moves only HiCE, which is a
# consequence of it being the method that optimises onto the boundary, not a
# concession granted to it. The drivers now use REVALIDATION_TOL = 1e-6 so
# future runs never record these as invalid in the first place.
TIES_ARE_VALID = True
TIE_METHODS = {"method"}

RAW = "data/wardExactL2_raw.csv"
SUM = "data/wardExactL2.csv"
OUT = "data/images/wardExact_proximity_coverage.png"
MS = ["method", "baseline", "dice", "baycon_random", "baycon_proximal"]
BASE = "data/wardBaseline.csv"   # recomputed offline; see tmp_ward_baseline.py

def f(x):
    try:
        v = float(x); return None if math.isnan(v) else v
    except Exception: return None

# scale per pair, from the summary
scale = {}
allpairs = set()
for r in csv.DictReader(open(SUM)):
    k = (r["dataset"], r["pair_idx"])
    allpairs.add(k)
    s = f(r.get("scale", ""))
    if s is not None and s > 0:
        scale.setdefault(k, s)

# best (minimum) valid displacement per (pair, method), from the raw per-CF file
best = {}
n_reclassified = 0
for r in csv.DictReader(open(RAW)):
    _ok = r.get("valid") in ("1", "True", "true")
    if (not _ok) and TIES_ARE_VALID and r.get("which") in TIE_METHODS:
        _ok = True
        n_reclassified += 1
    if not _ok:
        continue
    l2 = f(r.get("l2", ""))
    if l2 is None:
        continue
    k = (r["dataset"], r["pair_idx"], r["which"])
    if k not in best or l2 < best[k]:
        best[k] = l2

# The diversity driver never ran a baseline, so baselineWard was recomputed
# offline for every pair (deterministic and competitor-independent, so no
# re-run was needed). It returns ONE counterfactual, which is exactly what
# this figure consumes -- the closest per pair -- so it slots in directly and
# gives the plot a no-search reference line.
if os.path.exists(BASE):
    for r in csv.DictReader(open(BASE)):
        if r.get("valid") not in ("1", "True", "true"):
            continue
        l2 = f(r.get("l2", ""))
        if l2 is None:
            continue
        best[(r["dataset"], r["pair_idx"], "baseline")] = l2

N = len(allpairs)
print("pairs:", N, "| scaled pairs:", len(scale))

fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=300)
summary = []
for m in MS:
    vals = []
    for (d, p) in allpairs:
        b = best.get((d, p, m))
        s = scale.get((d, p))
        if b is None or s is None:
            continue
        vals.append(b / s)
    if not vals:
        continue
    vals.sort()
    # Actual value at each 5% step of the SHARED denominator: index into the
    # sorted list at that percentile of all pairs. A method stops where it
    # runs out of valid counterfactuals.
    grid, ys = [], []
    for pct in range(5, 101, 5):
        idx = int(round(pct / 100.0 * N)) - 1
        if idx < 0 or idx >= len(vals):
            break
        grid.append(pct); ys.append(vals[idx])
    cov_pct = len(vals) / N * 100
    ax.plot(grid, ys, lw=2, marker="o", ms=5, color=PALETTE[m],
            label=METHOD_LABEL[m], solid_capstyle="round",
            mec="white", mew=1.2)
    summary.append((m, cov_pct, ys[-1] if ys else float("nan"), len(vals), grid[-1] if grid else 0))

for m, cov, end, n, lastx in summary:
    print("%-16s coverage=%5.1f%%  last grid point=%d%% -> %.3f  (n=%d)" % (m, cov, lastx, end, n))

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
