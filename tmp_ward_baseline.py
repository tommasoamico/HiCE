"""
Recompute the Ward baseline offline for every pair of the finished diversity run.

The diversity driver only ran method/dice/baycon_*, so the experiment has no
naive reference point -- unlike the complete-linkage runs, which include the
min-max baseline. baselineWard is deterministic and competitor-independent
(it translates mu_A onto B's nearest sample, ignoring competitors), so it can
be reconstructed exactly without re-running 1800 pairs of search.

Pair identity is reproduced by replaying enumerate_pairs with the SAME
(seed, min_leaf, max_leaf, max_pairs) the run used, which is the same
reproducibility contract utilityFunctions.call_seed documents. The driver
skips pairs whose scale is ~0 without writing rows, so those are skipped here
too; pair_idx is the enumeration index either way, so the numbering lines up.
"""
import csv, os, sys
import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from modules.hiceCounterfactuals import hiceCounterfactuals
from modules.utilityFunctions import load_openml_cached
from modules.metrics import validity, tree_coverage

SUM = "data/hiceSparsityCoverageBigTimeout.csv"
OUT = "data/wardBaseline.csv"
REVALIDATION_TOL = 1e-9          # same as runHierarchical
SEED, MIN_LEAF, MAX_LEAF, MAX_PAIRS = 0, 1, 99999999, 100

datasets = []
for r in csv.DictReader(open(SUM)):
    if r["dataset"] not in datasets:
        datasets.append(r["dataset"])
print("datasets:", len(datasets), flush=True)

rows = []
for ds in datasets:
    try:
        X, metric, _ = load_openml_cached(ds, "ward")
    except Exception as e:
        print("  [skip] %s: %s" % (ds, str(e)[:70]), flush=True)
        continue
    Z = scipy_linkage(X, method="ward")
    cf = hiceCounterfactuals(X=X, Z=Z, metric=metric, linkage="ward", anchor="min")
    pairs = cf.enumerate_pairs(MIN_LEAF, MAX_LEAF, MAX_PAIRS, seed=SEED)
    n_ok = 0
    for idx, (A_lv, B_lv) in enumerate(pairs):
        A_pts, B_pts = X[sorted(A_lv)], X[sorted(B_lv)]
        scale = float(np.linalg.norm(B_pts.mean(0) - A_pts.mean(0)))
        if scale <= 1e-12:                 # driver skips these without writing
            continue
        comps, _, _, _ = cf.competitor_leaf_sets(A_lv, B_lv)
        comp_pts = [X[sorted(c)] for c in comps]
        A_prime, info = cf.baselineWard(B_lv, A_lv)
        ok = bool(validity(A_prime, B_pts, comp_pts, "ward", metric,
                           eps=-REVALIDATION_TOL, A_orig=A_pts))
        l2 = float(np.linalg.norm(np.asarray(info["t"], float)))
        # Tree coverage for the single baseline counterfactual, scored with
        # the SAME geometric routing used for DiCE/BayCon (tree_coverage, not
        # tree_coverage_from_targets -- the baseline carries no target-node
        # metadata, exactly like them). Only the valid CF is scored; an
        # invalid one represents nothing.
        B_node, _, _ = cf._resolve_branch(B_lv)
        pts = [np.asarray(A_prime, float).mean(0)] if ok else []
        n_rep, n_sub = tree_coverage(Z, len(X), B_node, X, pts, metric,
                                     n_A=A_pts.shape[0])
        tc = (n_rep / n_sub) if n_sub > 0 else ""
        rows.append(dict(dataset=ds, pair_idx=idx, which="baseline",
                         valid=ok, l2=l2, scale=scale,
                         n_A=len(A_lv), n_B=len(B_lv),
                         n_subbranches=n_sub, n_subbranches_represented=n_rep,
                         tree_coverage=tc))
        n_ok += ok
    print("  %-34s pairs=%4d  valid=%4d" % (ds, len(pairs), n_ok), flush=True)

os.makedirs("data", exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["dataset","pair_idx","which","valid",
                                        "l2","scale","n_A","n_B",
                                        "n_subbranches","n_subbranches_represented",
                                        "tree_coverage"])
    w.writeheader(); w.writerows(rows)
print("\nwrote %s (%d rows, %d valid = %.1f%%)" % (
    OUT, len(rows), sum(r["valid"] for r in rows),
    100*sum(r["valid"] for r in rows)/max(len(rows),1)))
