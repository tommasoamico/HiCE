"""
Re-score the between-branch run against a SHARED coverage denominator.

The run on disk scored `flatcut` against its own target list, so its coverage
was 1.000 by construction (runBetweenBranch.py, pre-fix). Re-solving is not
needed to repair this: betweenBranch_raw.csv records (pair, method,
target_node, valid) for every attempt, and the candidate sets are a
deterministic function of the dendrogram and the pair seed. So we rebuild the
candidate sets exactly as the driver does, take the reached targets from the
raw file, and re-score.

Reproducing the driver's construction EXACTLY is the whole correctness
requirement here: same enumerate_pairs(1, 99999999, max_pairs, seed), same
competitor->subtree expansion at the same depth, same breadth-first ordering,
same fcluster cut. Anything that drifts silently re-indexes pair_idx and the
raw rows attach to the wrong pair.

Writes data/betweenBranch_rescored.csv with the same schema plus `which`-wise
counts, and prints the HiCE / baseline / flatcut comparison.
"""
import csv, collections
import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster

from modules.hiceCounterfactuals import hiceCounterfactuals
from modules.utilityFunctions import load_openml_cached, get_children
from modules.metrics import (junction, specificity,
                             junction_specificity_coverage, _parent_map)
from runBetweenBranch import descendants

RAW = "data/betweenBranch_raw.csv"
OUT = "data/betweenBranch_rescored.csv"
K = 10
DEPTH = 2
MAX_PAIRS = 100
SEED = 0

# reached[(dataset, pair_idx, which)] = [target_node, ...] that were VALID
reached = collections.defaultdict(list)
attempted = collections.Counter()
for r in csv.DictReader(open(RAW)):
    key = (r["dataset"], int(r["pair_idx"]), r["which"])
    attempted[key] += 1
    if r["valid"] in ("1", "True", "true"):
        reached[key].append(int(r["target_node"]))

datasets = sorted({k[0] for k in attempted})
print("datasets in raw:", len(datasets))

cols = ["dataset", "pair_idx", "which", "n_shared_candidates", "n_offered",
        "n_attempted", "n_valid", "cells_reached", "cells_available",
        "cell_coverage", "junctions_reached", "junctions_available",
        "junction_coverage", "specificity_coverage",
        "distinct_specificities", "distinct_junctions"]
out_rows = []

for ds in datasets:
    X, _, _ = load_openml_cached(ds, 'ward')
    Z = scipy_linkage(X, method='ward')
    n = len(X)
    cf = hiceCounterfactuals(X=X, Z=Z, metric='euclidean', linkage='ward')
    par = _parent_map(Z, n)
    pairs = cf.enumerate_pairs(1, 99999999, MAX_PAIRS, seed=SEED)

    for idx, (A_lv, B_lv) in enumerate(pairs):
        if not any((ds, idx, w) in attempted for w in ("method", "baseline", "flatcut")):
            continue
        node_A, _, _ = cf._resolve_branch(A_lv)
        comps, _, _, _ = cf.competitor_leaf_sets(A_lv, B_lv)

        cand = []
        for c_leaves in comps:
            try:
                c_node, _, _ = cf._resolve_branch(c_leaves)
            except Exception:
                continue
            if c_node < n:
                continue
            cand.extend(descendants(Z, n, c_node, DEPTH))
        cand = sorted(set(cand))
        if not cand:
            continue

        def key(d):
            lam = junction(Z, n, node_A, d, par)
            return (specificity(Z, n, lam, d, par), lam, d)
        cand_ordered = sorted(cand, key=key)

        lab = fcluster(Z, t=max(K, 2), criterion='maxclust')
        flat = []
        for lv in np.unique(lab):
            leaves = [i for i in range(n) if lab[i] == lv]
            if set(leaves) <= set(A_lv) or len(leaves) < 2:
                continue
            try:
                fnode, _, _ = cf._resolve_branch(leaves)
            except Exception:
                continue
            flat.append(int(fnode))
        flat = sorted(set(flat))

        denom = sorted(set(cand_ordered) | set(flat))

        for which in ("method", "baseline", "flatcut"):
            kkey = (ds, idx, which)
            if kkey not in attempted:
                continue
            offered = (flat if which == "flatcut" else cand_ordered)[:K]
            got = reached[kkey]
            cov = junction_specificity_coverage(Z, n, node_A, got, denom)
            # how many DISTINCT specificities / junctions the valid set spans:
            # the paper's claim about flat cuts is that they collapse to one
            # level of granularity, which is a statement about this count.
            cells = set()
            for d_ in got:
                lam = junction(Z, n, node_A, d_, par)
                cells.add((lam, specificity(Z, n, lam, d_, par)))
            out_rows.append(dict(
                dataset=ds, pair_idx=idx, which=which,
                n_shared_candidates=len(denom), n_offered=len(offered),
                n_attempted=attempted[kkey], n_valid=len(got),
                distinct_specificities=len({c[1] for c in cells}),
                distinct_junctions=len({c[0] for c in cells}), **cov))
    print("  %-34s pairs scored: %d" % (ds, sum(1 for r in out_rows if r["dataset"] == ds) // 3))

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(out_rows)
print("wrote", OUT, len(out_rows), "rows")

# ---- report ----
import statistics as st
by = collections.defaultdict(list)
for r in out_rows:
    by[r["which"]].append(r)
print("\n%-9s %7s %7s %7s %7s %7s %7s" % (
    "method", "cells", "junc", "spec", "#spec", "#junc", "valid"))
for w in ("method", "baseline", "flatcut"):
    R = by[w]
    g = lambda c: [float(r[c]) for r in R if not np.isnan(float(r[c]))]
    print("%-9s %7.3f %7.3f %7.3f %7.2f %7.2f %7.1f" % (
        w, st.mean(g("cell_coverage")), st.mean(g("junction_coverage")),
        st.mean(g("specificity_coverage")),
        st.mean(g("distinct_specificities")), st.mean(g("distinct_junctions")),
        st.mean(g("n_valid"))))
