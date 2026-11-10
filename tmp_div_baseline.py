"""
Per-sub-branch baseline for the complete-linkage diversity experiments.

The diversity driver inherits runHierarchical's method list and runs no
baseline, so those figures had no reference line. Rather than compute ONE
baseline counterfactual and ask which sub-branch it lands in -- which needs
geometric routing, and under complete linkage that routing is an
approximation (it collapses A' to a centroid while the linkage distance is a
max over pairs; for Hamming a centroid of integer codes is meaningless) --
the baseline is run ONCE PER SUB-BRANCH, mirroring HiCE's descent.

Each alternative then has a KNOWN target node, so coverage is scored with
tree_coverage_from_targets, exactly as HiCE is: no routing, no approximation,
both methods scored by the same rule.

This is a different baseline from the single-counterfactual one: it changes
the TARGET per alternative, whereas HiCE keeps B as the target and adds
sibling-preference constraints. It answers "what if you moved straight at
each sub-branch, ignoring the competitors?".

Budget: alternatives are taken breadth-first (shallowest first, then node id),
the order HiCE's own selection uses, until k are valid or 3k nodes have been
tried -- the same k-slot budget, with a fair chance to replace invalid tries.
"""
import csv, math, os, time
import multiprocessing as mp
import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist
from modules.hiceCounterfactuals import hiceCounterfactuals
from modules.utilityFunctions import load_openml_cached, get_children
from modules.metrics import validity, tree_coverage_from_targets, proximity

SEED, MIN_LEAF, MAX_LEAF, MAX_PAIRS, K = 0, 1, 99999999, 100, 10
DS_BUDGET = 1800.0            # per-dataset wall, seconds
PAIR_BUDGET = 300.0           # per-PAIR wall, seconds


class PairTimeout(Exception):
    pass


def _pair_worker(q, fn):
    try:
        q.put(("ok", fn()))
    except Exception as e:                      # noqa: BLE001 - reported, not raised
        q.put(("err", "%s: %s" % (type(e).__name__, e)))


def run_bounded(fn, budget):
    """
    Run fn() under a HARD wall-clock wall, in a forked child that is killed if
    it overruns.

    DS_BUDGET alone cannot bound this run: it is tested between pairs, so a
    single pair that never returns stalls forever (mfeat-zernike burned 12.8h
    inside one pair with the dataset check never reached). An in-process
    SIGALRM cannot bound it either -- handlers run only between bytecodes, and
    these solves sit inside C extensions holding the GIL. Killing the child is
    the only mechanism that bounds arbitrary code. Mirrors
    runBetweenBranch.bounded_call.

    fork is required: fn is a closure over cf/X/Z and is never pickled, so the
    child inherits them from memory. Only the RESULT crosses the Queue.
    """
    if budget is None:
        return fn()
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_pair_worker, args=(q, fn), daemon=True)
    proc.start()
    proc.join(budget)
    if proc.is_alive():
        proc.terminate(); proc.join(5)
        if proc.is_alive():
            proc.kill(); proc.join(5)
        raise PairTimeout("exceeded %.0fs wall-clock limit" % budget)
    if q.empty():
        raise RuntimeError("pair child died without returning")
    tag, val = q.get()
    if tag == "err":
        raise RuntimeError(val)
    return val


def _flush(outbase, srows, rawrows):
    if srows:
        with open(outbase + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
            w.writeheader(); w.writerows(srows)
    if rawrows:
        with open(outbase + "_raw.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rawrows[0].keys()))
            w.writeheader(); w.writerows(rawrows)
JOBS = [("data/completeDiversityEuclidean.csv", "euclidean",
         "data/completeDivBaselineEuclidean"),
        ("data/completeDiversityHamming.csv", "hamming",
         "data/completeDivBaselineHamming")]

# DIV_METRIC runs ONE leg, so the two metrics go as separate processes rather
# than one after the other. They touch disjoint OUTBASEs, and a single process
# holding both legs in memory cannot be told to skip one once it has started.
_only = os.environ.get("DIV_METRIC", "").strip().lower()
if _only:
    JOBS = [j for j in JOBS if j[1] == _only]
    if not JOBS:
        raise SystemExit("DIV_METRIC=%r matches no job" % _only)

# ALL_PAIRS ignores the main summary's pair list and solves every enumerated
# pair. The pair enumeration is deterministic (same SEED/MIN_LEAF/MAX_LEAF/
# MAX_PAIRS as the drivers), so pair_idx means the same thing either way and
# the result is a SUPERSET of what the summary asks for; the plot joins on
# (dataset, pair_idx) and drops the extras. This removes a race: the baseline
# used to read the main CSV once at startup while the driver was still
# appending to it, so a dataset mid-flight (nursery at 2 of 100) silently
# fixed the baseline's coverage at whatever had landed by then.
ALL_PAIRS = os.environ.get("DIV_ALL_PAIRS", "") == "1"

for SRC, METRIC, OUTBASE in JOBS:
    want = {}
    for r in csv.DictReader(open(SRC)):
        if r.get("metric") == METRIC:
            want.setdefault(r["dataset"], set()).add(r["pair_idx"])
    print("=== %s: %d datasets%s" % (METRIC, len(want),
          "  [ALL_PAIRS]" if ALL_PAIRS else ""), flush=True)

    srows, rawrows = [], []
    for ds in sorted(want):
        X, metric, _ = load_openml_cached(ds, "complete")
        if metric != METRIC:
            continue
        Z = (scipy_linkage(pdist(X, metric="hamming"), method="complete")
             if metric == "hamming" else scipy_linkage(X, method="complete"))
        cf = hiceCounterfactuals(X=X, Z=Z, metric=metric, linkage="complete", anchor="min")
        pairs = cf.enumerate_pairs(MIN_LEAF, MAX_LEAF, MAX_PAIRS, seed=SEED)
        t_ds, nv_tot = time.perf_counter(), 0
        bailed = False
        for idx, (A_lv, B_lv) in enumerate(pairs):
            if not ALL_PAIRS and str(idx) not in want[ds]:
                continue
            if time.perf_counter() - t_ds > DS_BUDGET:
                # Give up on this dataset rather than stall the whole run; the
                # pairs already done are kept and reported.
                print("  [budget] %s: exceeded %.0fs after %d pairs -- skipping rest"
                      % (ds, DS_BUDGET, idx), flush=True)
                bailed = True
                break
            A_pts, B_pts = X[sorted(A_lv)], X[sorted(B_lv)]
            comps, _, _, _ = cf.competitor_leaf_sets(A_lv, B_lv)
            comp_pts = [X[sorted(c)] for c in comps]
            B_node, _, _ = cf._resolve_branch(B_lv)
            # Must match runCompleteDiversity's scale exactly: the hamming cost
            # is changed CELLS summed over A's |A| points, so the denominator
            # is |A|*d, not d.
            scale = (float(X.shape[1]) * float(A_pts.shape[0]) if metric == "hamming"
                     else float(np.linalg.norm(B_pts.mean(0) - A_pts.mean(0))))

            order, queue = [], [(B_node, 0)]
            par, sib = {}, {}
            while queue:
                node, dep = queue.pop(0)
                order.append((dep, node))
                ch = get_children(Z, len(X), node)
                if ch is None:
                    continue
                for c in ch:
                    # A sibling may be a LEAF even though a target may not, so
                    # record the pairing for every child; only internal nodes
                    # are enqueued as candidate targets.
                    par[c] = node
                    sib[c] = ch[1] if c == ch[0] else ch[0]
                    if c >= len(X):            # internal nodes only: never a leaf target
                        queue.append((c, dep + 1))
            order.sort()

            _pts_cache = {}

            def pts_of(n):
                # Memoised: the sibling check calls this once per level per
                # candidate per pair, and under complete linkage validity()
                # is O(|P||Q|d), so recomputing leaf sets on a large
                # dendrogram dominated the run (10h on mfeat-zernike).
                v = _pts_cache.get(n)
                if v is None:
                    v = X[[n]] if n < len(X) else X[sorted(cf.get_leaves(n))]
                    _pts_cache[n] = v
                return v

            def work():
                targets, l2s, raw_local, tried = [], [], [], 0
                for dep, node in order:
                    if len(targets) >= K or tried >= 3 * K:
                        break
                    tried += 1
                    try:
                        A_prime, info = cf.baselineCompleteLinkage(cf.get_leaves(node), A_lv)
                    except Exception:
                        continue
                    # Validity is ALWAYS against the original problem: dominate B
                    # against B's competitors. Targeting D does not exempt it.
                    # Hamming stays EXACTLY 0 (integer distances, no noise to
                    # absorb); Euclidean matches the drivers' REVALIDATION_TOL,
                    # raised to 1e-6 so a solution sitting on the open domination
                    # boundary is not rejected as invalid. See runHierarchical.py.
                    eps = 0.0 if metric == "hamming" else -1e-6
                    ok = bool(validity(A_prime, B_pts, comp_pts, "complete", metric,
                                       eps=eps, A_orig=A_pts))
                    # Solve the SAME problem HiCE does. Dominating B against B's
                    # competitors is only HiCE's level-0 constraint; its descent
                    # also imposes, at every split on the path from B down to the
                    # target, the sibling-preference constraint
                    #     d(A(t), child) < d(A(t), sibling),
                    # which is exactly validity() with near=child, far=[sibling]
                    # and no A_orig. Without these the baseline searched a strictly
                    # larger feasible set, so neither method dominated the other
                    # and HiCE's per-target optimality was not testable.
                    n = node
                    while ok and n != B_node and n in par:
                        ok = bool(validity(A_prime, pts_of(n), [pts_of(sib[n])],
                                           "complete", metric, eps=eps))
                        n = par[n]
                    # metrics.proximity is the single definition of cost: for
                    # hamming its l2 is the number of changed CELLS (summed over
                    # A's points), which is what HiCE records. Counting changed
                    # FEATURES here instead put the two columns in different units.
                    px = proximity(info, A_pts, A_prime, metric)
                    l2 = float(px["l2"])
                    # n_feat too: the paper normalises hamming proximity by the
                    # FEATURE count, and cells cannot be converted to features
                    # after the fact (see runCompleteDiversity's note).
                    raw_local.append(dict(dataset=ds, metric=metric, pair_idx=idx,
                                          which="baseline", cf_idx=len(targets),
                                          valid=ok, l2=l2, n_feat=px["n_feat"]))
                    if ok:
                        targets.append(node); l2s.append(l2)
                return targets, l2s, raw_local

            t0 = time.perf_counter()
            try:
                targets, l2s, raw_local = run_bounded(work, PAIR_BUDGET)
            except (PairTimeout, RuntimeError) as e:
                # The child is dead, so there are no partial results to keep:
                # the pair is recorded as yielding nothing, which is what a
                # method that cannot answer in the budget has earned.
                print("    [pair] %s idx=%d: %s" % (ds, idx, e), flush=True)
                targets, l2s, raw_local = [], [], []
            rawrows.extend(raw_local)
            el = time.perf_counter() - t0
            n_rep, n_sub = tree_coverage_from_targets(Z, len(X), B_node, targets)
            nv_tot += len(targets)
            srows.append(dict(dataset=ds, metric=metric, pair_idx=idx,
                              which="baseline", n_valid=len(targets), scale=scale,
                              avg_proximity_norm=(float(np.mean(l2s))/scale
                                                  if l2s and scale > 0 else ""),
                              n_subbranches=n_sub, n_subbranches_represented=n_rep,
                              tree_coverage=(n_rep/n_sub) if n_sub > 0 else "",
                              seconds=el))
        print("  %-28s %4d pairs  valid CFs=%5d  %6.1fs%s" % (
            ds, len(pairs) if ALL_PAIRS else len(want[ds]),
            nv_tot, time.perf_counter()-t_ds,
            "  (PARTIAL)" if bailed else ""), flush=True)
        # Write after EVERY dataset: a stall must not cost the whole run.
        _flush(OUTBASE, srows, rawrows)

    _flush(OUTBASE, srows, rawrows)
    print("wrote %s{,_raw}.csv  (%d summary, %d raw)\n" % (OUTBASE, len(srows), len(rawrows)), flush=True)
