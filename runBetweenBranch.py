"""
Between-branch diversity pilot (Ward, Euclidean).

The method section defines between-branch diversity but the experiments section
never measures it, although it forward-references the (lambda, delta) pairs
("We use the pairs ... to measure hierarchical diversity in Sec. exp").
This driver supplies that measurement.

Design
------
Alternative targets follow the paper's own prescription: the competitors in C,
plus their subtrees to a chosen depth. Each alternative is an INDEPENDENT
Eq. (10) problem with its own competitor set C_{B'} -- unlike within-branch
diversity, nothing is shared between two alternatives except the query.

Methods
-------
  method    HiCE: solve Eq. (10) per alternative target.
  baseline  the same per-target move, ignoring competitors (baselineWard).
  flatcut   cut the dendrogram into m clusters and use those as the targets.
            This tests the paper's explicit claim that a flat-cut method's
            "alternatives all receive labels of one cut and therefore share a
            single level of granularity" -- i.e. it should collapse to few
            distinct specificities.

Scoring
-------
Validity is always against the alternative's OWN target and OWN competitors:
targeting B' does not exempt it. Coverage is the fraction of available
(lambda, delta) cells reached, with the denominator fixed to the candidate
target set so a method is never credited for a cell it was never offered.
"""
import argparse, csv, time
import multiprocessing as mp
import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
from scipy.spatial.distance import pdist

from modules.hiceCounterfactuals import hiceCounterfactuals
from modules.utilityFunctions import load_openml_cached, get_children
from modules.metrics import (validity, proximity, junction, specificity,
                             junction_specificity_coverage, _parent_map)

REVALIDATION_TOL = 1e-6
# Raised from 1e-9 on 2026-11-01. HiCE minimises proximity, so its solution
# lands exactly on the strict domination boundary; the feasible set is open,
# the infimum is attained only in the limit, and a 1e-9 revalidation rejected
# the returned point as invalid. Measured margins on every affected pair of
# the Ward run: worst -1.4e-07, median -3.5e-09 absolute (worst relative
# -3.0e-08) -- Clarabel's interior-point convergence residue. 1e-6 sits
# comfortably above that noise floor and far below any real violation
# (metrics.DEGENERATE_TIE_TOL is 1e-3).
#
# EUCLIDEAN ONLY. Hamming distances are integers with no rounding noise, and
# any positive tolerance there turns strict domination into non-strict -- it
# would count exact ties as valid and overstate validity on categorical data.
# _reval_eps() pins Hamming to EXACTLY 0.0 and must keep doing so.


class MethodTimeout(Exception):
    pass


def _bb_worker(q, use_hice, cf, D_lv, A_lv, eps):
    try:
        if use_hice:
            # exhaustive=False: the paired A/B over identical pairs found the
            # exhaustive enumeration returned an identical ||t|| on 150/150
            # pairs (0.0000% mean improvement) while costing the long runtime
            # tail. It does not certify optimality either, since subsets of
            # size >= 2 are solved by multistart SLSQP. See 01_ward_phases.md.
            r = cf.counterfactual_ward_branch(B_leaves=D_lv, A_leaves=A_lv, eps=eps,
                                              sparsity=False, exhaustive=False)
        else:
            r = cf.baselineWard(D_lv, A_lv)
        q.put(("ok", r))
    except Exception as e:
        q.put(("err", "%s: %s" % (type(e).__name__, e)))


def bounded_call(use_hice, cf, D_lv, A_lv, eps, budget):
    """
    One solve under a HARD wall-clock wall, via a forked child that is killed
    if it overruns. An in-process SIGALRM cannot do this: handlers only run
    between bytecodes, and a solve sitting inside a C extension holds the GIL
    until it returns on its own. Killing the child is the only mechanism that
    bounds arbitrary code. (Mirrors runCompleteLinkage._run_one_bounded; see
    its docstring for the splice pair that ran two days under a 1200s limit.)
    """
    if budget is None:
        q = []
        _bb_worker(type("Q", (), {"put": lambda self, v: q.append(v)})(),
                   use_hice, cf, D_lv, A_lv, eps)
        tag, val = q[0]
        if tag == "err":
            raise RuntimeError(val)
        return val
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_bb_worker, args=(q, use_hice, cf, D_lv, A_lv, eps),
                       daemon=True)
    proc.start()
    proc.join(budget)
    if proc.is_alive():
        proc.terminate(); proc.join(5)
        if proc.is_alive():
            proc.kill(); proc.join(5)
        raise MethodTimeout("exceeded %.0fs wall-clock limit" % budget)
    if q.empty():
        raise RuntimeError("solver child died without returning")
    tag, val = q.get()
    if tag == "err":
        raise RuntimeError(val)
    return val


def descendants(Z, n, node, max_depth):
    """node and its descendants down to max_depth edges below it (branches only)."""
    out, queue = [], [(int(node), 0)]
    while queue:
        nd, dep = queue.pop(0)
        out.append(nd)
        if dep >= max_depth:
            continue
        ch = get_children(Z, n, nd)
        if ch is None:
            continue
        for c in ch:
            if c >= n:                      # never a leaf target
                queue.append((int(c), dep + 1))
    return out


def run_dataset(ds, k, max_pairs, depth, eps, seed, timeout, writer, raw_writer,
                fh=None, rfh=None):
    X, _, _ = load_openml_cached(ds, 'ward')
    Z = scipy_linkage(X, method='ward')
    n = len(X)
    cf = hiceCounterfactuals(X=X, Z=Z, metric='euclidean', linkage='ward')
    par = _parent_map(Z, n)
    pairs = cf.enumerate_pairs(1, 99999999, max_pairs, seed=seed)
    print("[data] %s: %s | %d pairs" % (ds, X.shape, len(pairs)), flush=True)

    for idx, (A_lv, B_lv) in enumerate(pairs):
        A_pts = X[sorted(A_lv)]
        node_A, _, _ = cf._resolve_branch(A_lv)
        node_B, _, _ = cf._resolve_branch(B_lv)
        comps, _, _, _ = cf.competitor_leaf_sets(A_lv, B_lv)

        # ---- candidate alternative targets: competitors + their subtrees ----
        cand = []
        for c_leaves in comps:
            try:
                c_node, _, _ = cf._resolve_branch(c_leaves)
            except Exception:
                continue
            if c_node < n:                  # a leaf competitor is not a target
                continue
            cand.extend(descendants(Z, n, c_node, depth))
        cand = sorted(set(cand))
        if not cand:
            continue

        # order the budget breadth-first by (junction, specificity), so it
        # spans junctions before it spends depth
        def key(d):
            lam = junction(Z, n, node_A, d, par)
            return (specificity(Z, n, lam, d, par), lam, d)
        cand_ordered = sorted(cand, key=key)

        # ---- flat-cut targets: clusters of one cut, k of them ----
        lab = fcluster(Z, t=max(k, 2), criterion='maxclust')
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

        # The coverage DENOMINATOR must be the same for every method, or the
        # numbers are not comparable. Scoring flatcut against its own target
        # list (the original `flat if which == 'flatcut' else cand_ordered`)
        # made its coverage 1.000 by construction -- it reached 6 of its own 6
        # cells while HiCE was scored against all 17 the candidate set offers --
        # which read as "a flat cut beats HiCE on every diversity metric" and is
        # purely an artefact. The shared denominator is the union of everything
        # any method could target, so no method is credited for a cell it was
        # never offered and none is let off for the cells it cannot reach.
        denom = sorted(set(cand_ordered) | set(flat))

        for which in ('method', 'baseline', 'flatcut'):
            targets = flat if which == 'flatcut' else cand_ordered
            reached, norms, t0 = [], [], time.perf_counter()
            for tnode in targets[:k]:
                D_lv = cf.get_leaves(tnode)
                # every alternative is an independent problem: its OWN competitors
                d_comps, _, _, _ = cf.competitor_leaf_sets(A_lv, D_lv)
                d_comp_pts = [X[sorted(c)] for c in d_comps]
                # 'flatcut' differs from 'method' only in WHICH targets are
                # offered, so it must use the same solver; otherwise the
                # comparison would measure the solver, not target selection.
                try:
                    A_prime, info = bounded_call(which in ('method', 'flatcut'),
                                                 cf, D_lv, A_lv, eps, timeout)
                except MethodTimeout:
                    raw_writer.writerow(dict(dataset=ds, pair_idx=idx, which=which,
                                             target_node=tnode, valid=0, l2='',
                                             scale='', error='MethodTimeout'))
                    continue
                except Exception as e:
                    raw_writer.writerow(dict(dataset=ds, pair_idx=idx, which=which,
                                             target_node=tnode, valid=0, l2='',
                                             scale='', error=str(e)[:60]))
                    continue
                ok = bool(validity(A_prime, X[sorted(D_lv)], d_comp_pts,
                                   'ward', 'euclidean',
                                   eps=-REVALIDATION_TOL, A_orig=A_pts))
                l2 = float(proximity(info, A_pts, A_prime, 'euclidean')['l2'])
                scale = float(np.linalg.norm(X[sorted(D_lv)].mean(0) - A_pts.mean(0)))
                raw_writer.writerow(dict(
                    dataset=ds, pair_idx=idx, which=which, target_node=tnode,
                    valid=int(ok), l2=l2, scale=scale, error=''))
                if ok:
                    reached.append(tnode)
                    norms.append(l2 / scale if scale > 1e-12 else np.nan)
            el = time.perf_counter() - t0
            cov = junction_specificity_coverage(Z, n, node_A, reached, denom)
            # cell_coverage above is against EVERY candidate; with k < len(denom)
            # it is capped by the budget rather than by the method's diversity.
            # Also score against the cells the first k targets could reach.
            # ...against the cells THIS method's own first k targets could
            # reach, which is what "capped by the budget" means. Not denom[:k]:
            # the shared denominator is a union sorted by node id, so its first
            # k are not the k targets anyone was actually given.
            inb = junction_specificity_coverage(Z, n, node_A, reached, targets[:k])
            writer.writerow(dict(
                dataset=ds, pair_idx=idx, which=which,
                n_candidates=len(denom), k=k, n_valid=len(reached),
                avg_proximity_norm=(float(np.nanmean(norms)) if norms else ''),
                seconds=el,
                cells_in_budget=inb['cells_available'],
                cell_coverage_budget=inb['cell_coverage'],
                junction_coverage_budget=inb['junction_coverage'], **cov))
            # stream per pair: a long dataset must not buffer its whole run,
            # or a kill loses everything written since the last dataset.
            if fh is not None:
                fh.flush()
            if rfh is not None:
                rfh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', nargs='+', default=['iris', 'glass', 'ecoli'])
    ap.add_argument('--out', default='data/betweenBranch.csv')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--max-pairs', type=int, default=25)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--eps', type=float, default=1e-6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--timeout', type=float, default=1200,
                    help="hard per-solve wall-clock budget, enforced by killing "
                         "a forked child; None disables it")
    a = ap.parse_args()

    scols = ['dataset', 'pair_idx', 'which', 'n_candidates', 'k', 'n_valid',
             'avg_proximity_norm', 'seconds', 'cells_reached', 'cells_available',
             'cell_coverage', 'cells_in_budget', 'cell_coverage_budget',
             'junctions_reached', 'junctions_available', 'junction_coverage',
             'junction_coverage_budget', 'specificity_coverage']
    rcols = ['dataset', 'pair_idx', 'which', 'target_node', 'valid', 'l2',
             'scale', 'error']
    raw_path = a.out.replace('.csv', '_raw.csv')
    with open(a.out, 'w', newline='') as fh, open(raw_path, 'w', newline='') as rfh:
        w = csv.DictWriter(fh, fieldnames=scols); w.writeheader()
        rw = csv.DictWriter(rfh, fieldnames=rcols); rw.writeheader()
        for ds in a.dataset:
            try:
                run_dataset(ds, a.k, a.max_pairs, a.depth, a.eps, a.seed, a.timeout, w, rw,
                            fh=fh, rfh=rfh)
                fh.flush(); rfh.flush()
            except Exception as e:
                print("[skip] %s: %s: %s" % (ds, type(e).__name__, e), flush=True)
    print("[out] %s and %s" % (a.out, raw_path))


if __name__ == '__main__':
    main()
