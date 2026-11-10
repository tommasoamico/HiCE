"""
Diversity experiment for COMPLETE linkage (Euclidean).

A deliberate near-clone of runHierarchical.py -- same pair enumeration, same
competitor sets, same k-counterfactual selection, same output schema -- so the
two experiments are directly comparable and the Ward figures work unchanged on
this CSV. Only the linkage-specific pieces differ:

  * LINKAGE = 'complete', and the tree is built with method='complete'.
  * HiCE uses diverseCounterfactualsComplete: each level of descent adds one
    witness-disjunction BLOCK (near = target child, far = its sibling) rather
    than Ward's single Apollonius surface, because complete-linkage distance
    is a max over pairs. See that method's docstring.
  * tree_coverage is told linkage='complete' so DiCE/BayCon counterfactuals
    are routed by FARTHEST-pair distance rather than Ward-weighted centroid
    distance. HiCE is scored by exact target-node identity
    (tree_coverage_from_targets), which is linkage-agnostic.
  * Per-NODE solves are bounded by Gurobi's own TimeLimit (passed through
    diverseCounterfactualsComplete's time_limit), which is what actually
    binds a C-extension solve. The outer per-call time_limit() is a SIGALRM
    and CANNOT interrupt Gurobi mid-solve, so a pair with very many nodes can
    still overrun its nominal budget -- it is bounded by
    (nodes x node_budget), not by --timeout. runCompleteLinkage.py solves
    this properly with a killed child process; if overruns show up here, port
    _run_one_bounded across rather than trusting --timeout.

Unlike Ward, complete linkage can be genuinely INFEASIBLE for a pair; such
nodes prune their subtree and are reported, not treated as failures.

Original header follows.

Diversity experiment for the hiceCounterfactuals class.

For each of --max-pairs randomly-chosen (A, B) branch pairs, generate up to
--k counterfactuals per method:

  - method           : cf.diverseCounterfactuals -- one CF per node of B's
                        subtree (framing B, sibling-preference constraints),
                        the k with the smallest depth (broadest first) kept.
  - dice             : DiCE genetic search asked for exactly k CFs
                        (total_CFs=k); DiCE's own diversity objective picks
                        the k, we only validity-filter and take the first k
                        of whatever came back valid.
  - baycon_random    : k CFs chosen uniformly at random from everything
                        BayCon's search accumulated.
  - baycon_proximal  : the k closest (by L2) CFs from that same BayCon pool.
    (baycon_random and baycon_proximal share a single BayCon search call
    per pair -- BayCon is not re-run for each.)

Ward linkage + Euclidean metric only: diverseCounterfactuals is Ward-only,
and the DiCE/BayCon wrappers only support Ward's always-feasible mu_B seed.

For every (pair, method) we compute, from the VALID counterfactuals only:
  - avg_proximity          : mean L2 displacement ||t_i|| from A.
  - avg_pairwise_distance  : mean pairwise L2 distance between the CFs'
                              displacement vectors, ||t_i - t_j|| (a
                              proxy for how diverse the returned set is).
Both are then divided by that pair's own characteristic scale,
scale = ||mu_B - mu_A|| (the direct centroid-to-centroid distance), so pairs
of very different sizes/scales become comparable before pooling across pairs
-- this is the "normalize both so we have something comparable" step.

Two CSVs are written (both append-only / resumable per (dataset, which)):
  - a raw file, one row per (pair, method, cf index): --out with "_raw"
    inserted before the extension.
  - the summary file (--out itself), one row per (pair, method) with the
    (normalized) proximity/diversity aggregates above.

A scatter plot (mean-over-pairs avg_proximity_norm vs. mean-over-pairs
avg_pairwise_distance_norm, one point per method, error bars = std across
pairs) is produced from the summary file into --out-dir.

Usage
-----
    python runHierarchical.py --dataset iris --out data/iris_diversity.csv
    python runHierarchical.py --dataset iris --k 10 --max-pairs 100
"""

import argparse
import csv
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist
from tqdm import tqdm

from modules.constants import (
    DIVERSITY_RAW_FIELDNAMES, DIVERSITY_SUMMARY_FIELDNAMES,
)
from modules.hiceCounterfactuals import hiceCounterfactuals
from modules.utilityFunctions import (
    MethodTimeout, _prepare_out_csv, time_limit,
    call_seed, _seed_everything, load_openml_cached,
)
from modules.metrics import (
    validity, proximity, tree_coverage, tree_coverage_from_targets, is_degenerate_tie,
)
from modules.dice import run_dice_baseline_multi
from modules.baycon import run_baycon_baseline_multi
from visualizations import _build_style, _style_axes


LINKAGE = 'complete'

# Revalidation numerical tolerance: the SOCP solver's own domination check
# (dB < dX) can land in a floating-point tie at ~1e-11 with a competitor when
# the true optimal displacement is ~0 (e.g. A is a near-duplicate of a real
# dataset row) -- metrics.validity's strict eps=0.0 comparison then rejects a
# genuinely optimal, feasible=True solution over solver rounding noise, not a
# real domination violation. Passed as a NEGATIVE eps (validity checks
# dB < dX - eps, so a negative eps relaxes the margin to dB < dX + tol),
# absorbing that noise without weakening the actual validity definition.
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


def _reval_eps(metric):
    """
    Revalidation eps, NEGATIVE, relaxing validity's strict `dB < dX - eps`.

    Euclidean: -REVALIDATION_TOL, absorbing solver rounding noise.

    Hamming: EXACTLY 0. Hamming distances are integers, so there is no
    rounding noise to absorb, and `dB < dX + 1e-9` is just `dB <= dX` on
    integers -- it silently turns strict domination into non-strict and
    counts ties as valid, overstating validity on categorical data while
    the exact solver correctly reports those same pairs infeasible.
    (Mirrors runCompleteLinkage._reval_eps; this driver was still passing
    -REVALIDATION_TOL for both metrics.)
    """
    return 0.0 if metric == 'hamming' else -REVALIDATION_TOL


# ======================================================================
# Per-method-call CF-list generation
# ======================================================================
def _hice_multi(cf, B_lv, A_lv, k, node_budget=120):
    """
    Up to k counterfactuals from cf.diverseCounterfactuals: feasible nodes
    sorted breadth-first (shallowest depth, then node id) so the k kept span
    as much of B's subtree as the budget allows, rather than one deep path.
    diverseCounterfactuals now builds a direct child-vs-sibling Apollonius
    constraint at each split (d(A, child) < d(A, sibling), using the
    child's own centroid/weight) on top of the original B-domination
    constraints, rather than framing the sibling as a competitor against B
    -- see that method's docstring for why the latter was too weak to
    reliably force the split. Every returned CF stays validated against the
    original (A, B) pair's competitors.

    include_leaf_targets=False: a leaf target can never itself count as a
    represented sub-branch (metrics.tree_coverage_from_targets), so this
    experiment only cares about branch (>=2-leaf) targets -- asking
    diverseCounterfactuals to skip leaves entirely both saves the wasted
    solver calls on nodes whose result could never be used for coverage,
    and means every one of the k returned CFs is coverage-eligible (no
    budget slot can fall through to a leaf once branch targets run out;
    if fewer than k branch targets exist/are feasible, fewer than k are
    simply returned instead).
    """
    if getattr(cf, 'metric', 'euclidean') == 'hamming':
        results = cf.diverseCounterfactualsCompleteHamming(
            B_leaves=B_lv, A_leaves=A_lv, both_children=True,
            include_leaf_targets=False, time_limit=node_budget)
    else:
        results = cf.diverseCounterfactualsComplete(
            B_leaves=B_lv, A_leaves=A_lv, both_children=True,
            include_leaf_targets=False, time_limit=node_budget)
    feasible = [r for r in results if r['info'].get('feasible', False)]

    # Breadth-first: shallowest depth first, then node id as a tiebreak, so
    # the k kept span as much of B's subtree as the budget allows.
    candidates = sorted(feasible, key=lambda r: (r['depth'], r['target_node']))
    chosen = candidates[:k]
    # Stash target_node onto info so _cf_list_stats can compute exact
    # (non-geometric) tree coverage for HiCE -- see
    # metrics.tree_coverage_from_targets for why this is needed instead of
    # tree_coverage's geometric routing.
    out = []
    for r in chosen:
        info = dict(r['info'])
        info['target_node'] = r['target_node']
        out.append((r['A_prime'], info))
    return out, len(results)


def _cf_list_stats(cf_results, A_pts, B_pts, comp_pts, metric, Z, n_leaves, B_node, X):
    """
    Recomputes validity independently for every (A_prime, info) pair, then
    aggregates proximity / pairwise-diversity / tree coverage over the
    valid ones only. Returns a dict of n_returned, n_valid, avg_proximity,
    avg_pairwise_distance, n_subbranches, n_subbranches_represented,
    tree_coverage, tree_coverage_norm, and per_cf (list of (valid, l2) for
    the raw CSV).

    Tree coverage: if info carries a 'target_node' (HiCE's diverseCounter-
    factuals results, stashed there by _hice_multi), coverage uses the
    EXACT target_node identity (metrics.tree_coverage_from_targets) rather
    than geometric routing -- a specialized HiCE CF routinely lands exactly
    ON its sibling-preference constraint's boundary (the proximity-
    minimizing objective's optimum sits on the newly-active constraint),
    which is a genuine, unresolvable tie for geometric nearest-side routing
    but not for this exact check, which already knows the intended side.
    Methods without that metadata (DiCE, BayCon) fall back to
    tree_coverage's Ward-weighted geometric routing.

    Every invalid CF is additionally checked against
    metrics.is_degenerate_tie: a margin of violation small enough that it's
    plausibly the solver converging to an unattainable strict-domination
    boundary (e.g. a duplicate-of-B competitor) rather than a real mistake
    -- see that function and the hiceSparsityCoverage breast-w
    investigation it came from. The count is surfaced (n_degenerate_ties)
    rather than silently folded into "invalid", so a run can be audited for
    how much of its shortfall from 100% validity is this known, largely
    harmless edge case vs. something else; a nonzero count is also printed
    live so it's visible during a run, not just after the fact.
    """
    per_cf = []
    valid_ts = []
    valid_centroids = []
    valid_target_nodes = []
    has_target_info = False
    n_degenerate_ties = 0
    for A_prime, info in cf_results:
        info = info if isinstance(info, dict) else {}
        if 'target_node' in info:
            has_target_info = True
        ok, margin = validity(A_prime, B_pts, comp_pts, LINKAGE, metric,
                               eps=_reval_eps(metric), A_orig=A_pts,
                               return_margin=True)
        if not ok and is_degenerate_tie(margin):
            n_degenerate_ties += 1
            print(f"[warn] degenerate tie: invalid CF for target_node="
                  f"{info.get('target_node')} with margin={margin:.3e} "
                  f"(solver feasible={info.get('feasible')}) -- see "
                  f"metrics.is_degenerate_tie", flush=True)
        prox = proximity(info, A_pts, A_prime, metric)
        # n_feat alongside l2: under hamming l2 counts changed CELLS (up to
        # |A|*d) while n_feat counts the FEATURES touched (up to d), and the
        # two are not inter-convertible -- cells/n_feat is the mean number of
        # A's points changed per touched feature, measured between 1 and 35
        # across pairs. The paper normalises by the feature count, so the
        # figure needs n_feat recorded per CF, not derived after the fact.
        per_cf.append((bool(ok), prox['l2'], prox['n_feat']))
        if ok:
            t = info.get('t')
            if t is None or np.ndim(t) == 0:
                t = np.asarray(A_prime).mean(0) - np.asarray(A_pts).mean(0)
            valid_ts.append(np.asarray(t, dtype=float).ravel())
            valid_centroids.append(np.asarray(A_prime, dtype=float).mean(0))
            if 'target_node' in info:
                valid_target_nodes.append(info['target_node'])

    n_returned = len(cf_results)
    n_valid = len(valid_ts)

    if has_target_info:
        n_repr, n_sub = tree_coverage_from_targets(Z, n_leaves, B_node, valid_target_nodes)
    else:
        n_repr, n_sub = tree_coverage(Z, n_leaves, B_node, X, valid_centroids, metric,
                                       n_A=A_pts.shape[0], linkage=LINKAGE)
    tc = (n_repr / n_sub) if n_sub > 0 else np.nan
    tc_norm = (n_repr / n_valid) if n_valid > 0 else np.nan

    if n_valid == 0:
        return dict(n_returned=n_returned, n_valid=0,
                    n_degenerate_ties=n_degenerate_ties,
                    avg_proximity=np.nan, avg_pairwise_distance=np.nan,
                    n_subbranches=n_sub, n_subbranches_represented=n_repr,
                    tree_coverage=tc, tree_coverage_norm=tc_norm, per_cf=per_cf)

    avg_proximity = float(np.mean([np.linalg.norm(t) for t in valid_ts]))
    if n_valid >= 2:
        pairwise = [np.linalg.norm(valid_ts[i] - valid_ts[j])
                    for i in range(n_valid) for j in range(i + 1, n_valid)]
        avg_pairwise_distance = float(np.mean(pairwise))
    else:
        avg_pairwise_distance = np.nan
    return dict(n_returned=n_returned, n_valid=n_valid,
                n_degenerate_ties=n_degenerate_ties,
                avg_proximity=avg_proximity,
                avg_pairwise_distance=avg_pairwise_distance,
                n_subbranches=n_sub, n_subbranches_represented=n_repr,
                tree_coverage=tc, tree_coverage_norm=tc_norm, per_cf=per_cf)


# ======================================================================
# Main loop
# ======================================================================
def run(dataset, out_csv, k, min_leaf, max_leaf, max_pairs,
        seed=0, eps=0.0, timeout=300, dice_timeout=None, only_pairs=None):
    dataset_label = dataset
    # DiCE gets its own, usually shorter, wall-clock budget: its genetic
    # search cannot self-terminate, and on this experiment's pairs it hits
    # the limit on ~55% of calls, every one of them returning ZERO valid
    # counterfactuals after burning the full budget -- i.e. that time is
    # pure waste, and it dominates total run time (83% of wall-clock).
    # HiCE keeps the longer `timeout` because ITS slow calls do return
    # usable results (its large-B pairs are exactly the low-tree-coverage
    # ones), so shortening HiCE's budget would drop those pairs and
    # artificially inflate its mean tree coverage. None = same as timeout.
    if dice_timeout is None:
        dice_timeout = timeout

    X, metric, types = load_openml_cached(dataset, LINKAGE)
    print(f"[data] OpenML '{dataset}': {X.shape} (metric={metric})")

    Z = (scipy_linkage(pdist(X, metric='hamming'), method=LINKAGE)
         if metric == 'hamming' else scipy_linkage(X, method=LINKAGE))
    print(f"[tree] linkage={LINKAGE}, {len(X)} leaves")

    cf = hiceCounterfactuals(X=X, Z=Z, metric=metric, linkage=LINKAGE, anchor='min')

    which_list = ['method', 'dice', 'baycon_random', 'baycon_proximal']
    # DiCE and BayCon search a rigid translation, which is the right move
    # space under Euclidean complete linkage but not under Hamming, where the
    # move is a per-coordinate relabeling. They are dropped there rather than
    # producing numbers that are not comparable -- same decision as
    # runCompleteLinkage.EUCLIDEAN_ONLY.
    which_list = ([w for w in which_list if w == 'method']
                  if metric == 'hamming' else which_list)
    pairs = cf.enumerate_pairs(min_leaf, max_leaf, max_pairs, seed=seed)

    # --only-pairs: solve a SUBSET of the enumerated pairs, named by index.
    #
    # The enumeration is a deterministic function of (min_leaf, max_leaf,
    # max_pairs, seed), so pair_idx means the same thing in every run that
    # shares them, and a subset run is directly joinable onto a full one. This
    # exists to re-solve pairs whose rows are already on disk but miss a
    # column a later driver version records (the hamming raw file predates
    # n_feat), WITHOUT paying again for the pairs that only ever timed out:
    # restricting to the pairs that returned something valid turns a 1200-pair
    # re-run into a 352-pair one, and a timeout had no n_feat to recover
    # anyway. Pairs not listed are skipped before any solve.
    n_to_solve = len(pairs) if only_pairs is None else sum(
        1 for i in range(len(pairs)) if str(i) in only_pairs)
    if only_pairs is not None:
        print(f"[only-pairs] {n_to_solve} of {len(pairs)} enumerated pairs selected")
        if n_to_solve == 0:
            print(f"[skip] no selected pairs for '{dataset_label}'; nothing to do.")
            return

    # per-(pair, method) resumability: a run killed partway through (e.g. an
    # OOM kill) restarts from exactly where it left off rather than redoing
    # completed (pair, method) combinations or -- worse -- the whole dataset.
    done_pairs = _pairs_done_in_csv(out_csv, dataset_label, LINKAGE)
    if len(done_pairs) >= n_to_solve * len(which_list):
        print(f"[skip] all {len(pairs)} pairs x {which_list} already present for "
              f"'{dataset_label}'/{LINKAGE} in {out_csv}; nothing to do.")
        return
    if done_pairs:
        print(f"[resume] {len(done_pairs)} (pair, method) combinations already "
              f"present; continuing from there.")

    print(f"[pairs] testing {len(pairs)} (A,B) pairs; methods={which_list}; k={k}")

    raw_path = _derive_raw_path(out_csv)
    summary_fields = _prepare_out_csv(out_csv, DIVERSITY_SUMMARY_FIELDNAMES)
    raw_fields = _prepare_out_csv(raw_path, DIVERSITY_RAW_FIELDNAMES)
    summary_file = open(out_csv, 'a', newline='')
    raw_file = open(raw_path, 'a', newline='')
    summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields, extrasaction='ignore')
    raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fields, extrasaction='ignore')

    try:
        for idx, (A_lv, B_lv) in enumerate(tqdm(pairs)):
            if only_pairs is not None and str(idx) not in only_pairs:
                continue
            to_run = [w for w in which_list if (idx, w) not in done_pairs]
            if not to_run:
                continue

            A_pts = X[sorted(A_lv)]
            B_pts = X[sorted(B_lv)]
            comps_leaves, s_A, s_B, s_star = cf.competitor_leaf_sets(A_lv, B_lv)
            comp_pts = [X[sorted(c)] for c in comps_leaves]
            # Hamming codes are labels, not magnitudes, so a centroid distance
            # is meaningless; normalise by the feature count so
            # avg_proximity_norm is the FRACTION of cells changed -- the same
            # convention the hamming figures use.
            # metrics.proximity's hamming l2 is the number of changed CELLS,
            # summed over A's |A| points, so it ranges up to |A|*d -- NOT d.
            # Dividing by d alone gives "mean features changed per point",
            # which exceeds 1 and is not comparable across datasets; the
            # fraction of changed cells needs |A|*d.
            scale = (float(X.shape[1]) * float(A_pts.shape[0]) if metric == 'hamming'
                     else float(np.linalg.norm(B_pts.mean(0) - A_pts.mean(0))))
            B_node, _, _ = cf._resolve_branch(B_lv)

            if scale <= 1e-12:
                # A's centroid already coincides exactly with B's: Ward
                # distance-to-B is already at its global minimum (0), so no
                # translation can ever strictly dominate A's own unmoved
                # position (ties forever at 0, per metrics.validity's A_orig
                # handling) -- there is no valid counterfactual to find here
                # for ANY method, by construction, not a solver shortfall.
                # Also avoids avg_proximity_norm's divide-by-zero guard,
                # which would otherwise silently NaN this pair out of the
                # coverage curves while its raw n_valid could still look
                # nonzero (exactly what happened before this check existed).
                # Skip the pair entirely -- no rows written for it at all,
                # for any method, so it drops out of the shared pair-count
                # denominator too instead of lingering as a phantom miss.
                continue

            # method -> list of (which_for_seed, k-sized CF list) pairs to
            # process; baycon_random/baycon_proximal share one BayCon call.
            per_method_cfs = {}
            per_method_seed = {}
            per_method_seconds = {}
            per_method_error = {}
            per_method_returned = {}

            if 'method' in to_run:
                cseed = call_seed(dataset_label, LINKAGE, seed, idx, 'method')
                _seed_everything(cseed)
                err = ''; cfs = []; n_returned = 0
                t0 = time.perf_counter()
                try:
                    with time_limit(timeout):
                        cfs, n_returned = _hice_multi(cf, B_lv, A_lv, k)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                secs = time.perf_counter() - t0
                per_method_cfs['method'] = cfs
                per_method_seed['method'] = cseed
                per_method_seconds['method'] = secs
                per_method_error['method'] = err
                per_method_returned['method'] = n_returned

            if 'dice' in to_run:
                cseed = call_seed(dataset_label, LINKAGE, seed, idx, 'dice')
                _seed_everything(cseed)
                err = ''; cfs = []
                t0 = time.perf_counter()
                try:
                    with time_limit(dice_timeout):
                        cfs = run_dice_baseline_multi(
                            A_pts, B_pts, comp_pts, LINKAGE, X_all=X,
                            mu_A=A_pts.mean(0), k=k, eps=eps, seed=cseed)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                secs = time.perf_counter() - t0
                per_method_cfs['dice'] = cfs
                per_method_seed['dice'] = cseed
                per_method_seconds['dice'] = secs
                per_method_error['dice'] = err
                per_method_returned['dice'] = len(cfs)

            if 'baycon_random' in to_run or 'baycon_proximal' in to_run:
                cseed = call_seed(dataset_label, LINKAGE, seed, idx, 'baycon')
                _seed_everything(cseed)
                err = ''; both = {'random': [], 'proximal': []}
                t0 = time.perf_counter()
                try:
                    with time_limit(timeout):
                        both = run_baycon_baseline_multi(
                            A_pts, B_pts, comp_pts, LINKAGE, X_all=X,
                            mu_A=A_pts.mean(0), k=k, eps=eps, seed=cseed)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                secs = time.perf_counter() - t0
                for which, key in (('baycon_random', 'random'), ('baycon_proximal', 'proximal')):
                    if which in to_run:
                        per_method_cfs[which] = both[key]
                        per_method_seed[which] = cseed
                        per_method_seconds[which] = secs
                        per_method_error[which] = err
                        per_method_returned[which] = len(both[key])

            for which in to_run:
                cfs = per_method_cfs[which]
                err = per_method_error[which]
                secs = per_method_seconds[which]
                cseed = per_method_seed[which]
                stats = _cf_list_stats(cfs, A_pts, B_pts, comp_pts, metric,
                                        Z, cf.n, B_node, X) if not err else \
                    dict(n_returned=0, n_valid=0, n_degenerate_ties=0, avg_proximity=np.nan,
                         avg_pairwise_distance=np.nan,
                         n_subbranches=np.nan, n_subbranches_represented=np.nan,
                         tree_coverage=np.nan, tree_coverage_norm=np.nan, per_cf=[])

                prox_norm = stats['avg_proximity'] / scale if scale > 1e-12 else np.nan
                pair_norm = stats['avg_pairwise_distance'] / scale if scale > 1e-12 else np.nan

                summary_row = dict(
                    dataset=dataset_label, linkage=LINKAGE, metric=metric,
                    pair_idx=idx, which=which, base_seed=seed, call_seed=cseed,
                    n_A=len(A_lv), n_B=len(B_lv),
                    k_requested=k, n_returned=stats['n_returned'], n_valid=stats['n_valid'],
                    n_degenerate_ties=stats['n_degenerate_ties'],
                    scale=scale, avg_proximity=stats['avg_proximity'],
                    avg_pairwise_distance=stats['avg_pairwise_distance'],
                    avg_proximity_norm=prox_norm, avg_pairwise_distance_norm=pair_norm,
                    n_subbranches=stats['n_subbranches'],
                    n_subbranches_represented=stats['n_subbranches_represented'],
                    tree_coverage=stats['tree_coverage'],
                    tree_coverage_norm=stats['tree_coverage_norm'],
                    seconds=secs, error=err,
                )
                for field in summary_fields:
                    summary_row.setdefault(field, '')
                summary_writer.writerow(summary_row)

                for cf_idx, (ok, l2, n_feat) in enumerate(stats['per_cf']):
                    raw_row = dict(
                        dataset=dataset_label, linkage=LINKAGE, metric=metric,
                        pair_idx=idx, which=which, base_seed=seed, call_seed=cseed,
                        n_A=len(A_lv), n_B=len(B_lv), cf_idx=cf_idx,
                        valid=int(ok), l2=l2, n_feat=n_feat, seconds=secs, error='',
                    )
                    for field in raw_fields:
                        raw_row.setdefault(field, '')
                    raw_writer.writerow(raw_row)

            summary_file.flush()
            raw_file.flush()

            if (idx + 1) % 25 == 0:
                print(f"  ... {idx+1}/{len(pairs)}")
    finally:
        summary_file.close()
        raw_file.close()

    print(f"[out] appended summary rows to {out_csv}, raw rows to {raw_path}")


def _derive_raw_path(out_csv):
    base, ext = os.path.splitext(out_csv)
    return f"{base}_raw{ext or '.csv'}"


def _pairs_done_in_csv(out_csv, dataset_label, linkage):
    """
    Set of (pair_idx, which) already present in the summary CSV for this
    (dataset, linkage) -- finer-grained than utilityFunctions._methods_in_csv
    (which only tracks whole methods), so a run interrupted partway through
    (e.g. killed for memory) can resume from the exact (pair, method) it left
    off at instead of restarting the whole method from pair 0. Requires the
    SAME --seed/--min-leaf/--max-leaf/--max-pairs as the original run, since
    those determine which pair_idx maps to which (A, B) pair.
    """
    done = set()
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return done
    with open(out_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('dataset') == dataset_label and row.get('linkage') == linkage:
                try:
                    done.add((int(row['pair_idx']), row.get('which')))
                except (KeyError, ValueError):
                    continue
    return done


# ======================================================================
# Plots: per-pair trajectories over the diversity summary CSV
# ======================================================================
def plot_coverage_curve(df, out_path, order, labels, colors, metric_col,
                         x_label, y_label, title):
    """
    Same coverage-curve construction as visualizations.py's
    plot_validity_proximity, applied to this experiment's per-pair summary
    df: for each method, sort its pairs by metric_col ascending (dropping
    pairs where metric_col is undefined -- avg_proximity_norm needs >=1
    valid CF for that pair, avg_pairwise_distance_norm needs >=2), then take
    the running mean of metric_col as a function of the fraction of ALL
    pairs in the dataset covered so far (shared denominator across methods
    = every distinct pair_idx). Rather than plotting that running mean at
    every pair, it's resampled onto a fixed 5%-spaced grid (5, 10, 15, ...)
    up to the method's own coverage rate, plus the exact coverage-rate
    point itself if it doesn't land on the grid -- markers at each of those
    points, straight lines between them. A method's curve stops at x =
    (pairs with a defined metric_col) / (all pairs) -- its coverage rate,
    shown in the legend -- so a method that fails to produce a usable value
    for many pairs (e.g. low-validity BayCon variants) visibly stops early.

    The universe of "all pairs" is distinct (dataset, pair_idx) pairs when
    `df` spans more than one dataset (e.g. a pooled cross-dataset call),
    since pair_idx is only unique WITHIN a dataset -- using plain
    pair_idx.nunique() there would undercount the universe and let
    coverage rates run past 100%.
    """
    if "dataset" in df.columns and df["dataset"].nunique() > 1:
        n_universe = len(df[["dataset", "pair_idx"]].drop_duplicates())
    else:
        n_universe = df["pair_idx"].nunique()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for which in order:
        sub = df[df["which"] == which]
        if len(sub) == 0 or n_universe == 0:
            continue
        vals = sub[metric_col].dropna().sort_values()
        n_valid = len(vals)
        if n_valid == 0:
            continue

        k = np.arange(1, n_valid + 1)
        running_mean = vals.to_numpy().cumsum() / k
        x_frac = k / n_universe * 100

        grid = np.arange(5, 105, 5, dtype=float)
        grid = grid[grid <= x_frac[-1]]
        if len(grid) == 0 or grid[-1] < x_frac[-1] - 1e-9:
            grid = np.append(grid, x_frac[-1])
        y_grid = np.interp(grid, x_frac, running_mean)

        color = colors.get(which, "#333333")
        ax.plot(
            grid, y_grid, color=color, linewidth=2, marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=1,
            label=f"{labels.get(which, which)} (coverage {n_valid/n_universe:.0%})",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, color="#0b0b0b", fontsize=11)
    ax.set_xlim(0, 100)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_metric_trajectory(df, out_path, order, labels, colors,
                            x_col, y_col, x_label, y_label, title):
    """
    One trajectory per method: each pair contributes a single point,
    (x_col, y_col) at that pair. Pairs are sorted by x_col ascending and
    connected in that order -- x_col and y_col are NOT sorted
    independently, each point stays a real pair drawn from a single
    pair_idx. A row with NaN in either column (e.g. avg_pairwise_distance
    is undefined when a method returned fewer than 2 valid CFs for that
    pair) is dropped. A method left with fewer than 2 usable pairs (e.g.
    low-validity BayCon variants) is still drawn as isolated marker(s)
    rather than skipped outright, so its (sparse) presence stays visible.
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    for which in order:
        sub = df[df["which"] == which]
        if len(sub) == 0:
            continue
        pts = sub[[x_col, y_col]].dropna()
        if len(pts) == 0:
            continue
        pts = pts.sort_values(x_col)
        color = colors.get(which, "#333333")
        ax.plot(
            pts[x_col], pts[y_col],
            marker='o', markersize=5, linewidth=1.5,
            color=color, label=f"{labels.get(which, which)} (n={len(pts)})",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, color="#0b0b0b", fontsize=11)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


PROX_LABEL = "Normalized proximity  (||t|| / ||mu_B - mu_A||), sorted ascending"
DIV_LABEL = "Normalized pairwise diversity  (avg ||t_i - t_j|| / ||mu_B - mu_A||)"


def plot_proximity_diversity(df, out_path, order, labels, colors,
                              title="Proximity vs. diversity trade-off"):
    plot_metric_trajectory(
        df, out_path, order, labels, colors,
        x_col="avg_proximity_norm", y_col="avg_pairwise_distance_norm",
        x_label=PROX_LABEL, y_label=DIV_LABEL, title=title)


def plot_diversity_validity_proximity(df, out_path, order, labels, colors,
                                       title="Validity vs. proximity trade-off"):
    plot_coverage_curve(
        df, out_path, order, labels, colors, metric_col="avg_proximity_norm",
        x_label="Most-proximal fraction of pairs (%)",
        y_label="Mean normalized proximity of the top-x% pairs",
        title=title)


def plot_diversity_validity_diversity(df, out_path, order, labels, colors,
                                       title="Validity vs. diversity trade-off"):
    plot_coverage_curve(
        df, out_path, order, labels, colors, metric_col="avg_pairwise_distance_norm",
        x_label="Most-diverse fraction of pairs (%)",
        y_label="Mean normalized pairwise diversity of the top-x% pairs",
        title=title)


def plot_tree_coverage(df, out_path, order, labels, colors,
                        metric_col="tree_coverage",
                        title="Tree coverage by method"):
    """
    Grouped bar chart, two bars per method, +/- 1 std across pairs as error
    bars:

      raw      : tree_coverage itself, n_subbranches_represented / n_sub --
                 fraction of ALL of B's sub-branches represented.
      achieved : n_subbranches_represented / min(n_sub, k_requested) --
                 fraction of the BUDGET-ACHIEVABLE sub-branches represented.

    A pair only ever returns up to k_requested counterfactuals, so when a
    branch B has more than k_requested sub-branches, 100% raw coverage is
    mathematically unreachable no matter how good the method is -- at most
    k_requested of them can ever be represented. The "achieved" bar divides
    by that reachable ceiling instead, so it isolates how well a method
    used its budget from how big B's subtree happened to be; the gap
    between the two bars for a method is exactly how much of its shortfall
    from 100% raw coverage is a budget ceiling rather than a method
    failure. The xtick label also reports what fraction of that method's
    pairs actually hit the ceiling (n_sub > k_requested).

    Unlike the proximity/diversity coverage curves, tree coverage isn't
    ranked by "closeness" -- it's a per-pair fraction in [0, 1] regardless
    of how proximal the CFs that earned it are -- so it's shown on its own
    rather than against a most-proximal/most-diverse x-axis. A pair where B
    is a single leaf (no sub-branches to cover) has undefined (NaN)
    coverage and is dropped, same as a method that returned no valid CFs.
    """
    methods = [m for m in order if m in df["which"].unique()]
    has_budget_cols = {"n_subbranches", "n_subbranches_represented", "k_requested"} <= set(df.columns)

    bar_labels, raw_means, raw_stds, ach_means, ach_stds, bar_colors, pct_capped = \
        [], [], [], [], [], [], []
    for m in methods:
        sub = df[df["which"] == m]
        raw_vals = sub[metric_col].dropna()
        if len(raw_vals) == 0:
            continue
        bar_labels.append(labels.get(m, m))
        raw_means.append(raw_vals.mean() * 100)
        raw_stds.append(raw_vals.std() * 100 if len(raw_vals) > 1 else 0)
        bar_colors.append(colors.get(m, "#4a3aa7"))

        if has_budget_cols:
            b = sub[["n_subbranches", "n_subbranches_represented", "k_requested"]].dropna()
            achievable = np.minimum(b["n_subbranches"], b["k_requested"])
            ach_vals = (b["n_subbranches_represented"] / achievable).replace(
                [np.inf, -np.inf], np.nan).dropna()
            ach_means.append(ach_vals.mean() * 100 if len(ach_vals) else np.nan)
            ach_stds.append(ach_vals.std() * 100 if len(ach_vals) > 1 else 0)
            pct_capped.append((b["n_subbranches"] > b["k_requested"]).mean()
                               if len(b) else np.nan)
        else:
            ach_means.append(np.nan)
            ach_stds.append(0)
            pct_capped.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    x = np.arange(len(bar_labels))
    width = 0.35
    ax.bar(x - width / 2, raw_means, width, yerr=raw_stds, color=bar_colors,
           alpha=0.55, capsize=3, error_kw=dict(ecolor="#52514e", linewidth=1.0),
           label="raw (% of all sub-branches)")
    ax.bar(x + width / 2, ach_means, width, yerr=ach_stds, color=bar_colors,
           capsize=3, error_kw=dict(ecolor="#52514e", linewidth=1.0),
           label="achieved (% of budget-reachable sub-branches)")

    xticklabels = [
        f"{lbl}\n({pc:.0%} budget-capped)" if np.isfinite(pc) else lbl
        for lbl, pc in zip(bar_labels, pct_capped)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Mean tree coverage (%)")
    ax.set_title(title, color="#0b0b0b", fontsize=13, pad=12)
    ax.set_ylim(bottom=0)
    _style_axes(ax)
    ax.xaxis.grid(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, nargs='+', default=['iris'],
                     help="one or more OpenML dataset names; Ward+Euclidean only")
    ap.add_argument('--out', type=str, default='data/completeDiversityEuclidean.csv')
    ap.add_argument('--out-dir', type=str, default='data/images')
    ap.add_argument('--k', type=int, default=10, help="counterfactuals requested per (pair, method)")
    ap.add_argument('--min-leaf', type=int, default=1)
    ap.add_argument('--max-leaf', type=int, default=15)
    ap.add_argument('--max-pairs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eps', type=float, default=0.0)
    ap.add_argument('--timeout', type=float, default=1200,
                     help="per-call wall-clock limit in seconds (0/negative disables)")
    ap.add_argument('--dice-timeout', type=float, default=None,
                     help="per-call wall-clock limit for DiCE only, in seconds; "
                          "defaults to --timeout. See run()'s note on why DiCE "
                          "warrants a shorter budget than the other methods.")
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--only-pairs', type=str, default=None,
                     help="a previous summary CSV; solve only the (dataset, "
                          "pair_idx) combinations it records as having "
                          "returned at least one valid counterfactual for "
                          "--only-pairs-which. Pair indices are comparable "
                          "across runs only when --min-leaf/--max-leaf/"
                          "--max-pairs/--seed match the reference run.")
    ap.add_argument('--only-pairs-which', type=str, default='method',
                     help="which method's validity selects the pairs for "
                          "--only-pairs (default: method)")
    a = ap.parse_args()

    # Read the selection once, not per dataset: the reference file is the full
    # run's summary and is re-read for nothing otherwise.
    only_pairs = None
    if a.only_pairs:
        only_pairs = {}
        n_sel = 0
        with open(a.only_pairs, newline='') as fh:
            for r in csv.DictReader(fh):
                if r.get('which') != a.only_pairs_which:
                    continue
                try:
                    nv = float(r.get('n_valid') or 'nan')
                except ValueError:
                    continue
                # A pair that returned nothing valid has no per-CF row to
                # complete, so re-solving it would buy nothing -- and those
                # are exactly the pairs that cost the full timeout.
                if not (nv > 0):
                    continue
                only_pairs.setdefault(r['dataset'], set()).add(r['pair_idx'])
                n_sel += 1
        print(f"[only-pairs] {a.only_pairs}: {n_sel} pairs across "
              f"{len(only_pairs)} datasets (which={a.only_pairs_which})")

    timeout = a.timeout if a.timeout and a.timeout > 0 else None
    dice_timeout = (a.dice_timeout if a.dice_timeout and a.dice_timeout > 0
                    else (None if a.dice_timeout is not None else timeout))

    for dataset in a.dataset:
        print(f"\n===== dataset: {dataset} =====")
        if only_pairs is not None and dataset not in only_pairs:
            print(f"[skip] '{dataset}' has no selected pairs in {a.only_pairs}")
            continue
        try:
            run(dataset, a.out, a.k, a.min_leaf, a.max_leaf, a.max_pairs,
                seed=a.seed, eps=a.eps, timeout=timeout,
                dice_timeout=dice_timeout,
                only_pairs=None if only_pairs is None else only_pairs[dataset])
        except ValueError as e:
            print(f"[skip] '{dataset}': {e}")
            continue
        except Exception as e:
            # One dataset failing (a dead OpenML, a blocked route) must not
            # end a multi-day run. This gap is why the Ward run was set to
            # die at its final dataset once openml.org became unreachable:
            # OpenMLServerError is not a ValueError, so it propagated.
            print(f"[error] '{dataset}': {type(e).__name__}: {e} "
                  f"-- continuing with the next dataset", flush=True)
            continue

    if a.no_plot:
        return

    df = pd.read_csv(a.out)
    os.makedirs(a.out_dir, exist_ok=True)
    for dataset in df["dataset"].dropna().unique():
        sub = df[df["dataset"] == dataset]
        order, labels, colors = _build_style(sub["which"].unique())
        plot_proximity_diversity(
            sub, os.path.join(a.out_dir, f"{dataset}_proximity_diversity.png"),
            order, labels, colors,
            title=f"Proximity vs. diversity trade-off -- {dataset}")
        plot_diversity_validity_proximity(
            sub, os.path.join(a.out_dir, f"{dataset}_diversity_validity_proximity.png"),
            order, labels, colors,
            title=f"Validity vs. proximity trade-off -- {dataset}")
        plot_diversity_validity_diversity(
            sub, os.path.join(a.out_dir, f"{dataset}_diversity_validity_diversity.png"),
            order, labels, colors,
            title=f"Validity vs. diversity trade-off -- {dataset}")


if __name__ == '__main__':
    main()
