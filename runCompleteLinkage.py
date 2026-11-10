"""
Complete-linkage single-counterfactual comparison.

For each of --max-pairs randomly-chosen (A, B) branch pairs, produce ONE
counterfactual per method and score it:

  - method    : the exact complete-linkage solver. metric='euclidean' goes to
                counterfactual_complete_linkage_branch_gurobi (convex MIQP
                with constraint row generation, certified globally optimal
                when info['optimal']); metric='hamming' goes to the exact
                0-1 MILP _solve_branch_hamming (rigid relabeling move).
  - baseline  : baselineCompleteLinkage -- the min-max/MEB translation, which
                minimises A's FARTHEST distance to B while IGNORING
                competitors. Its achieved d(A,B) is the feasibility floor, so
                it is both a baseline and a diagnostic: when even the
                baseline cannot dominate, the pair's region is plausibly
                empty rather than the solver having failed.
  - dice      : DiCE genetic search, seeded with that same MEB translation.
  - baycon    : BayCon search, same seed, same single call.

Why this script exists
----------------------
runHiceSparsity.py cannot do this job: it is a REPLAY driver that
reconstructs pairs from an existing results CSV (--in-csv) in order to sweep
sparsity modes, so it presupposes a run that already happened. The original
single-CF comparison (which produced data/allCompetitors.csv) came from an
older runHierarchical.py carrying --linkage/--no-hice flags; today's
runHierarchical.py is the k-counterfactual diversity experiment and no
longer has them. Hence a dedicated driver.

Linkage/metric
--------------
--linkage defaults to 'complete'. The metric is resolved from the data by
load_openml (all-continuous -> euclidean, all-categorical -> hamming; mixed
is rejected), so complete linkage covers BOTH metrics here, unlike Ward
which admits euclidean only. Z is built consistently with the resolved
metric (hamming data is linked on its Hamming pdist, not on raw codes).
'ward' is accepted too, purely so the same harness can regenerate a Ward
single-CF baseline for cross-checking.

Timeouts
--------
--timeout bounds every method call; --dice-timeout overrides it for DiCE
only, which matters because DiCE cannot self-terminate: its genetic search
burns the whole budget and returns nothing, and the MethodTimeout raised by
time_limit's SIGALRM is SWALLOWED inside dice_ml rather than propagating.
A DiCE row that timed out therefore has error='' and looks like an ordinary
empty result -- identify those by `seconds` reaching the limit, never by the
error column. (Observed under Ward on 55% of DiCE calls, and reproduced
under complete linkage.)

Output is append-only and resumable per (dataset, linkage, pair_idx, which),
so an interrupted run continues from exactly where it stopped. Resuming
REQUIRES the same --seed/--min-leaf/--max-leaf/--max-pairs, since those
determine which pair_idx maps to which (A, B) pair.

Usage
-----
    python runCompleteLinkage.py --dataset iris --out data/complete_iris.csv
    python runCompleteLinkage.py --dataset iris hayes-roth --max-pairs 50 \
        --timeout 1200 --dice-timeout 600
"""

import argparse
import csv
import multiprocessing as mp
import os
import time

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist
from tqdm import tqdm

from modules.constants import COMPLETE_FIELDNAMES
from modules.hiceCounterfactuals import hiceCounterfactuals
from modules.utilityFunctions import (
    MethodTimeout, _prepare_out_csv, time_limit, call_seed, _seed_everything,
    load_openml_cached, complete_linkage_meb_seed,
)
from modules.metrics import validity, proximity
from modules.dice import run_dice_baseline_multi
from modules.baycon import run_baycon_baseline_multi

WHICH_LIST = ['method', 'baseline', 'dice', 'baycon']

# DiCE and BayCon are run for EUCLIDEAN complete linkage only.
#
# Both search over a rigid translation t, and under Euclidean complete
# linkage that is the right move space: d(A(t), S) = max_{a,s} ||a + t - s||
# is fully determined by t, so optimising a vector is exactly the problem
# (verified: they return independently-revalidated counterfactuals there).
#
# Under HAMMING it is not. The admissible move is a per-coordinate
# RELABELING f_j applied to all of A, which is not a point in feature space,
# so there is nothing for a point-optimising search to translate. Worse, our
# ConstraintClassifier would score such a search with Euclidean geometry
# regardless: distLinkage hardcodes metric_='euclidean' for 'complete'. Left
# enabled, they return fractional "codes" that are not valid categorical
# assignments at all, yet can pass a numeric domination check -- on
# hayes-roth that produced valid=True rows with EVERY cell changed, which
# would be a silently wrong baseline rather than a missing one.
#
# Both libraries do support categorical features natively (dice_ml via
# Data(continuous_features=...), BayCon via DataAnalyzer(cat_features=...));
# our wrappers simply declare every column continuous. Making them genuine
# Hamming baselines therefore means changing the SEARCH VARIABLE to the
# relabeling itself (one categorical variable per (coordinate, source value)
# with domain = that coordinate's A u B values) and giving
# ConstraintClassifier a Hamming branch -- deliberately out of scope here,
# so these rows are recorded as not-applicable instead of fabricated.
EUCLIDEAN_ONLY = {'dice', 'baycon'}
NA_HAMMING = ("NotApplicable: searches a rigid translation, which is not the "
              "Hamming relabeling move space -- see runCompleteLinkage."
              "EUCLIDEAN_ONLY")

# Revalidation tolerance, same rationale as runHierarchical.REVALIDATION_TOL:
# passed as a NEGATIVE eps so validity's `dB < dX - eps` test relaxes to
# `dB < dX + tol`, absorbing solver rounding at ~1e-11 without weakening the
# definition. It matters more here than under Ward, because this solver's own
# eps is a margin on SQUARED distances while validity compares UNSQUARED ones.
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
    Revalidation eps, as a NEGATIVE value to relax validity's `dB < dX - eps`.

    Euclidean: -REVALIDATION_TOL, absorbing solver rounding noise.

    Hamming: EXACTLY 0. Hamming distances are integers, so there is no
    rounding noise to absorb -- and applying the Euclidean tolerance there
    silently turns strict domination into NON-strict, because `dB < dX + 1e-9`
    is just `dB <= dX` on integers. Ties are common on small alphabets
    (observed on hayes-roth: d_H(A',B) equal to the nearest competitor on
    every pair tested, margin exactly 0), so the tolerance would mark a tie
    as valid and systematically overstate validity for every method on
    categorical data -- while the exact solver, which correctly requires
    STRICT domination, reported those same pairs infeasible. That mismatch
    looked like a solver bug and was not.
    """
    return 0.0 if metric == 'hamming' else -REVALIDATION_TOL


def _build_tree(X, metric, linkage):
    """Linkage matrix consistent with the resolved ground metric."""
    if metric == 'hamming':
        return scipy_linkage(pdist(X, metric='hamming'), method=linkage)
    return scipy_linkage(X, method=linkage)


def _worker(q, args):
    """Child-process entry point: run one method call and post the result back."""
    try:
        q.put(("ok", _run_one(*args)))
    except Exception as e:                       # noqa: BLE001 - reported, not raised
        q.put(("err", f"{type(e).__name__}: {e}"))


def _run_one_bounded(args, budget):
    """
    Run one method call with a HARD wall-clock wall, by running it in a child
    process and killing the child if it overruns.

    An in-process timer cannot do this. utilityFunctions.time_limit raises from
    a SIGALRM handler, and Python only runs signal handlers between bytecodes --
    but scipy.optimize.milp (HiGHS) and Gurobi are C extensions that hold the
    GIL for the whole solve, so the alarm is queued and never delivered until
    the solve returns on its own. Passing a limit into the solver helps but is
    not sufficient either: the model BUILD is unbounded Python work, and on
    splice (3005 x 60, ~30k witnesses) a nominal 60s budget was still running
    after 5 minutes. Before any of this, one splice pair ran for TWO DAYS at
    100% CPU under a nominal 1200s limit.

    Killing a child process is the only mechanism that bounds arbitrary code.
    Cost: the arguments are pickled across. That is cheap here (the largest is
    the dataset matrix) relative to a budget measured in minutes. Children run
    sequentially, so at most one holds a Gurobi WLS session at a time, which
    matters because the academic licence allows only two.

    Returns (A_prime, info) on success, or raises MethodTimeout / the child's
    exception so the caller records it exactly as before.
    """
    if budget is None:
        return _run_one(*args)
    ctx = mp.get_context("fork")                 # fork: no re-import, shares the loaded data
    q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(q, args), daemon=True)
    proc.start()
    proc.join(budget)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        raise MethodTimeout(f"exceeded {budget}s wall-clock limit (hard-killed)")
    try:
        status, payload = q.get_nowait()
    except Exception:
        raise MethodTimeout(
            f"child exited without a result (exitcode={proc.exitcode}) "
            f"within {budget}s")
    if status == "err":
        raise RuntimeError(payload)
    return payload


def _pairs_done(out_csv, dataset_label, linkage):
    """Set of (pair_idx, which) already recorded for this (dataset, linkage)."""
    done = set()
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return done
    with open(out_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('dataset') == dataset_label and row.get('linkage') == linkage:
                try:
                    done.add((int(row['pair_idx']), row.get('which')))
                except (KeyError, ValueError):
                    continue
    return done


def _run_one(which, cf, A_lv, B_lv, A_pts, B_pts, comp_pts, X, metric,
             linkage, eps, cseed, t_seed, budget=None):
    """One method, one pair -> (A_prime, info). Raises on solver failure."""
    if which == 'method':
        if metric == 'hamming':
            # `budget` is passed INTO the solver because the time_limit()
            # wrapper around this call cannot interrupt HiGHS (a C extension
            # holding the GIL). Without it a single pair ran for two days on
            # splice under a nominal 1200s limit.
            return cf.counterfactual_complete_linkage_branch(
                B_lv, A_lv, metric='hamming', eps=eps, time_limit=budget)
        return cf.counterfactual_complete_linkage_branch(
            B_lv, A_lv, metric='euclidean', eps=eps, time_limit=budget or 60)

    if which == 'baseline':
        return cf.baselineCompleteLinkage(B_lv, A_lv)

    if which == 'dice':
        out = run_dice_baseline_multi(A_pts, B_pts, comp_pts, linkage, X_all=X,
                                       mu_A=A_pts.mean(0), k=1, eps=eps, seed=cseed)
        return (out[0] if out else (None, {'feasible': False}))

    if which == 'baycon':
        out = run_baycon_baseline_multi(A_pts, B_pts, comp_pts, linkage, X_all=X,
                                         mu_A=A_pts.mean(0), k=1, eps=eps, seed=cseed)
        pool = out['proximal'] if isinstance(out, dict) else out
        return (pool[0] if pool else (None, {'feasible': False,
                                             'one_class_pool':
                                                 out.get('one_class_pool', False)
                                                 if isinstance(out, dict) else False}))
    raise ValueError(f"unknown method {which!r}")


def run(dataset, out_csv, linkage, min_leaf, max_leaf, max_pairs,
        seed=0, eps=1e-6, timeout=1200, dice_timeout=None):
    if dice_timeout is None:
        dice_timeout = timeout

    X, metric, _types = load_openml_cached(dataset, linkage)
    print(f"[data] '{dataset}': {X.shape} (metric={metric}, linkage={linkage})")
    Z = _build_tree(X, metric, linkage)
    cf = hiceCounterfactuals(X=X, Z=Z, metric=metric, linkage=linkage, anchor='min')

    pairs = cf.enumerate_pairs(min_leaf, max_leaf, max_pairs, seed=seed)
    done = _pairs_done(out_csv, dataset, linkage)
    if len(done) >= len(pairs) * len(WHICH_LIST):
        print(f"[skip] all {len(pairs)} pairs x {WHICH_LIST} already present "
              f"for '{dataset}'/{linkage}; nothing to do.")
        return
    if done:
        print(f"[resume] {len(done)} (pair, method) combinations already present.")
    print(f"[pairs] {len(pairs)} (A,B) pairs; methods={WHICH_LIST}")

    fields = _prepare_out_csv(out_csv, COMPLETE_FIELDNAMES)
    fh = open(out_csv, 'a', newline='')
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
    try:
        for idx, (A_lv, B_lv) in enumerate(tqdm(pairs)):
            to_run = [m for m in WHICH_LIST if (idx, m) not in done]
            if not to_run:
                continue
            A_pts, B_pts = X[sorted(A_lv)], X[sorted(B_lv)]
            comps_leaves, s_A, s_B, s_star = cf.competitor_leaf_sets(A_lv, B_lv)
            comp_pts = [X[sorted(c)] for c in comps_leaves]

            # The MEB seed is shared by DiCE and BayCon; recording whether it
            # is itself valid separates "the baselines failed" from "this
            # pair's region is (near) empty, so they had nothing to find".
            if metric == 'euclidean':
                t_seed = complete_linkage_meb_seed(A_pts, B_pts)
                seed_valid = bool(validity(A_pts + t_seed, B_pts, comp_pts,
                                            linkage, metric, eps=_reval_eps(metric),
                                            A_orig=A_pts))
            else:
                t_seed, seed_valid = None, ''

            for which in to_run:
                cseed = call_seed(dataset, linkage, seed, idx, which)
                if metric == 'hamming' and which in EUCLIDEAN_ONLY:
                    # Recorded, not omitted: an explicit N/A row keeps the
                    # (pair, method) grid complete and resumable, and makes
                    # the exclusion visible in the CSV rather than looking
                    # like a method that silently found nothing.
                    w.writerow(dict(
                        dataset=dataset, linkage=linkage, metric=metric,
                        pair_idx=idx, which=which, base_seed=seed,
                        call_seed=cseed, n_A=len(A_lv), n_B=len(B_lv),
                        n_competitors=len(comp_pts), s_A=s_A, s_B=s_B,
                        s_star=s_star, valid=False, l2='', l2_sq='', l1='',
                        linf='', n_features_changed='', seconds=0.0,
                        aborted_timeout=False, error=NA_HAMMING,
                        seed_valid=seed_valid))
                    fh.flush()
                    continue
                _seed_everything(cseed)
                lim = dice_timeout if which == 'dice' else timeout
                err, A_prime, info = '', None, {}
                t0 = time.perf_counter()
                try:
                    # Hard wall via a child process. The in-process time_limit
                    # is kept as a cheap outer guard for the pickling/startup
                    # path, but _run_one_bounded is what actually bounds the
                    # solve -- see its docstring.
                    with time_limit(lim + 60 if lim else None):
                        A_prime, info = _run_one_bounded(
                            (which, cf, A_lv, B_lv, A_pts, B_pts, comp_pts,
                             X, metric, linkage, eps, cseed, t_seed, lim),
                            lim)
                except MethodTimeout as e:
                    err = f"MethodTimeout: {e}"
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                secs = time.perf_counter() - t0
                info = info if isinstance(info, dict) else {}

                if A_prime is not None:
                    ok, margin = validity(A_prime, B_pts, comp_pts, linkage, metric,
                                           eps=_reval_eps(metric), A_orig=A_pts,
                                           return_margin=True)
                else:
                    ok, margin = False, float('nan')
                prox = proximity(info, A_pts, A_prime, metric)

                w.writerow(dict(
                    dataset=dataset, linkage=linkage, metric=metric, pair_idx=idx,
                    which=which, base_seed=seed, call_seed=cseed,
                    n_A=len(A_lv), n_B=len(B_lv), n_competitors=len(comp_pts),
                    s_A=s_A, s_B=s_B, s_star=s_star,
                    valid=bool(ok), l2=prox['l2'], l2_sq=prox['l2_sq'],
                    l1=prox['l1'], linf=prox['linf'],
                    n_features_changed=prox['n_feat'],
                    sparsity=info.get('sparsity', ''),
                    solver_feasible=info.get('feasible', ''),
                    solver_optimal=info.get('optimal', ''),
                    ward_phase=info.get('phase', ''),
                    n_active_balls='',
                    used_fallback='', seconds=secs,
                    aborted_timeout=bool(err.startswith('MethodTimeout')),
                    error=err,
                    rowgen_rounds=info.get('rowgen_rounds', ''),
                    rowgen_rows=info.get('rowgen_rows', ''),
                    rowgen_converged=info.get('rowgen_converged', ''),
                    rowgen_stalled_blocks=info.get('rowgen_stalled_blocks', ''),
                    revalidated=info.get('revalidated', ''),
                    margin_unsquared=info.get('margin_unsquared', margin),
                    one_class_pool=info.get('one_class_pool', ''),
                    seed_valid=seed_valid,
                ))
                fh.flush()
    finally:
        fh.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=str, nargs='+', default=['iris'])
    ap.add_argument('--out', type=str, default='data/complete_linkage.csv')
    ap.add_argument('--linkage', type=str, default='complete',
                     choices=['complete', 'ward'])
    ap.add_argument('--min-leaf', type=int, default=1)
    ap.add_argument('--max-leaf', type=int, default=15)
    ap.add_argument('--max-pairs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eps', type=float, default=1e-6)
    ap.add_argument('--timeout', type=float, default=1200,
                     help="per-call wall-clock limit in seconds (0/negative disables)")
    ap.add_argument('--dice-timeout', type=float, default=None,
                     help="per-call limit for DiCE only; defaults to --timeout. "
                          "DiCE cannot self-terminate and swallows the timeout, "
                          "so its timed-out rows carry error='' -- see module docstring.")
    a = ap.parse_args()

    timeout = a.timeout if a.timeout and a.timeout > 0 else None
    dice_timeout = (a.dice_timeout if a.dice_timeout and a.dice_timeout > 0
                    else (None if a.dice_timeout is not None else timeout))

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    for dataset in a.dataset:
        print(f"\n===== dataset: {dataset} ({a.linkage}) =====")
        try:
            run(dataset, a.out, a.linkage, a.min_leaf, a.max_leaf, a.max_pairs,
                seed=a.seed, eps=a.eps, timeout=timeout, dice_timeout=dice_timeout)
        except ValueError as e:
            print(f"[skip] '{dataset}': {e}")
            continue
        except Exception as e:
            # Anything else (a dead OpenML, a solver blowing up on one
            # dataset) must not take down the whole multi-day run. The output
            # is resumable per (dataset, linkage, pair_idx, which), so a
            # dataset abandoned here is simply picked up by the next launch.
            print(f"[error] '{dataset}': {type(e).__name__}: {e} "
                  f"-- continuing with the next dataset", flush=True)
            continue


if __name__ == '__main__':
    main()
