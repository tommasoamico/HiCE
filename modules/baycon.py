"""
baycon.py: a BayCon-based competitor baseline for the branch counterfactual.

Idea
----
Same setup as the DiCE baseline (see dice.py): our method solves the
counterfactual exactly via geometry. Here we let BayCon (a Bayesian-
optimization-driven, model-agnostic counterfactual generator) find a valid
move by treating our validity test as a black-box binary classifier:

    f(t) = 1  iff  the rigidly-translated branch A + t dominates B against every
                   live competitor (a valid counterfactual under our definition)
    f(t) = 0  otherwise.

BayCon then searches for a translation t that flips the query point (t = 0,
"no move", class 0) to class 1 (valid), guided by a surrogate RandomForest
regressor trained on a fitness score that rewards proximity to the initial
instance and reaching the target class (see baycon/common/ScoreCalculator.py).

Key design choices (mirrors dice.py):

  * BayCon searches over the shared rigid translation t in R^d, not individual
    points, so the move stays rigid (A translated as a whole), matching our
    method. We reuse the exact same ConstraintClassifier as the black-box
    "model" BayCon queries.

  * BayCon's DataAnalyzer needs feature ranges/bounds and a labelled pool to
    bootstrap its neighbourhood sampling. We build that pool the same way as
    the DiCE baseline: the real dataset expressed as translations,
    t_i = x_i - mu_A (data-grounded ranges, on-manifold seeding), PLUS a
    guaranteed-valid seed (Ward: t_seed = mu_B - mu_A).

  * Like the DiCE baseline, we only support Ward linkage here: under Ward,
    landing A's centroid on B's centroid drives the Ward distance to 0, which
    is <= every competitor distance, so t_seed is ALWAYS feasible. Complete
    linkage has no such guaranteed-feasible translation, so it's not
    implemented here (raises NotImplementedError, matching dice.py).
"""

import random

import numpy as np
from joblib import parallel_backend

import baycon.baycon.bayesian_generator as baycon_gen
from baycon.common.DataAnalyzer import DataAnalyzer
from baycon.common.Target import Target
from modules.dice import ConstraintClassifier
from modules.utilityFunctions import complete_linkage_meb_seed


# ======================================================================
# The BayCon baseline runner
# ======================================================================
def run_baycon_baseline(A_pts, B_pts, comp_pts, linkage, X_all, mu_A=None,
                   eps=0.0, t_bound=5.0, n_train=400, seed=0):
    """
    Run BayCon against the ConstraintClassifier and return the best valid
    counterfactual translation it finds.

    Search space & training frame
    ------------------------------
    BayCon searches over the rigid translation t. Its DataAnalyzer is built
    from the real dataset expressed as translations, t_i = x_i - mu_A
    (data-grounded ranges, on-manifold seeding), plus a guaranteed-valid seed:

      * Ward: t_seed = mu_B - mu_A. Landing A's centroid on B's centroid drives
        the Ward distance to 0, which is <= every competitor distance, so
        t_seed is always feasible (Ward feasibility guarantee). This
        unconditionally supplies a class-1 example for BayCon's pool.
      * Complete linkage: no single translation is guaranteed feasible (the
        farthest-pair distance has a positive floor), so the mu_B seed does not
        transfer. Not implemented here yet, raises NotImplementedError.

    If, despite the seed, the pool is still one-class (shouldn't happen for
    Ward), we augment with a uniform box sample as a safety net.

    No fallback
    -----------
    If BayCon returns nothing valid, we do not
      fall back to t_seed (the
    guaranteed-valid centroid move for Ward); we report that no
    counterfactual was found: A_prime is None and info['found'] is False.

    Parameters
    ----------
    A_pts, B_pts : (n_A, d), (n_B, d).
    comp_pts : list of (n_X, d)   live competitor point sets.
    linkage : {'ward', 'complete'} (complete not yet supported here).
    X_all : (n, d)   the full dataset, used to build data-grounded t-ranges.
    mu_A : (d,), optional   A's centroid (defaults to A_pts.mean(0)).

    Returns
    -------
    A_prime : (n_A, d) or None if no valid counterfactual was found.
    info : dict  t, feasible (exact recheck), distance_moved, found,
                 n_cfs_returned, method.
    """

    # BayCon's exploration (InstancesGenerator.generate_random, etc.) draws
    # from the GLOBAL numpy/python random state (no local Generator to pass
    # in), so reseed both here -- this makes a standalone call with the same
    # `seed` and inputs fully reproducible, independent of whatever ran
    # before it.
    random.seed(seed)
    np.random.seed(seed)

    A_pts = np.asarray(A_pts, float)
    B_pts = np.asarray(B_pts, float)
    X_all = np.asarray(X_all, float)
    d = A_pts.shape[1]
    if mu_A is None:
        mu_A = A_pts.mean(axis=0)
    mu_B = B_pts.mean(axis=0)

    clf = ConstraintClassifier(A_pts, B_pts, comp_pts, linkage, eps=eps)

    # ---- guaranteed-valid seed (linkage-specific) ----
    if linkage == 'ward':
        t_seed = mu_B - mu_A                      # always Ward-feasible
    elif linkage == 'complete':
        # Complete linkage has no always-feasible analogue of Ward's mu_B seed,
        # so use the min-max/MEB translation: the t minimising A's FARTHEST
        # distance to B. That is the feasibility floor for d(A(t), B) -- the
        # best shot any translation has -- but it ignores competitors, so it
        # may itself be invalid. The search pool below is built from it either
        # way and the one-class safety net handles the case where nothing valid
        # turns up. See utilityFunctions.complete_linkage_meb_seed.
        t_seed = complete_linkage_meb_seed(A_pts, B_pts)
    else:
        raise ValueError(f"Unsupported linkage {linkage!r}.")

    # ---- training pool: dataset-as-translations + the valid seed + the query ----
    T_pool = np.vstack([X_all - mu_A, t_seed[None, :], np.zeros((1, d))])
    y_pool = clf.predict(T_pool)

    # safety net: if still one-class (shouldn't happen for Ward, since t_seed is
    # valid and t=0 is invalid), augment with a uniform box sample.
    # We need examples from the other calss for the surrogate
    if len(set(y_pool.tolist())) < 2:
        rng = np.random.default_rng(seed)
        T_box = rng.uniform(-t_bound, t_bound, size=(n_train, d))
        T_pool = np.vstack([T_pool, T_box])
        y_pool = clf.predict(T_pool)

    # If the pool is still single-class the search has nothing to learn from.
    # Under Ward this cannot happen (t_seed is valid, t=0 is not), but under
    # complete linkage the MEB seed carries no feasibility guarantee, so a pair
    # whose counterfactual region is genuinely empty lands here. Return "found
    # nothing" rather than letting DataAnalyzer/the RF fit raise on one class:
    # an empty result is the honest outcome and is what the callers already
    # handle, whereas an exception here would escape the generation try/except
    # below (DataAnalyzer is constructed outside it).
    if len(set(y_pool.tolist())) < 2:
        return None, {
            "t": np.zeros(d),
            "feasible": False,
            "distance_moved": float("nan"),
            "found": False,
            "n_cfs_returned": 0,
            "method": "baycon",
            "one_class_pool": True,
        }

    feature_names = np.array([f"t{j}" for j in range(d)])
    target = Target(target_type="classification", target_feature="valid", target_value=1)
    data_analyzer = DataAnalyzer(T_pool, y_pool, feature_names, target)

    # query = no move (t = 0, class 0); ask BayCon to reach class 1 (valid)
    initial_instance = np.zeros(d)
    initial_prediction = 0

    best_t, n_ret = None, 0
    try:
        counterfactuals, _ = baycon_gen.run(
            initial_instance, initial_prediction, target, data_analyzer, clf)
        n_ret = len(counterfactuals)
        if n_ret:
            # keep only the CFs that are ACTUALLY valid under our exact test,
            # then take the closest by our own L2.
            valid_cfs = [t for t in counterfactuals if clf.predict([t])[0] == 1]
            if valid_cfs:
                best_t = min(valid_cfs, key=lambda t: np.linalg.norm(t))
    except Exception:
        best_t = None

    # ---- no fallback: report failure to find a counterfactual as such ----
    if best_t is None:
        return None, {
            "t": None,
            "feasible": False,
            "distance_moved": None,
            "found": False,
            "n_cfs_returned": n_ret,
            "method": "baycon",
        }

    valid = bool(clf.predict([best_t])[0])          # exact recheck
    return A_pts + best_t, {
        "t": best_t,
        "feasible": valid,
        "distance_moved": float(np.linalg.norm(best_t)),
        "found": True,
        "n_cfs_returned": n_ret,
        "method": "baycon",
    }


def run_baycon_baseline_multi(A_pts, B_pts, comp_pts, linkage, X_all, mu_A=None,
                               k=10, eps=0.0, t_bound=5.0, n_train=400, seed=0):
    """
    Diversity variant of run_baycon_baseline: runs BayCon's search ONCE (same
    setup as run_baycon_baseline) and, unlike the single-CF version, does NOT
    collapse the accumulated candidate pool down to the single closest CF.
    Instead it exact-rechecks every candidate BayCon's search ever generated
    (`counterfactuals`, already >= MINIMUM_COUNTERFACTUALS_SIZE=10 when the
    search succeeds) and returns TWO independently-selected k-sized subsets
    of the valid ones, treated as two separate methods for comparison:

      - "random"   : k valid candidates chosen uniformly at random (seeded).
      - "proximal" : the k valid candidates with the smallest L2 norm
                     (BayCon's own closest-to-query candidates).

    Returns
    -------
    dict with keys "random" and "proximal", each a list of (A_prime, info)
    tuples of length <= k (0 if BayCon found nothing valid). info dicts have
    the same shape as run_baycon_baseline's: t, feasible (always True here),
    distance_moved, found (always True), n_cfs_returned (BayCon's raw
    candidate-pool size before the validity filter), method.
    """
    random.seed(seed)
    np.random.seed(seed)

    A_pts = np.asarray(A_pts, float)
    B_pts = np.asarray(B_pts, float)
    X_all = np.asarray(X_all, float)
    d = A_pts.shape[1]
    if mu_A is None:
        mu_A = A_pts.mean(axis=0)
    mu_B = B_pts.mean(axis=0)

    clf = ConstraintClassifier(A_pts, B_pts, comp_pts, linkage, eps=eps)

    if linkage == 'ward':
        t_seed = mu_B - mu_A                      # always Ward-feasible
    elif linkage == 'complete':
        # Complete linkage has no always-feasible analogue of Ward's mu_B seed,
        # so use the min-max/MEB translation: the t minimising A's FARTHEST
        # distance to B. That is the feasibility floor for d(A(t), B) -- the
        # best shot any translation has -- but it ignores competitors, so it
        # may itself be invalid. The search pool below is built from it either
        # way and the one-class safety net handles the case where nothing valid
        # turns up. See utilityFunctions.complete_linkage_meb_seed.
        t_seed = complete_linkage_meb_seed(A_pts, B_pts)
    else:
        raise ValueError(f"Unsupported linkage {linkage!r}.")

    T_pool = np.vstack([X_all - mu_A, t_seed[None, :], np.zeros((1, d))])
    y_pool = clf.predict(T_pool)

    if len(set(y_pool.tolist())) < 2:
        rng = np.random.default_rng(seed)
        T_box = rng.uniform(-t_bound, t_bound, size=(n_train, d))
        T_pool = np.vstack([T_pool, T_box])
        y_pool = clf.predict(T_pool)

    # If the pool is still single-class the search has nothing to learn from.
    # Under Ward this cannot happen (t_seed is valid, t=0 is not), but under
    # complete linkage the MEB seed carries no feasibility guarantee, so a pair
    # whose counterfactual region is genuinely empty lands here. Return "found
    # nothing" rather than letting DataAnalyzer/the RF fit raise on one class:
    # an empty result is the honest outcome and is what the callers already
    # handle, whereas an exception here would escape the generation try/except
    # below (DataAnalyzer is constructed outside it).
    if len(set(y_pool.tolist())) < 2:
        return {"random": [], "proximal": []}

    feature_names = np.array([f"t{j}" for j in range(d)])
    target = Target(target_type="classification", target_feature="valid", target_value=1)
    data_analyzer = DataAnalyzer(T_pool, y_pool, feature_names, target)

    initial_instance = np.zeros(d)
    initial_prediction = 0

    n_ret = 0
    valid_ts = []
    try:
        # BayCon's internal RandomForestRegressor(n_jobs=-1) otherwise spins
        # up a fresh loky process pool on every epoch's retrain; across
        # hundreds of (pair, method) calls in a batch run those pools
        # accumulate (observed leaked-semaphore warnings and OOM kills), so
        # force the threading backend here -- no new processes, at some cost
        # to RF-fit parallelism that doesn't matter for these small problems.
        with parallel_backend('threading', n_jobs=1):
            counterfactuals, _ = baycon_gen.run(
                initial_instance, initial_prediction, target, data_analyzer, clf)
        n_ret = len(counterfactuals)
        if n_ret:
            valid_ts = [t for t in counterfactuals if clf.predict([t])[0] == 1]
    except Exception:
        valid_ts = []

    def _to_results(ts):
        return [(A_pts + t, {
            "t": t,
            "feasible": True,
            "distance_moved": float(np.linalg.norm(t)),
            "found": True,
            "n_cfs_returned": n_ret,
            "method": "baycon",
        }) for t in ts]

    if not valid_ts:
        return {"random": [], "proximal": []}

    rng = np.random.default_rng(seed)
    random_idx = rng.choice(len(valid_ts), size=min(k, len(valid_ts)), replace=False)
    random_ts = [valid_ts[i] for i in random_idx]

    proximal_ts = sorted(valid_ts, key=lambda t: np.linalg.norm(t))[:k]

    return {"random": _to_results(random_ts), "proximal": _to_results(proximal_ts)}
