from itertools import product

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.optimize import minimize, milp, LinearConstraint, Bounds
from scipy.spatial.distance import cdist
from typing import Dict, Iterable
import gurobipy as gp
from gurobipy import GRB
import miniball
from scipy.optimize import minimize
import cvxpy as cp


def load_synthetic(n=60, d=4, k=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5, (k, d))
    X = np.vstack([c + rng.normal(0, 0.5, (n // k, d)) for c in centers])
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def detect_column_types(df, max_cardinality_ratio=0.05, max_unique_categorical=20):
    import pandas as pd
    n = len(df); types = {}
    for col in df.columns:
        s = df[col]
        if (not pd.api.types.is_numeric_dtype(s)) or pd.api.types.is_bool_dtype(s):
            types[col] = 'categorical'; continue
        nun = s.nunique(dropna=True)
        if pd.api.types.is_integer_dtype(s):
            types[col] = ('categorical'
                          if (nun <= max_unique_categorical or nun <= max_cardinality_ratio * n)
                          else 'continuous')
        else:
            types[col] = 'categorical' if nun <= 2 else 'continuous'
    return types












# In small dimesnion it would be more convenient computing the convex hull
# In high dimesnion, can use random projections acceoting a (1+epsilon) error or prune
# now we leave it like this, candidate for optimisation
def complete_linkage_max(P_pts, Q_pts, metric_):
        return cdist(P_pts, Q_pts, metric=metric_).max() 







def names_to_indices(names, index_to_name):
    """
    Convert a list of animal names to their corresponding indices.
    
    Parameters
    ----------
    names : list of str
        Animal names to look up.
    index_to_name : dict
        Mapping from index -> name (your existing dict).
    
    Returns
    -------
    list of int
    """
    name_to_index = {v: k for k, v in index_to_name.items()}
    
    indices = []
    for name in names:
        if name not in name_to_index:
            raise ValueError(f"'{name}' not found. Available names: {sorted(name_to_index.keys())}")
        indices.append(name_to_index[name])
    
    return indices



def round_to_binary(xp):
    return np.round(xp).astype(int).clip(0, 1)

def complete_linkage_dist_hamming(xp, points):
    """Max Hamming distance from xp to any point in points."""
    xp_bin = xp if xp.dtype == int else round_to_binary(xp)
    return cdist([xp_bin], points, metric='hamming').max()

def counterfactual_complete_linkage_discrete(x_idx, target_leaf_indices, X, Z, index_to_name, metric='hamming'):
    n = len(X)
    x = X[x_idx].astype(int)
    B_points = X[target_leaf_indices].astype(int)

    # --- Find B node and its parent (B') ---
    B_node = find_node_for_leaves(target_leaf_indices, Z, n)
    if B_node is None:
        raise ValueError("target_leaf_indices do not form a clean subtree in Z.")

    h = None
    for i in range(len(Z)):
        left, right = int(Z[i, 0]), int(Z[i, 1])
        if left == B_node or right == B_node:
            h = Z[i, 2]
            C_node = right if left == B_node else left
            break

    if h is None:
        raise ValueError("Could not find parent of B node — B may be the root.")

    C_leaves = get_leaves(C_node, Z, n)
    C_points = X[C_leaves].astype(int)

    # --- Find x's FIRST merge partner (the cluster it merges with at lowest height) ---
    first_merge_leaves = None
    first_merge_height = None
    for i in range(len(Z)):
        left, right = int(Z[i, 0]), int(Z[i, 1])
        if left == x_idx:
            first_merge_leaves = get_leaves(int(right), Z, n)
            first_merge_height = Z[i, 2]
            break
        elif right == x_idx:
            first_merge_leaves = get_leaves(int(left), Z, n)
            first_merge_height = Z[i, 2]
            break

    first_merge_points = X[first_merge_leaves].astype(int)

    d_x_B     = cdist([x], B_points,          metric=metric).max()
    d_x_C     = cdist([x], C_points,          metric=metric).max()
    d_x_first = cdist([x], first_merge_points, metric=metric).max()

    print(f"Original point           : {index_to_name[x_idx]} (index {x_idx})")
    print(f"First merge partner      : {[index_to_name[i] for i in first_merge_leaves]} at height {first_merge_height:.6f}")
    print(f"d_linkage(x, B)          : {d_x_B:.6f}  (h={h:.6f})")
    print(f"d_linkage(x, C)          : {d_x_C:.6f}")
    print(f"d_linkage(x, first)      : {d_x_first:.6f}  ← binding constraint")

    def is_feasible(xp):
        dB     = cdist([xp], B_points,           metric=metric).max()
        dC     = cdist([xp], C_points,           metric=metric).max()
        dfirst = cdist([xp], first_merge_points,  metric=metric).max()
        cond1 = dB <= h          # attaches inside B before B merges with C
        cond2 = dB < dC          # goes into B's side not C's
        cond3 = dB < dfirst      # merges with B before own original cluster
        return cond1 and cond2 and cond3

    # Greedy bit-flip
    xp = x.copy()
    n_features = len(x)
    flipped = []

    for step in range(n_features):
        if is_feasible(xp):
            break

        best_flip = None
        best_score = np.inf

        for j in range(n_features):
            if j in flipped:
                continue

            candidate = xp.copy()
            candidate[j] = 1 - candidate[j]

            dB     = cdist([candidate], B_points,          metric=metric).max()
            dC     = cdist([candidate], C_points,          metric=metric).max()
            dfirst = cdist([candidate], first_merge_points, metric=metric).max()

            # Score: minimize dB, bonus for increasing gap with C and first-merge partner
            score = dB - 0.01 * (dC - dB) - 0.01 * (dfirst - dB)

            if score < best_score:
                best_score = score
                best_flip = j

        if best_flip is None:
            break

        xp[best_flip] = 1 - xp[best_flip]
        flipped.append(best_flip)

    feasible   = is_feasible(xp)
    dB_final   = cdist([xp], B_points,          metric=metric).max()
    dC_final   = cdist([xp], C_points,          metric=metric).max()
    dfirst_fin = cdist([xp], first_merge_points, metric=metric).max()
    bits_moved = int(np.sum(xp != x))

    print(f"\nFeasible                 : {feasible}")
    print(f"Bits flipped             : {bits_moved}")
    print(f"Counterfactual d to B    : {dB_final:.6f}  (need <= {h:.6f})")
    print(f"Counterfactual d to C    : {dC_final:.6f}  (need > {dB_final:.6f})")
    print(f"Counterfactual d to first: {dfirst_fin:.6f}  (need > {dB_final:.6f})")

    # --- Report the counterfactual ---
    feature_names = list(X.columns) if hasattr(X, 'columns') else [f"feature_{i}" for i in range(n_features)]
    print(f"\nOriginal  ({index_to_name[x_idx]}): {dict(zip(feature_names, x.tolist()))}")
    print(f"Counterfactual            : {dict(zip(feature_names, xp.tolist()))}")
    print(f"\nChanges:")
    for j in flipped:
        print(f"  {feature_names[j]}: {x[j]} -> {xp[j]}")
    if not feasible:
        print("WARNING: could not find a feasible binary counterfactual within n_features flips.")

    info = {
        "h": h,
        "d_original_to_B": d_x_B,
        "d_original_to_C": d_x_C,
        "d_original_to_first_merge": d_x_first,
        "first_merge_partner": [index_to_name[i] for i in first_merge_leaves],
        "first_merge_height": first_merge_height,
        "feasible": feasible,
        "bits_flipped": bits_moved,
        "flipped_indices": flipped,
        "counterfactual_d_to_B": dB_final,
        "counterfactual_d_to_C": dC_final,
        "counterfactual_d_to_first": dfirst_fin,
    }

    return xp, info







 
# ---------------------------------------------------------------- helpers
def unit(d, rng):
    u = rng.standard_normal(d)
    return u / (np.linalg.norm(u) + 1e-12)
 
 
def feasible(p, convex_ball_in, convex_half, rev_balls, tol=1e-6):
    for (kappa, R) in convex_ball_in:
        if np.linalg.norm(p - kappa) > R + tol:
            return False
    for (a, b) in convex_half:
        if a @ p > b + tol:
            return False
    for rb in rev_balls:
        if np.linalg.norm(p - rb['kappa']) < rb['R'] - tol:
            return False
    return True
 
 
def active(p, rev_balls, index_to_name, tol=1e-4):
    nids = [rb['nid'] for rb in rev_balls
            if abs(np.linalg.norm(p - rb['kappa']) - rb['R']) < tol]
    if index_to_name is not None:
        return [index_to_name[i] for i in nids]
    return nids
 
 
 
def ward_info(t, feasible_, optimal_, n_comp, active_balls, phase=''):
    return {
        "t": t,
        "feasible": feasible_,
        "optimal": optimal_,
        "distance_moved": float(np.linalg.norm(t)),
        "n_competitors": n_comp,
        "active_balls": active_balls,   # already mapped by active(); store as-is
        "phase": phase,
    }



# we can reuse a lot of distances here instead of computing cdist every time, need to implement
# it across the whole pipeline
def _dominates(A_prime, B_pts, competitor_pts_list, linkage, eps):
    """
    Validity predicate: does the moved branch A_prime dominate B i.e. is it
    strictly closer to B than to every competitor under the given linkage?

      complete : d(A, S) = max_{a in A, s in S} ||a - s||   (farthest pair)
      ward     : d(A, S) = sqrt(2 nA nS / (nA + nS)) * ||mu_A - mu_S||  (centroids)

    Returns True iff d(A_prime, B) < d(A_prime, X) - eps for every competitor X.
    The full competitor set is re-checked every call, because a revert can
    change which competitor is nearest.
    """
    if linkage == 'complete':
        dB = cdist(A_prime, B_pts, metric='euclidean').max()
        for Xp in competitor_pts_list:
            dX = cdist(A_prime, Xp, metric='euclidean').max()
            if not (dB < dX - eps):
                return False
        return True

    elif linkage == 'ward':
        nA = len(A_prime)
        muA = A_prime.mean(axis=0)

        def wd(P):
            nS = len(P)
            c = np.sqrt(2.0 * nA * nS / (nA + nS))
            return c * np.linalg.norm(muA - P.mean(axis=0))

        dB = wd(B_pts)
        for Xp in competitor_pts_list:
            if not (dB < wd(Xp) - eps):
                return False
        return True

    else:
        raise ValueError(f"Unsupported linkage {linkage!r}; use 'complete' or 'ward'.")


def greedy_sparsify(A_orig, t, B_pts, competitor_pts_list, linkage,
                    eps=1e-9):
    """
    DiCE-style greedy sparsification of a rigid-translation counterfactual
    (Euclidean only, the Hamming/relabeling solver already minimizes the
    number of coordinates changed).

    Given the original branch A_orig (n_A x d) and a VALID displacement t (d,)
    for which A_orig + t dominates B, revert individual coordinates of t back
    to 0 (their original value) one at a time, in ascending order of |t_j|
    (most-revertible first), keeping each revert only if the counterfactual
    stays valid. The result is a sparser t (fewer changed features) that is
    still exactly valid.

    Properties
    ----------
    - Validity is exact: every kept revert is verified against ALL competitors
      (and, for Ward, the size-weighted centroid distances), so the returned
      counterfactual provably still dominates B.
    - Only moves toward the original, so displacement never increases and
      rigidity is preserved (t stays a single shared vector; reverting
      coordinate j just sets t_j = 0, applied uniformly to all of A).
    - Sparsity is greedy/locally-minimal (like DiCE): you cannot revert any
      further single feature without breaking validity, but this is not
      guaranteed to be the globally sparsest valid displacement (feature
      interactions make the result order-dependent — the ascending-|t_j|
      order is the standard, sensible heuristic).

    Parameters
    ----------
    A_orig : (n_A, d)  original branch points.
    t : (d,)  valid displacement; A_orig + t must dominate B.
    B_pts : (n_B, d)  target branch points.
    competitor_pts_list : list of (n_X, d) arrays  the live competitors.
    linkage : {'complete', 'ward'}.
    eps : float  strict-domination margin (match the solver's).
    refine_magnitude : bool
        If True, after fixing WHICH features stay changed, additionally shrink
        their magnitudes: for each surviving feature, binary-search t_j toward
        0 to the smallest |t_j| that keeps validity. Reduces move size on kept
        features without changing the sparsity pattern. (Off by default.)

    Returns
    -------
    t_sparse : (d,)  sparsified (and optionally magnitude-refined) displacement.
    info : dict  changed-before/after, reverted coordinates, validity flags.
    """
    A_orig:np.ndarray = np.asarray(A_orig, dtype=float)
    t:np.ndarray = np.asarray(t, dtype=float).copy()
    d:int = t.shape[0]

    start_valid = _dominates(A_orig + t, B_pts, competitor_pts_list, linkage, eps) 

    # coordinates actually changed, ascending |t_j| (smallest move = most revertible first)
    changed = [j for j in range(d) if abs(t[j]) > 0.0]
    order = sorted(changed, key=lambda j: abs(t[j]))

    reverted = []
    for j in order:
        saved = t[j]
        t[j] = 0.0                                     # try full revert to original
        if _dominates(A_orig + t, B_pts, competitor_pts_list, linkage, eps):
            reverted.append(j)                         # keep it: feature not needed
        else:
            t[j] = saved                               # undo: feature must stay changed

    
    final_changed = [j for j in range(d) if abs(t[j]) > 0.0]
    info = {
        "start_valid": start_valid,
        "changed_before": sorted(changed),
        "n_changed_before": len(changed),
        "reverted": sorted(reverted),
        "changed_after": final_changed,
        "n_changed_after": len(final_changed),
        "still_valid": _dominates(A_orig + t, B_pts, competitor_pts_list, linkage, eps),
    }
    return t, info


def project_sphere_closed_form(mu_A, kappa, R):
    """
    Nearest point on the sphere {||p - kappa|| = R} to mu_A, in closed form:
        p* = kappa + R (mu_A - kappa) / ||mu_A - kappa||.
    Exact, O(d). Degenerate mu_A == kappa: any surface point is optimal; return
    kappa + R e_0.
    """
    g = mu_A - kappa
    nrm = np.linalg.norm(g)
    if nrm < 1e-12:
        u = np.zeros_like(mu_A, dtype=float); u[0] = 1.0
    else:
        u = g / nrm
    return kappa + R * u
 

def project_single_sphere_sdp(mu_A, kappa, R, convex_ball_in, convex_half,
                              solver=None):
    """
    Exact projection of mu_A onto {||p - kappa|| = R} ∩ K via the Shor SDP,
    tight for a single quadratic equality. Returns p or None.
    """
    d = len(mu_A)
    if solver is None:
        solver = cp.CLARABEL if hasattr(cp, "CLARABEL") else cp.SCS
    p = cp.Variable(d)
    P = cp.Variable((d, d), symmetric=True)
    M = cp.bmat([[P, cp.reshape(p, (d, 1), order='C')],
                 [cp.reshape(p, (1, d), order='C'), np.array([[1.0]])]])
    cons = [M >> 0,
            cp.trace(P) - 2 * kappa @ p + float(kappa @ kappa) == R ** 2]
    for (kc, Rc) in convex_ball_in:
        cons.append(cp.norm(p - kc) <= Rc)
    for (a, b) in convex_half:
        cons.append(a @ p <= b)
    obj = cp.Minimize(cp.trace(P) - 2 * mu_A @ p + float(mu_A @ mu_A))
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=solver)
    except Exception:
        return None
    return None if p.value is None else np.asarray(p.value, dtype=float)
 