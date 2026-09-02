"""
dice.py: a DiCE-based competitor baseline for the branch counterfactual.

Idea
----
Our method solves the counterfactual exactly via geometry. As a competitor
baseline we let DiCE (a generic, model-agnostic counterfactual generator) find
a valid move by treating our validity test as a black-box binary classifier:

    f(t) = 1  iff  the rigidly-translated branch A + t dominates B against every
                   live competitor (a valid counterfactual under our definition)
    f(t) = 0  otherwise.

DiCE then searches for a translation t that flips the query point (t = 0, "no
move", class 0) to class 1 (valid), minimizing change.

Key design choices (see the accompanying discussion):

  * DiCE optimizes over the shared rigid translation t in R^d, not individual
    points, so the move stays rigid (A translated as a whole), matching our
    method.

  * The classifier is a hard threshold (validity is boolean), hence
    non-differentiable, so we use DiCE's model-agnostic genetic search
    (method="genetic"), never the gradient backend. Basically we are 
    simulating a binary classifier.

  * predict_proba returns a soft score sigmoid(scale * margin), where
    margin = min_X (d(A+t, X) - d(A+t, B)) is positive exactly when valid, and the sigmpoid makes >0.5 positive.
    d(A+t, X) - d(A+t, B) is indeed negative when the A is closer to the closest competitor
    than to B.
    This gives the genetic search a smooth preference signal (get closer to
    validity) while the hard label `predict` stays the exact boolean test.

"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from dice_ml import Dice, Data, Model
from modules.utilityFunctions import distLinkage, wardDistance, complete_linkage_max


# ======================================================================
# The constraint-as-classifier wrapper
# ======================================================================
class ConstraintClassifier:
    """
    Sklearn-style binary classifier over the rigid translation t (each input
    row is one candidate t of length d), exposing our domination test to DiCE.

    predict(T)       -> array of {0,1}: 1 iff A + t dominates B vs all competitors.
    predict_proba(T) -> array of [P(0), P(1)] with
                        P(1) = sigmoid(scale * margin),
                        margin = min_X ( d(A+t, X) - d(A+t, B) ).
                        margin > eps  <=>  valid, so P(1) > 0.5 there; the soft
                        value gives genetic DiCE a gradient of preference.

    Parameters
    ----------
    A_pts, B_pts : (n_A, d), (n_B, d)   query and target branch points.
    comp_pts : list of (n_X, d)         live competitor point sets.
    linkage : {'complete', 'ward'}.
    eps : float                          strict-domination margin (match solver).
    scale : float                        sigmoid steepness for the soft score.
    """

    def __init__(self, A_pts, B_pts, comp_pts, linkage, eps=0.0, scale=5.0):
        self.A = np.asarray(A_pts, float)
        self.B = np.asarray(B_pts, float)
        self.comp = [np.asarray(c, float) for c in comp_pts]
        self.linkage = linkage
        self.eps = eps
        self.scale = scale
        self.d = self.A.shape[1]
        self.classes_ = np.array([0, 1])          # sklearn-compat attribute

    def _margin(self, t):
        Ap = self.A + t
        dB = distLinkage(Ap, self.B, self.linkage)
        if not self.comp:
            return 1.0                            # no competitors -> trivially valid
        return min(distLinkage(Ap, X, self.linkage) - dB for X in self.comp)

    def predict(self, T):
        T = np.atleast_2d(np.asarray(T, float))
        return np.array([1 if self._margin(t) > self.eps else 0 for t in T], dtype=int)

    def predict_proba(self, T):
        T = np.atleast_2d(np.asarray(T, float))
        out = []
        for t in T:
            m = self._margin(t)
            p1 = 1.0 / (1.0 + np.exp(-self.scale * m))
            out.append([1.0 - p1, p1])
        return np.array(out)


# ======================================================================
# The DiCE baseline runner
# ======================================================================
def run_dice_baseline(A_pts, B_pts, comp_pts, linkage, X_all, mu_A=None,
                      total_CFs=1, eps=0.0, t_bound=5.0,
                      n_train=400, seed=0, verbose=False):
    """
    Run genetic DiCE against the ConstraintClassifier and return the best valid
    counterfactual translation it finds.
 
    Search space & training frame
    ------------------------------
    DiCE searches over the rigid translation t. Its Data frame declares the
    feature (t-coordinate) ranges and must contain BOTH classes so a class-1
    (valid) target exists. We build the frame from the REAL dataset expressed as
    translations, t_i = x_i - mu_A (data-grounded ranges, on-manifold seeding),
    PLUS a guaranteed-valid seed:
 
      * Ward: t_seed = mu_B - mu_A. Landing A's centroid on B's centroid drives
        the Ward distance to 0, which is <= every competitor distance, so
        t_seed is ALWAYS feasible (Ward feasibility guarantee). This
        unconditionally supplies a class-1 example (and a valid fallback).
      * Complete linkage: no single translation is guaranteed feasible (the
        farthest-pair distance has a positive floor), so the mu_B seed does NOT
        transfer. Not implemented here yet — raises NotImplementedError.
 
    If, despite the seed, the frame is still one-class (shouldn't happen for
    Ward), we augment with a uniform box sample as a safety net.
 
    No fallback
    -----------
    If DiCE returns nothing valid, we do NOT fall back to t_seed (the
    guaranteed-valid centroid move for Ward) -- we report that no
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
    
    from dice_ml import Dice, Data, Model
 
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
        raise NotImplementedError(
            "Complete-linkage DiCE seed not implemented: mu_B is not guaranteed "
            "feasible under complete linkage. Seed with the min-max/MEB baseline "
            "translation instead (handled separately).")
    else:
        raise ValueError(f"Unsupported linkage {linkage!r}.")
 
    # ---- training frame: dataset-as-translations + the valid seed ----
    T_frame = np.vstack([X_all - mu_A, t_seed[None, :]])
    y = clf.predict(T_frame)
 
    # safety net: if still one-class (shouldn't happen for Ward, since t_seed is
    # valid and t=0 is invalid), augment with a uniform box sample.
    if len(set(y.tolist())) < 2:
        rng = np.random.default_rng(seed)
        T_box = rng.uniform(-t_bound, t_bound, size=(n_train, d))
        T_frame = np.vstack([T_frame, T_box])
        y = clf.predict(T_frame)
 
    cols = [f"t{j}" for j in range(d)]
    df = pd.DataFrame(T_frame, columns=cols)
    df["valid"] = y
 
    data = Data(dataframe=df, continuous_features=cols, outcome_name="valid")
    model = Model(model=clf, backend="sklearn", model_type="classifier")
    exp = Dice(data, model, method="genetic")
 
    # query = no move (t = 0, class 0); ask DiCE to reach class 1 (valid)
    q = pd.DataFrame([{c: 0.0 for c in cols}])
    best_t, n_ret = None, 0
    try:
        res = exp.generate_counterfactuals(
            q, total_CFs=total_CFs, desired_class=1, verbose=verbose)
        cf = res.cf_examples_list[0].final_cfs_df
        if cf is not None and len(cf) > 0:
            T = cf[cols].to_numpy(float)
            n_ret = len(T)
            # keep only the CFs that are ACTUALLY valid under our exact test,
            # then take the closest by our own L2.
            valid_cfs = [t for t in T if clf.predict([t])[0] == 1]
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
            "method": "dice-genetic",
        }

    valid = bool(clf.predict([best_t])[0])          # exact recheck
    return A_pts + best_t, {
        "t": best_t,
        "feasible": valid,
        "distance_moved": float(np.linalg.norm(best_t)),
        "found": True,
        "n_cfs_returned": n_ret,
        "method": "dice-genetic",
    }