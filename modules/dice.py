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

import random

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from dice_ml import Dice, Data, Model
from modules.utilityFunctions import distLinkage, wardDistance, complete_linkage_max, complete_linkage_meb_seed


# ======================================================================
# The constraint-as-classifier wrapper
# ======================================================================
class ConstraintClassifier:
    """
    Sklearn-style binary classifier over the rigid translation t (each input
    row is one candidate t of length d), exposing our domination test to DiCE.

    predict(T)       -> array of {0,1}: 1 iff A + t dominates B vs all competitors
                        AND vs A's own unmoved position (t=0 is never valid,
                        and neither is any t that still leaves A effectively
                        where it started -- matches metrics.validity's A_orig
                        handling and HiCE's own solver, which always treats
                        A's unmoved position as a competitor).
    predict_proba(T) -> array of [P(0), P(1)] with
                        P(1) = sigmoid(scale * margin),
                        margin = min_X ( d(A+t, X) - d(A+t, B) ), X ranging over
                        comp_pts AND A's own original (unmoved) position.
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
        # A's own unmoved position is always a competitor too, so t=0 (and
        # anything that still leaves A effectively where it started) is never
        # valid -- see class docstring.
        self.comp = [np.asarray(c, float) for c in comp_pts] + [self.A]
        self.linkage = linkage
        self.eps = eps
        self.scale = scale
        self.d = self.A.shape[1]
        self.classes_ = np.array([0, 1])          # sklearn-compat attribute

        # ---- precomputed centroids/constants for the vectorized Ward path ----
        # Ward distance only depends on cluster centroids and sizes, and a rigid
        # translation shifts the centroid by exactly t (mean(A + t) = mean(A) + t),
        # so the whole margin becomes centroid arithmetic -- no need to touch the
        # raw points (or call cdist) per candidate t.
        self.mu_A = self.A.mean(axis=0)
        self.n_A = len(self.A)
        if self.linkage == 'ward':
            self.mu_B = self.B.mean(axis=0)
            n_B = len(self.B)
            self.c_AB = np.sqrt(2.0 * self.n_A * n_B / (self.n_A + n_B))
            if self.comp:
                self.mu_comp = np.stack([c.mean(axis=0) for c in self.comp])   # (k, d)
                n_comp = np.array([len(c) for c in self.comp], dtype=float)    # (k,)
                self.c_comp = np.sqrt(2.0 * self.n_A * n_comp / (self.n_A + n_comp))
            else:
                self.mu_comp = None
                self.c_comp = None

    def _margin(self, t):
        """Single-t margin, exact (used as the complete-linkage fallback)."""
        Ap = self.A + t
        dB = distLinkage(Ap, self.B, self.linkage)
        if not self.comp:
            return 1.0                            # no competitors -> trivially valid
        return min(distLinkage(Ap, X, self.linkage) - dB for X in self.comp)

    def _margin_batch(self, T):
        """
        Vectorized margin for a batch of candidate translations T, (m, d).
        Ward only: reduces to centroid distances, precomputed in __init__.
        """
        muAp = self.mu_A[None, :] + T                                   # (m, d)
        dB = self.c_AB * np.linalg.norm(muAp - self.mu_B[None, :], axis=1)  # (m,)
        if self.mu_comp is None:
            return np.full(T.shape[0], np.inf)    # no competitors -> trivially valid
        diff = muAp[:, None, :] - self.mu_comp[None, :, :]               # (m, k, d)
        dX = self.c_comp[None, :] * np.linalg.norm(diff, axis=2)         # (m, k)
        return dX.min(axis=1) - dB                                       # (m,)

    def _margins(self, T):
        if self.linkage == 'ward':
            return self._margin_batch(T)
        return np.array([self._margin(t) for t in T])

    def predict(self, T):
        T = np.atleast_2d(np.asarray(T, float))
        m = self._margins(T)
        return (m > self.eps).astype(int)

    def predict_proba(self, T):
        T = np.atleast_2d(np.asarray(T, float))
        m = self._margins(T)
        p1 = 1.0 / (1.0 + np.exp(-self.scale * m))
        return np.column_stack([1.0 - p1, p1])


class RelabelEncoding:
    """
    Flat categorical encoding of a per-coordinate relabelling.

    Under the Hamming metric the admissible move is not a translation but a
    relabelling: on each coordinate j a function f_j maps every value that A
    takes there to one target value, applied to all members of A at once. That
    is not a point in R^d, which is why a translation search (DiCE, BayCon as
    used for the Euclidean case) has nothing to optimise over.

    It IS, however, a flat vector of CATEGORICAL variables: one slot per
    (coordinate, source value present in A), whose domain is that coordinate's
    A-union-B values. That is exactly the input shape dice_ml and BayCon
    expect, and it is where their native categorical support is finally used.

    Targets are restricted to values appearing in A or B on that coordinate:
    mapping an A-value to something B never takes can only add distance to B.
    Identity is therefore always available, so the identity relabelling is a
    valid query point.
    """

    def __init__(self, A_pts, B_pts):
        A = np.asarray(A_pts)
        B = np.asarray(B_pts)
        self.d = A.shape[1]
        self.slots = []          # (coordinate, source value)
        self.domains = []        # allowed target values, per slot
        for j in range(self.d):
            src = sorted({int(v) for v in A[:, j]})
            tgt = sorted({int(v) for v in A[:, j]} | {int(v) for v in B[:, j]})
            for a in src:
                self.slots.append((j, a))
                self.domains.append(tgt)
        self.n_slots = len(self.slots)

    def identity(self):
        """The query point: every value maps to itself."""
        return np.array([a for (_j, a) in self.slots], dtype=float)

    def feature_names(self):
        return ["f%d_%d" % (j, a) for (j, a) in self.slots]

    def snap(self, v):
        """Round a candidate to the nearest legal value in each slot.

        A genetic search may propose values off the domain; snapping keeps
        every candidate a well-formed relabelling instead of discarding it.
        """
        out = np.empty(self.n_slots, dtype=int)
        for i, x in enumerate(np.asarray(v, dtype=float).ravel()):
            dom = self.domains[i]
            out[i] = min(dom, key=lambda c: abs(c - x))
        return out

    def apply(self, v, A_pts):
        """Apply the relabelling encoded by v to A."""
        v = self.snap(v)
        A = np.asarray(A_pts)
        out = A.copy()
        mapping = {}
        for i, (j, a) in enumerate(self.slots):
            mapping[(j, a)] = int(v[i])
        for j in range(self.d):
            col = A[:, j]
            out[:, j] = [mapping.get((j, int(x)), int(x)) for x in col]
        return out


class RelabelConstraintClassifier:
    """
    ConstraintClassifier for the Hamming relabelling move space.

    Same contract as ConstraintClassifier -- an sklearn-style predict over
    candidate moves -- but a candidate is a relabelling vector rather than a
    translation, and the dissimilarity is complete-linkage HAMMING (max over
    member pairs of the count of differing coordinates) rather than Euclidean.
    distLinkage hardcodes metric_='euclidean' for 'complete', so scoring
    categorical candidates through it would compare integer codes with
    Euclidean geometry.
    """

    def __init__(self, A_pts, B_pts, comp_pts, eps=0.0, scale=5.0):
        self.A = np.asarray(A_pts)
        self.B = np.asarray(B_pts)
        self.comp = [np.asarray(c) for c in comp_pts]
        self.eps = eps
        self.scale = scale
        self.enc = RelabelEncoding(self.A, self.B)

    @staticmethod
    def _dmax(P, Q):
        return float(max(int(np.count_nonzero(p != q)) for p in P for q in Q))

    def _margin(self, v):
        Ap = self.enc.apply(v, self.A)
        dB = self._dmax(Ap, self.B)
        cands = list(self.comp) + [self.A]     # A's unmoved position competes too
        return min(self._dmax(Ap, X) - dB for X in cands)

    def _margins(self, V):
        # dice_ml hands us the whole frame row, which carries the padding
        # column the genetic explainer needs (see run_dice_relabel_multi);
        # only the leading slot columns are part of the relabelling.
        V = np.atleast_2d(np.asarray(V, float))[:, :len(self.enc.domains)]
        return np.array([self._margin(v) for v in V])

    def predict(self, V):
        return (self._margins(V) > self.eps).astype(int)

    def predict_proba(self, V):
        m = self._margins(V)
        p1 = 1.0 / (1.0 + np.exp(-self.scale * m))
        return np.column_stack([1.0 - p1, p1])


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

    # DiCE's genetic search draws from the GLOBAL numpy/python random state
    # (no local Generator to pass in), so reseed both here -- this makes a
    # standalone call with the same `seed` and inputs fully reproducible,
    # independent of whatever ran before it.
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
        # may itself be invalid. The search frame below is built from it either
        # way and the one-class safety net handles the case where nothing valid
        # turns up. See utilityFunctions.complete_linkage_meb_seed.
        t_seed = complete_linkage_meb_seed(A_pts, B_pts)
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


def run_dice_baseline_multi(A_pts, B_pts, comp_pts, linkage, X_all, mu_A=None,
                             k=10, eps=0.0, t_bound=5.0, n_train=400, seed=0,
                             verbose=False):
    """
    Diversity variant of run_dice_baseline: asks DiCE for exactly k
    counterfactuals (total_CFs=k) -- DiCE's genetic search already optimizes
    for diversity among the CFs it returns, so no extra selection is layered
    on top here. Every returned row is exact-rechecked against our validity
    test; only the valid ones are kept, in the order DiCE returned them
    (first-k, not re-ranked by proximity).

    Same setup (classifier, training frame, guaranteed-valid seed) as
    run_dice_baseline; see its docstring for details.

    Returns
    -------
    results : list of (A_prime, info) tuples, length <= k (possibly 0 if
        DiCE returned nothing valid). Each info dict has the same shape as
        run_dice_baseline's: t, feasible (always True here), distance_moved,
        found (always True), n_cfs_returned (DiCE's raw row count before the
        validity filter), method.
    """
    from dice_ml import Dice, Data, Model

    # see run_dice_baseline: DiCE's genetic search draws from the GLOBAL
    # numpy/python random state, so reseed here for standalone reproducibility.
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
        # may itself be invalid. The search frame below is built from it either
        # way and the one-class safety net handles the case where nothing valid
        # turns up. See utilityFunctions.complete_linkage_meb_seed.
        t_seed = complete_linkage_meb_seed(A_pts, B_pts)
    else:
        raise ValueError(f"Unsupported linkage {linkage!r}.")

    T_frame = np.vstack([X_all - mu_A, t_seed[None, :]])
    y = clf.predict(T_frame)

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

    q = pd.DataFrame([{c: 0.0 for c in cols}])
    n_ret = 0
    valid_ts = []
    try:
        res = exp.generate_counterfactuals(
            q, total_CFs=k, desired_class=1, verbose=verbose)
        cf = res.cf_examples_list[0].final_cfs_df
        if cf is not None and len(cf) > 0:
            T = cf[cols].to_numpy(float)
            n_ret = len(T)
            valid_ts = [t for t in T if clf.predict([t])[0] == 1]
    except Exception:
        valid_ts = []

    results = []
    for t in valid_ts[:k]:
        results.append((A_pts + t, {
            "t": t,
            "feasible": True,
            "distance_moved": float(np.linalg.norm(t)),
            "found": True,
            "n_cfs_returned": n_ret,
            "method": "dice-genetic",
        }))
    return results

def _relabel_frame(clf, seed, n_rand=400):
    """
    Training frame for a search over the relabelling encoding.

    Rows are candidate relabellings: the identity (the query), the min-max
    relabelling seed, and random draws from each slot's domain. Labelled by
    the classifier, this gives the search both classes to learn from -- the
    same role the translation frame plays in the Euclidean wrappers.
    """
    import numpy as _np
    enc = clf.enc
    rng = _np.random.default_rng(seed)
    rows = [enc.identity()]
    try:
        from modules.hiceCounterfactuals import _hamming_minmax_relabel
        relabel, _R, _apply = _hamming_minmax_relabel(clf.A, clf.B)
        rows.append(_np.array([relabel.get(j, {}).get(a, a)
                               for (j, a) in enc.slots], dtype=float))
    except Exception:
        pass
    for _ in range(n_rand):
        rows.append(_np.array([rng.choice(dom) for dom in enc.domains], dtype=float))
    V = _np.vstack(rows)
    return V, clf.predict(V)


def run_dice_relabel_multi(A_pts, B_pts, comp_pts, k=10, eps=0.0, seed=0,
                           verbose=False):
    """
    DiCE over the Hamming RELABELLING move space.

    The Euclidean wrapper hands DiCE a translation t; that is meaningless on
    categorical codes, where the admissible move is a per-coordinate
    relabelling. Here the search variable is the relabelling itself, encoded
    as one CATEGORICAL slot per (coordinate, source value) -- which is what
    dice_ml's native categorical support expects, and why
    continuous_features=[] below.
    """
    from dice_ml import Dice, Data, Model
    import pandas as _pd, numpy as _np, random as _random
    _random.seed(seed); _np.random.seed(seed)

    clf = RelabelConstraintClassifier(A_pts, B_pts, comp_pts, eps=eps)
    enc = clf.enc
    V, y = _relabel_frame(clf, seed)
    if len(set(y.tolist())) < 2:
        return []                      # nothing valid to learn from

    cols = enc.feature_names()
    df = _pd.DataFrame(V, columns=cols)
    # dice_ml's genetic search cannot run on an ALL-categorical frame: its
    # compute_proximity_loss reshapes the continuous-feature block, which is
    # empty when continuous_features=[], and dies with
    #   ValueError: cannot reshape array of size 0 into shape (0)
    # A single constant continuous column makes that term well-defined (and
    # identically zero) while the categorical proximity term works normally.
    # It is excluded from features_to_vary, so it can never be changed and
    # cannot affect which relabelling is proposed.
    df["_const"] = 0.0
    df["valid"] = y
    data = Data(dataframe=df, continuous_features=["_const"], outcome_name="valid")
    model = Model(model=clf, backend="sklearn", model_type="classifier")
    exp = Dice(data, model, method="genetic")

    qd = dict(zip(cols, enc.identity())); qd["_const"] = 0.0
    q = _pd.DataFrame([qd])
    out = []
    try:
        res = exp.generate_counterfactuals(q, total_CFs=k, desired_class=1,
                                           features_to_vary=cols,
                                           verbose=verbose)
        cfs = res.cf_examples_list[0].final_cfs_df
        if cfs is not None and len(cfs):
            for v in cfs[cols].to_numpy(float):
                if clf.predict([v])[0] != 1:
                    continue
                Ap = enc.apply(v, clf.A)
                out.append((Ap, {"t": enc.snap(v), "feasible": True,
                                 "found": True, "method": "dice-relabel",
                                 "distance_moved": float(_np.count_nonzero(
                                     (Ap != clf.A).any(axis=0)))}))
    except Exception:
        return out[:k]
    return out[:k]


def run_baycon_relabel_multi(A_pts, B_pts, comp_pts, k=10, eps=0.0, seed=0):
    """BayCon over the same relabelling encoding; see run_dice_relabel_multi."""
    import numpy as _np, random as _random
    from joblib import parallel_backend
    import baycon.baycon.bayesian_generator as _bg
    from baycon.common.DataAnalyzer import DataAnalyzer
    from baycon.common.Target import Target
    _random.seed(seed); _np.random.seed(seed)

    clf = RelabelConstraintClassifier(A_pts, B_pts, comp_pts, eps=eps)
    enc = clf.enc
    V, y = _relabel_frame(clf, seed)
    if len(set(y.tolist())) < 2:
        return {"random": [], "proximal": []}

    names = _np.array(enc.feature_names())
    target = Target(target_type="classification", target_feature="valid", target_value=1)
    # every slot is categorical: its domain is a set of labels, not a scale
    da = DataAnalyzer(V, y, names, target, cat_features=list(names))
    out = []
    try:
        with parallel_backend("threading", n_jobs=1):
            cands, _ = _bg.run(enc.identity(), 0, target, da, clf)
        for v in cands:
            if clf.predict([v])[0] != 1:
                continue
            Ap = enc.apply(v, clf.A)
            out.append((Ap, {"t": enc.snap(v), "feasible": True, "found": True,
                             "method": "baycon-relabel",
                             "distance_moved": float(_np.count_nonzero(
                                 (Ap != clf.A).any(axis=0)))}))
    except Exception:
        pass
    if not out:
        return {"random": [], "proximal": []}
    rng = _np.random.default_rng(seed)
    idx = rng.choice(len(out), size=min(k, len(out)), replace=False)
    rand = [out[i] for i in idx]
    prox = sorted(out, key=lambda t: t[1]["distance_moved"])[:k]
    return {"random": rand, "proximal": prox}
