from typing import List, Dict, Iterable
import numpy as np
from modules.utilityFunctions import project_sphere_closed_form, project_single_sphere_sdp, active, feasible, unit, ward_info, complete_linkage_max, get_children, _dominates
import miniball
from scipy.spatial.distance import cdist
from scipy.optimize import  milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp
from scipy.optimize import minimize
from itertools import combinations


def _hamming_minmax_relabel(A_pts, B_pts):
    """
    Complete-linkage Hamming BASELINE: the rigid relabeling of A that
    minimises A's FARTHEST Hamming distance to B, ignoring competitors.

    This is the categorical counterpart of the Euclidean MEB baseline
    (t* = -center(MEB(V_AB))): same role, same "min-max, competitors
    ignored" semantics, so the achieved distance is the feasibility FLOOR
    min_t d(A(t), B). If even this cannot beat a competitor, the pair's
    counterfactual region is plausibly empty rather than the solver having
    failed.

    Move
    ----
    Per coordinate j a single function f_j is applied to EVERY point of A:
    each source value alpha that A takes on j maps to one target value beta.
    No bijection is required (two A-values may collapse onto one target).
    This is the same move class as _solve_branch_hamming -- binary XOR is the
    two-value special case -- and it is what keeps A a valid node: equal
    A-values stay equal because f_j is a well-defined function.

    Formulation (exact MILP, solved with scipy/HiGHS -- no licence needed)
    --------------------------------------------------------------------
    With indicators y[j, alpha, beta] = 1 iff f_j(alpha) = beta, Hamming
    distance is affine:

        d_H(f(a), b) = sum_j (1 - y[j, a_j, b_j]) = d - sum_j y[j, a_j, b_j]

    so bounding the worst pair by an epigraph variable h gives

        min  h + w * sum_j moved_j
        s.t. sum_beta y[j, alpha, beta] = 1          for each (j, alpha)
             sum_j y[j, a_j, b_j] + h >= d           for each (a, b) in AxB
             moved_j + y[j, alpha, alpha] >= 1       for each (j, alpha)
             y, moved binary;  0 <= h <= d

    The `moved_j` term breaks ties toward relabeling FEWER coordinates. It
    cannot disturb the min-max optimum: h takes integer values in unit
    steps, and w is chosen so w * d < 1, so no tie-break can ever buy a
    reduction in h. Without it the solver returns an arbitrary member of a
    frequently large optimal set, which made the baseline non-deterministic
    and needlessly dense.

    Targets per coordinate are restricted to the values appearing in A or B
    on that coordinate: mapping an A-value to something B never takes can
    only ever add distance to B. Identity is therefore always available
    (every source value is also a target), so "keep as-is" is always
    feasible and the problem is never infeasible -- unlike the
    competitor-constrained solve.

    Returns
    -------
    relabel : {j: {alpha: beta}}
        Per-coordinate relabeling map, matching the `info['t']` convention
        of _solve_branch_hamming.
    R : int
        Achieved max Hamming distance from the relabeled A to B, in
        coordinates (the feasibility floor).
    apply : callable
        apply(A_in) -> relabeled copy of A_in, for reusing the same f on
        other rows.
    """
    A = np.asarray(A_pts)
    B = np.asarray(B_pts)
    n_A, d = A.shape

    src = [sorted({int(v) for v in A[:, j]}) for j in range(d)]
    tgt = [sorted({int(v) for v in A[:, j]} | {int(v) for v in B[:, j]})
           for j in range(d)]

    # ---- variable layout: y[j,alpha,beta], then moved_j, then h ----
    ycol, col = {}, 0
    for j in range(d):
        for al in src[j]:
            for be in tgt[j]:
                ycol[(j, al, be)] = col
                col += 1
    mcol = {}
    for j in range(d):
        mcol[j] = col
        col += 1
    hcol = col
    n_vars = col + 1

    w = 1.0 / (2.0 * d + 1.0)          # w * d < 1, so h always dominates
    c_obj = np.zeros(n_vars)
    c_obj[hcol] = 1.0
    for j in range(d):
        c_obj[mcol[j]] = w

    rows, lo, hi = [], [], []

    # f_j is a function: exactly one target per source value
    for j in range(d):
        for al in src[j]:
            r = np.zeros(n_vars)
            for be in tgt[j]:
                r[ycol[(j, al, be)]] = 1.0
            rows.append(r); lo.append(1.0); hi.append(1.0)

    # every A-B pair within h:  sum_j y[j,a_j,b_j] + h >= d
    for a in A:
        for b in B:
            r = np.zeros(n_vars)
            for j in range(d):
                r[ycol[(j, int(a[j]), int(b[j]))]] += 1.0
            r[hcol] = 1.0
            rows.append(r); lo.append(float(d)); hi.append(np.inf)

    # moved_j = 1 if any source value on j is not kept
    for j in range(d):
        for al in src[j]:
            r = np.zeros(n_vars)
            r[mcol[j]] = 1.0
            r[ycol[(j, al, al)]] = 1.0
            rows.append(r); lo.append(1.0); hi.append(np.inf)

    constraints = LinearConstraint(np.array(rows), lo, hi)
    integrality = np.ones(n_vars)
    integrality[hcol] = 0                      # h is continuous (integral at opt)
    ub = np.ones(n_vars)
    ub[hcol] = float(d)
    bounds = Bounds(lb=np.zeros(n_vars), ub=ub)

    res = milp(c_obj, constraints=constraints, bounds=bounds,
               integrality=integrality)
    if not res.success or res.x is None:
        # Identity is always feasible, so this means a solver failure rather
        # than genuine infeasibility; fall back to the identity relabeling so
        # the caller still gets a well-formed baseline it can score.
        relabel = {j: {al: al for al in src[j]} for j in range(d)}
    else:
        x = res.x
        relabel = {}
        for j in range(d):
            m = {}
            for al in src[j]:
                best = max(tgt[j], key=lambda be: x[ycol[(j, al, be)]])
                m[al] = best
            relabel[j] = m

    def apply(A_in):
        A_in = np.asarray(A_in)
        out = A_in.copy()
        for j in range(A_in.shape[1]):
            mj = relabel.get(j, {})
            for row in range(A_in.shape[0]):
                v = int(A_in[row, j])
                if v in mj:
                    out[row, j] = mj[v]
        return out

    A_rel = apply(A)
    R = int(max(int(np.count_nonzero(a != b)) for a in A_rel for b in B))
    return relabel, R, apply


class hiceCounterfactuals:

    def __init__(self, X:np.ndarray, Z:np.ndarray, 
                 metric:str, linkage:str, anchor='min'):


        # data matrix
        self.X:np.ndarray = X

        self.n:int = len(self.X)

        self.d:int = X.shape[1]

        # Linkage matrix returned by sipy
        self.Z:np.ndarray = Z

        # The distance metric
        self.metric:str=metric.strip().lower()

        # The linkage criterion used
        self.linkage:str=linkage.strip().lower()

        # anchor to chose the competitor set
        self.anchor:str = anchor.strip().lower()

        assert self.anchor == 'min', "Only min is implemented as the anchor to choose the competitor set"

        assert self.linkage in ["complete", "ward"], "Complete and Ward linkage are the only one supported right now"

        assert self.metric in ['euclidean', 'hamming'], "Only euclidean and hamming are implemented"

        if self.linkage == 'ward' and self.metric != 'euclidean':
            raise ValueError("ward linkage admits the euclidean metric only")


    def baseline(self, B_leaves: List[int], A_leaves: List[int]):
        """
        Baseline counterfactual router, mirroring findCounterfactual.
    
        Dispatches on self.linkage to the linkage-appropriate naive baseline. All
        baselines share the same "minimum displacement" reading: move A as little
        as possible so it sits at / touches the target branch B, ignoring
        competitors. The linkage only decides how the *method* (not the baseline)
        measures domination; the baseline itself is deliberately a simple
        least-move-toward-B reference for whichever linkage is in use.
    
            complete : min-max / MEB move — rigidly translate A to minimize its
                    farthest distance to B (complete linkage's own functional).
            ward     : representative (centroid mu_A) -> closest sample of B, i.e.
                    the minimum displacement that lands A's representative on B;
                    reduces to "move the leaf to B's nearest sample" for a leaf.
    
        Returns
        -------
        A_prime : np.ndarray (n_A x d)  the moved branch (or None if a baseline
                cannot produce a move, though the current baselines always do).
        info : dict  baseline-specific diagnostics (t, distance_moved, and a
                linkage-appropriate distance-to-B / feasibility-floor field).
        """
        if self.linkage == 'complete':
            A_prime, info = self.baselineCompleteLinkage(B_leaves=B_leaves, A_leaves=A_leaves)
        elif self.linkage == 'ward':
            A_prime, info = self.baselineWard(B_leaves=B_leaves, A_leaves=A_leaves)
        else:
            raise ValueError(f"Unsupported linkage {self.linkage!r}; use 'complete' or 'ward'.")
    
        return A_prime, info

    def baselineWard(self, B_leaves: List[int], A_leaves: List[int]):
        """
        Ward baseline: rigidly translate branch A so its representative (the
        centroid mu_A, Ward's own notion of a cluster's location) lands on the
        closest sample of the target branch B, ignoring competitors.
    
            b* = argmin_{b in B} || mu_A - b ||
            t  = b* - mu_A
            A' = A + t
    
        Reduction: for a single-leaf A, mu_A = x, so this moves the leaf onto the
        closest sample of B — exactly the requested leaf behavior. For a branch it
        moves the whole cloud rigidly by the same vector, sending its centroid to
        B's nearest point. Seamless because leaf vs branch differ only in whether
        mu_A is one point or a mean.
    
        Returns
        -------
        A_prime : np.ndarray (n_A x d)
        info : dict  (t, distance_moved, target_leaf, ward_distance_to_B, and the
                    Ward-native centroid->centroid move for comparison)
        """
        A_leaves_sorted = sorted(int(l) for l in A_leaves)
        B_leaves_sorted = sorted(int(l) for l in B_leaves)
    
        A_pts = np.asarray(self.X[A_leaves_sorted], dtype=float)
        B_pts = np.asarray(self.X[B_leaves_sorted], dtype=float)
    
        mu_A = A_pts.mean(axis=0)
        
    
        # representative -> closest sample of B ---
        dists = cdist(mu_A[None, :], B_pts, metric='euclidean')[0]
        k = int(np.argmin(dists))
        b_star = B_pts[k]
        t = b_star - mu_A
        A_prime = A_pts + t
    
        def ward_dist(P, Q):
            nP, nQ = len(P), len(Q)
            c = np.sqrt(2.0 * nP * nQ / (nP + nQ))
            return float(c * np.linalg.norm(P.mean(0) - Q.mean(0)))
    
    
        info = {
            "t": t,
            "distance_moved": float(np.linalg.norm(t)),
            "target_leaf": B_leaves_sorted[k],
            "ward_distance_to_B": ward_dist(A_prime, B_pts),
            "A_leaves_sorted": A_leaves_sorted,
            "metric": "ward",
        }
        return A_prime, info

    def baseline2(self, B_leaves: List[int], A_leaves: List[int]):
        """
        Second baseline, ward-only: rigidly translate branch A so its
        representative (centroid mu_A) lands on B's representative (centroid
        mu_B), instead of B's nearest sample as in `baselineWard`/`baseline`.

            t  = mu_B - mu_A
            A' = A + t

        Returns
        -------
        A_prime : np.ndarray (n_A x d)
        info : dict  (t, distance_moved, ward_distance_to_B, and the
                    Ward-native centroid->centroid move for comparison)
        """
        if self.linkage != 'ward':
            raise ValueError(
                f"baseline2 is only defined for linkage='ward' (got {self.linkage!r}).")

        A_leaves_sorted = sorted(int(l) for l in A_leaves)
        B_leaves_sorted = sorted(int(l) for l in B_leaves)

        A_pts = np.asarray(self.X[A_leaves_sorted], dtype=float)
        B_pts = np.asarray(self.X[B_leaves_sorted], dtype=float)

        mu_A = A_pts.mean(axis=0)
        mu_B = B_pts.mean(axis=0)

        t = mu_B - mu_A
        A_prime = A_pts + t

        def ward_dist(P, Q):
            nP, nQ = len(P), len(Q)
            c = np.sqrt(2.0 * nP * nQ / (nP + nQ))
            return float(c * np.linalg.norm(P.mean(0) - Q.mean(0)))

        info = {
            "t": t,
            "distance_moved": float(np.linalg.norm(t)),
            "ward_distance_to_B": ward_dist(A_prime, B_pts),
            "A_leaves_sorted": A_leaves_sorted,
            "metric": "ward",
        }
        return A_prime, info


    # ======================================================================
    # The baseline method
    # ======================================================================
    def baselineCompleteLinkage(self, B_leaves: List[int], A_leaves: List[int]):
        """
        Complete-linkage baseline: rigidly move branch A to minimize its
        *farthest* distance to branch B (min-max, matching complete linkage),
        ignoring competitors. 
    
        L2 (metric='euclidean'):
            t* = argmin_t max_{a,b} ||a + t - b|| which is the equivalent of -(center of MEB of {a-b}),
            solved exactly with `miniball`.
    
        Hamming (metric='hamming'), binary OR categorical:
            rigid RELABELING move — per coordinate a fixed function f_j applied
            to all of A (each A-value -> one target value; no bijection needed),
            minimizing the worst-pair Hamming distance. Generalizes binary XOR
            and subsumes overwrite; solved exactly as a small MILP.
    
        info['cl_distance_to_B'] is the achieved max A-B distance = feasibility
        floor min_t d(A(t), B). Baseline ignores competitors, so A_prime may
        still lose to one (expected; not proof no counterfactual exists).
    
        Returns
        -------
        A_prime : np.ndarray (n_A x d)
        info : dict
        """
        metric = getattr(self, 'metric', 'euclidean')
    
        A_leaves_sorted = sorted(int(l) for l in A_leaves)
        B_leaves_sorted = sorted(int(l) for l in B_leaves)
    
        if metric == 'euclidean':
            A_pts = np.asarray(self.X[A_leaves_sorted], dtype=float)
            B_pts = np.asarray(self.X[B_leaves_sorted], dtype=float)
    
            V = np.array([a - b for a in A_pts for b in B_pts], dtype=float)
            C, r2 = miniball.get_bounding_ball(V)
            t_star = -C
    
            A_prime = A_pts + t_star
            
            R = float(complete_linkage_max(A_prime, B_pts, metric_='euclidean'))
    
            info = {
                "t": t_star,
                "distance_moved": float(np.linalg.norm(t_star)),
                "cl_distance_to_B": R,
                "A_leaves_sorted": A_leaves_sorted,
                "metric": "euclidean",
            }
            return A_prime, info
    
        elif metric == 'hamming':
            A_pts = np.asarray(self.X[A_leaves_sorted])
            B_pts = np.asarray(self.X[B_leaves_sorted])
    
            relabel, R, apply = _hamming_minmax_relabel(A_pts, B_pts)
            A_prime = apply(A_pts)
    
            changed = int(np.count_nonzero((A_prime != A_pts).any(axis=0)))
            
            R_check = int(round(complete_linkage_max(A_prime.astype(float), B_pts.astype(float), metric_='hamming') * A_pts.shape[1]))
    
            info = {
                "t": relabel,                # {j: {alpha: beta}} per-coordinate map
                "distance_moved": changed,   # coordinates changed for >=1 point of A
                "cl_distance_to_B": R_check, # raw max Hamming distance (coords)
                "A_leaves_sorted": A_leaves_sorted,
                "metric": "hamming",
            }
            return A_prime, info
    
        else:
            raise ValueError(f"Unsupported metric {metric!r}; use 'euclidean' or 'hamming'.")


    def findCounterfactual(self, B_leaves: List[int], A_leaves: List[int]):
        '''
        B_leaves: Leaf indices of the target branch B
        A_leaves: Leaf indices of the branch to find a counterfactual for
        '''
        if self.linkage == 'complete':
            movedBranch, info = self.counterfactual_complete_linkage_branch(
                B_leaves=B_leaves, A_leaves=A_leaves, metric=self.metric, anchor=self.anchor
            )
        elif self.linkage == 'ward':
            # Ward is defined on centroids in Euclidean space; metric is not used
            # to change the geometry of the solve.
            movedBranch, info = self.counterfactual_ward_branch(
                B_leaves=B_leaves, A_leaves=A_leaves, anchor=self.anchor,
                index_to_name=getattr(self, 'index_to_name', None), sparsity=True
            )
        else:
            raise ValueError(f"Unsupported linkage {self.linkage!r}; use 'complete' or 'ward'.")
    
        return movedBranch, info


    def counterfactual_ward_branch(self, B_leaves, A_leaves,
                               anchor='min', eps=1e-6, index_to_name=None,
                               exhaustive=False, socp_solver=None,
                               sparsity=True, lam=1.0, verbose=False,
                               allowed_features=None, extra_competitors=None):
        """
        Ward-linkage counterfactual: rigidly translate branch A by t so A becomes
        Ward-closer to target branch B than to every live competitor.

        Objective
        ---------
        sparsity=False      : minimize ||t||^2  (least Euclidean displacement).
        sparsity=True       : minimize ||t||^2 + lam * ||t||_1  (elastic net),
                            which drives coordinates of t to exact zero,
                            yielding a FEATURE-SPARSE counterfactual. lam is
                            the sparsity weight (passed in); lam=0 recovers
                            the L2 case.
        sparsity='adaptive' : same elastic-net objective, but lam is derived
                            from the data instead of passed in:
                                t_star = mu_B - mu_A
                                lam = ||t_star||^2 / ||t_star||  (= ||t_star||)
                            i.e. lam scales with the distance between the
                            origin and target cluster centroids; the `lam`
                            argument is ignored in this mode.

        Geometry (objective-independent)
        --------------------------------
        Ward distance to a cluster S is c_{AS} ||mu_A + t - mu_S|| with
        c_{AS}=sqrt(2 n_A n_S/(n_A+n_S)) (centroid distance). With p = mu_A + t each
        competitor gives one Apollonius constraint, typed by size (w_S = c_{AS}^2):
            n_X < n_B : ball INTERIOR   ||p-kappa_X|| <= R_X   (convex, SOC)
            n_X = n_B : half-space                              (convex, linear)
            n_X > n_B : ball EXTERIOR   ||p-kappa_X|| >= R_X   (reverse-convex)
        Feasible region = convex body K minus the reverse-convex balls. Always
        feasible (mu_B in it). The L1 term changes ONLY the objective, never the
        constraints, so the Apollonius geometry is identical for both objectives.

        Exactness note under sparsity
        -----------------------------
        For the pure-L2 objective the reverse-convex sphere repair is EXACT (the
        single-active-ball case is solved by the radial closed form, or the tight
        S-lemma SDP when K binds). NEITHER tool survives the L1 term: the radial
        projection is L2-specific, and the Shor/S-lemma SDP is NOT tight for a
        quadratic-plus-L1 objective (its relaxation becomes loose). So with
        sparsity=True the single-sphere repair falls back to multistart SLSQP on the
        L2^2+lam*L1 objective, which is a heuristic (no global certificate) for the
        reverse-convex phases. Phase 1 (the convex projection) remains exact under
        L1, since adding lam*||t||_1 keeps it a convex program. In practice most
        instances resolve in Phase 1; the certificate is only weakened on the rarer
        sphere-binding instances.

        Parameters
        ----------
        sparsity : bool or 'adaptive'
                            False: pure L2. True: L2^2 + lam*L1 with lam as given.
                            'adaptive': L2^2 + lam*L1 with lam = ||mu_B-mu_A||^2/||mu_B-mu_A||.
        lam : float         L1 weight (used when sparsity=True; ignored/overridden
                            when sparsity='adaptive').
        exhaustive : bool   Phase-4 behaviour (see below).
        socp_solver : cvxpy solver constant, optional (default CLARABEL, else ECOS).
        allowed_features : array-like of {0,1}, length d, optional
                            Hard sparsity mask: allowed_features[i]==0 forces
                            t[i]==0 (feature i may not move); ==1 leaves it
                            free. When given, the whole problem is solved in
                            the lower-dimensional subspace spanned by the free
                            coordinates only (t restricted to R^k, k = number
                            of 1s), and sparsity/lam are overridden to False/0
                            since the L1 penalty is no longer needed to induce
                            sparsity — the mask already fixes it exactly.
                            Feature restriction only adds LINEAR equalities
                            (p_fixed = mu_A_fixed) on top of the still-purely-
                            quadratic L2 objective, so the exact single-sphere
                            fast paths (radial closed form / S-lemma SDP) for
                            the reverse-convex repair remain exact and are
                            NOT disabled by this mask (unlike sparsity=True,
                            whose L1 term does break them).
        extra_competitors : list of leaf-index lists, optional
                            Additional competitors injected on top of the live
                            structural competitor set, each treated exactly
                            like any other competitor node (its centroid/size
                            feed the same Apollonius construction). Used by
                            diverseCounterfactuals to bias A towards a
                            specific sub-branch of B by adding B's sibling
                            sub-branch as a competitor, without re-targeting
                            the objective away from B ("framing B").

        Returns
        -------
        A_prime : np.ndarray (n_A x d)
        info : dict  t, feasible, optimal, distance_moved, n_competitors,
                    active_balls, phase, plus sparsity/lam echoed back.
        """
        base = self._build_ward_geometry(B_leaves, A_leaves, anchor=anchor,
                                          allowed_features=allowed_features,
                                          sparsity=sparsity, lam=lam, verbose=verbose)
        if base is None:
            raise ValueError("A_leaves and B_leaves resolve to the same node.")

        A_pts, mu_A, mu_B, n_A = base["A_pts"], base["mu_A"], base["mu_B"], base["n_A"]
        free_idx, fixed_idx, E, k_free = base["free_idx"], base["fixed_idx"], base["E"], base["k_free"]
        sparsity, lam, use_sparsity = base["sparsity"], base["lam"], base["use_sparsity"]
        allowed_features = base["allowed_features"]
        d = base["d"]
        w_B = base["w_B"]

        competitor_nodes = dict(base["competitor_nodes"])
        convex_ball_in = list(base["convex_ball_in"])
        convex_half = list(base["convex_half"])
        rev_balls = list(base["rev_balls"])

        if extra_competitors:
            for i, leaves in enumerate(extra_competitors):
                leaves_sorted = sorted(int(l) for l in leaves)
                nid = f"__extra_{i}__"
                competitor_nodes[nid] = leaves_sorted
                self._add_competitor_apollonius(nid, leaves_sorted, mu_A, mu_B, n_A, w_B,
                                                 convex_ball_in, convex_half, rev_balls)

        if verbose:
            print(f"n_A={n_A}(s_A={base['s_A']}) n_B={base['n_B']}(s_B={base['s_B']}) "
                f"competitors={len(competitor_nodes)} sparsity={sparsity} lam={lam}")

        if not competitor_nodes:
            info = ward_info(np.zeros(d), True, True, 0, [], "guard-no-competitors")
            info["sparsity"] = sparsity; info["lam"] = lam if sparsity else 0.0
            info["allowed_features"] = None if allowed_features is None else np.asarray(allowed_features).tolist()
            return A_pts.copy(), info

        if socp_solver is None:
            socp_solver = cp.CLARABEL if hasattr(cp, "CLARABEL") else cp.ECOS

        return self._solve_ward_core(A_pts, mu_A, mu_B, E, free_idx, fixed_idx, k_free,
                                      convex_ball_in, convex_half, rev_balls,
                                      sparsity, lam, use_sparsity, allowed_features,
                                      len(competitor_nodes), d, socp_solver,
                                      exhaustive, index_to_name)

    def _build_ward_geometry(self, B_leaves, A_leaves, anchor='min',
                              allowed_features=None, sparsity=True, lam=1.0,
                              verbose=False):
        """
        Build the part of a Ward branch counterfactual that is identical
        regardless of which extra (sibling) competitors get layered on top:
        resolves A/B, the feature-subspace mask, the base live competitor set
        (live blocks at the anchor step excluding A and B, plus A's own
        unmoved position), and that base set's Apollonius data. This is the
        "build" half of counterfactual_ward_branch, extracted so
        diverseCounterfactuals can compute it once and reuse it at every node
        of B's subtree instead of rebuilding it per node.

        Returns None if A and B resolve to the same node (caller raises).
        """
        n = len(self.X)
        d = self.X.shape[1]

        A_node, A_leaves_sorted, s_A = self._resolve_branch(A_leaves)
        B_node, B_leaves_sorted, s_B = self._resolve_branch(B_leaves)
        if A_node == B_node:
            return None

        A_pts = np.asarray(self.X[A_leaves_sorted], dtype=float)
        B_pts = np.asarray(self.X[B_leaves_sorted], dtype=float)
        n_A, n_B = len(A_pts), len(B_pts)
        mu_A = A_pts.mean(axis=0)
        mu_B = B_pts.mean(axis=0)

        def w(nS):
            return 2.0 * n_A * nS / (n_A + nS)
        w_B = w(n_B)

        # ---- feature mask: restrict t to the subspace of allowed features ----
        free_idx = np.arange(d)
        fixed_idx = np.array([], dtype=int)
        if allowed_features is not None:
            mask = np.asarray(allowed_features).astype(bool).reshape(-1)
            if mask.shape[0] != d:
                raise ValueError(f"allowed_features must have length d={d}, got {mask.shape[0]}.")
            if not mask.any():
                raise ValueError("allowed_features must allow at least one feature to change.")
            free_idx = np.where(mask)[0]
            fixed_idx = np.where(~mask)[0]
            if sparsity:
                if verbose:
                    print("allowed_features given: overriding sparsity=False, lam=0.0 "
                          "(the mask already induces exact sparsity).")
                sparsity = False
            lam = 0.0

        # ---- sparsity mode: False (no L1), True (fixed lam), 'adaptive' (data-driven lam) ----
        if sparsity not in (False, True, 'adaptive'):
            raise ValueError("sparsity must be False, True, or 'adaptive'.")
        if sparsity == 'adaptive':
            t_star = mu_B - mu_A
            norm_t_star = float(np.linalg.norm(t_star))
            lam = (norm_t_star ** 2) / norm_t_star if norm_t_star > 1e-12 else 0.0
        use_sparsity = bool(sparsity)

        # ---- Step 1: anchor + live partition ----
        if anchor != 'min':
            raise NotImplementedError("Only anchor='min' (s* = min(s_A, s_B)) is implemented.")
        s_star = min(s_A, s_B)
        live_partition = self._alive_clusters_before_row(s_star + 1)

        # ---- Step 2: base competitor set ----
        A_set, B_set = set(A_leaves_sorted), set(B_leaves_sorted)
        competitor_nodes = {nid: leaves for nid, leaves in live_partition.items()
                            if not (set(leaves) <= A_set or set(leaves) <= B_set)}

        # Always add A's own unmoved position as a competitor: a counterfactual
        # that leaves A exactly where it already was is not a real
        # counterfactual, even if A happens to already dominate every live
        # structural competitor (e.g. when A is the sibling of the branch the
        # last query moved). This never removes a solution that genuinely
        # requires movement — it only rules out the degenerate t=0 answer.
        competitor_nodes["__A_unmoved__"] = A_leaves_sorted

        # ---- feature-subspace embedding: p = mu_A + E @ t_free, t_free in R^k ----
        # k = len(free_idx) <= d. E is the d x k matrix scattering the free
        # coordinates back into the full space (identity columns at free_idx,
        # zero elsewhere), so t = E @ t_free is automatically zero on fixed_idx.
        k_free = len(free_idx)
        E = np.eye(d)[:, free_idx]

        # ---- Apollonius data for the base competitor set only ----
        convex_ball_in = []   # (kappa, R)         ||p-kappa|| <= R
        convex_half = []      # (a, b)             a·p <= b
        rev_balls = []        # {kappa, R, nid}    ||p-kappa|| >= R  (reverse-convex)
        for nid, leaves in competitor_nodes.items():
            self._add_competitor_apollonius(nid, leaves, mu_A, mu_B, n_A, w_B,
                                             convex_ball_in, convex_half, rev_balls)

        return {
            "A_node": A_node, "A_leaves_sorted": A_leaves_sorted, "s_A": s_A,
            "B_node": B_node, "B_leaves_sorted": B_leaves_sorted, "s_B": s_B,
            "A_pts": A_pts, "B_pts": B_pts, "n_A": n_A, "n_B": n_B,
            "mu_A": mu_A, "mu_B": mu_B, "w_B": w_B,
            "free_idx": free_idx, "fixed_idx": fixed_idx, "E": E, "k_free": k_free,
            "sparsity": sparsity, "lam": lam, "use_sparsity": use_sparsity,
            "allowed_features": allowed_features, "d": d,
            "competitor_nodes": competitor_nodes,
            "convex_ball_in": convex_ball_in, "convex_half": convex_half, "rev_balls": rev_balls,
        }

    def _add_competitor_apollonius(self, nid, leaves, mu_A, mu_B, n_A, w_B,
                                    convex_ball_in, convex_half, rev_balls):
        """
        Compute one competitor's Apollonius constraint (ball/half-space, typed
        by sign(w_B - w(n_X)), exactly the per-competitor construction used in
        counterfactual_ward_branch) and append it in place to the matching
        list. Returns (kind, list_index) so a caller can undo the append later
        via list.pop() -- used by diverseCounterfactuals to accumulate/retract
        sibling competitors during its DFS traversal without rebuilding the
        base geometry.
        """
        mu_X = np.asarray(self.X[leaves], dtype=float).mean(axis=0)
        n_X = len(leaves)

        def w(nS):
            return 2.0 * n_A * nS / (n_A + nS)

        coeff = w_B - w(n_X)
        bvec = w_B * mu_B - w(n_X) * mu_X
        cval = w_B * (mu_B @ mu_B) - w(n_X) * (mu_X @ mu_X)
        if abs(coeff) < 1e-12:
            convex_half.append((-2.0 * bvec, -cval))
            return ("half", len(convex_half) - 1)
        kappa = bvec / coeff
        R = np.sqrt(max(kappa @ kappa - cval / coeff, 0.0))
        if coeff > 0:
            convex_ball_in.append((kappa, R))
            return ("ball_in", len(convex_ball_in) - 1)
        rev_balls.append({"kappa": kappa, "R": R, "nid": nid})
        return ("rev", len(rev_balls) - 1)

    def _solve_ward_core(self, A_pts, mu_A, mu_B, E, free_idx, fixed_idx, k_free,
                          convex_ball_in, convex_half, rev_balls,
                          sparsity, lam, use_sparsity, allowed_features,
                          n_competitors, d, socp_solver, exhaustive, index_to_name):
        """
        Phase 1-4 solve logic of counterfactual_ward_branch, extracted to
        operate on already-built Apollonius data (convex_ball_in/convex_half/
        rev_balls) so it can be reused across nodes of a diversity traversal
        without rebuilding the geometry. Semantics are byte-for-byte
        identical to the inline version this replaces.
        """
        # ---- objective helper (numeric, for scoring / SLSQP) ----
        def obj_val(p):
            t = p - mu_A
            v = float(t @ t)
            if use_sparsity:
                v += lam * float(np.sum(np.abs(t)))
            return v

        # ---- convex projection onto K (SOCP; + L1 term if sparsity) ----
        def project_K_socp():
            t_free = cp.Variable(k_free)
            p = mu_A + E @ t_free
            cons = []
            for (kappa, R) in convex_ball_in:
                cons.append(cp.norm(p - kappa) <= R)
            for (a, b) in convex_half:
                cons.append(a @ p <= b)
            objective = cp.sum_squares(t_free)
            if use_sparsity:
                objective = objective + lam * cp.norm1(t_free)   # elastic net
            prob = cp.Problem(cp.Minimize(objective), cons)
            prob.solve(solver=socp_solver)
            if t_free.value is None:
                return mu_A.copy()
            return mu_A + E @ np.asarray(t_free.value, dtype=float)

        # ---- sphere ∩ K projection ----
        def project_spheres_capK(active_spheres, n_restarts=8):
            """
            L2 objective, single sphere:
                (1) closed-form radial foot if it lies in K (exact);
                (2) else S-lemma SDP (exact); (3) else SLSQP backstop.
            L1 objective (sparsity=True): the radial foot and the S-lemma SDP are
            NOT valid (radial is L2-only; the Shor SDP is not tight with an L1
            term), so we go straight to multistart SLSQP on L2^2+lam*L1.
            Feature restriction (allowed_features), by contrast, only adds
            LINEAR equalities (p_fixed = mu_A_fixed) on top of the same purely
            quadratic objective, so both fast paths stay exact: the radial
            foot is taken on sphere ∩ {p_fixed = mu_A_fixed} (itself a sphere
            in the free coordinates), and the fixed rows are added to the SDP
            as ordinary linear constraints (S-lemma tightness only needs one
            quadratic constraint plus arbitrary linear ones).
            Multiple spheres: multistart SLSQP either way.
            """
            # ---- single sphere, pure L2: exact fast paths (restriction-aware) ----
            if len(active_spheres) == 1 and not use_sparsity:
                kappa, R = active_spheres[0]
                p_cf = project_sphere_closed_form(mu_A, kappa, R, fixed_idx=fixed_idx)
                if p_cf is not None and feasible(p_cf, convex_ball_in, convex_half, rev_balls):
                    return p_cf
                p_sdp = project_single_sphere_sdp(mu_A, kappa, R, convex_ball_in, convex_half,
                                                   fixed_idx=fixed_idx)
                if p_sdp is not None and feasible(p_sdp, convex_ball_in, convex_half, rev_balls):
                    return p_sdp
                # fall through to SLSQP

            # ---- SLSQP on the (possibly L1) objective, over t_free in R^k ----
            # p(t_free) = mu_A + E @ t_free stays exactly on the allowed
            # subspace by construction, so no extra equality constraint on
            # the fixed coordinates is needed.
            def to_p(t_free):
                return mu_A + E @ t_free

            eq_cons = [{'type': 'eq',
                        'fun': (lambda t_free, k=k, R=R: np.linalg.norm(to_p(t_free) - k) - R)}
                    for (k, R) in active_spheres]
            ineq_cons = []
            for (kappa, R) in convex_ball_in:
                ineq_cons.append({'type': 'ineq',
                                'fun': (lambda t_free, k=kappa, R=R: R - np.linalg.norm(to_p(t_free) - k))})
            for (a, b) in convex_half:
                ineq_cons.append({'type': 'ineq',
                                'fun': (lambda t_free, a=a, b=b: b - a @ to_p(t_free))})
            cons = eq_cons + ineq_cons

            rng = np.random.default_rng(0)
            starts = []
            k0, R0 = active_spheres[0]
            dirc_full = mu_A - k0
            dirc = E.T @ dirc_full
            nrm = np.linalg.norm(dirc)
            starts.append((E.T @ k0) + R0 * (dirc / nrm if nrm > 1e-12 else unit(k_free, rng)))
            for _ in range(n_restarts - 1):
                starts.append((E.T @ k0) + R0 * unit(k_free, rng))

            # objective + gradient (in t_free space); L1 uses a smooth-ish
            # subgradient (sign), which SLSQP tolerates in practice with
            # multistart.
            def fobj(t_free):
                return obj_val(to_p(t_free))
            def fjac(t_free):
                t = E @ t_free
                g = 2.0 * t
                if use_sparsity:
                    g = g + lam * np.sign(t)
                return E.T @ g

            best, best_val = None, np.inf
            for s in starts:
                res = minimize(fobj, s, method='SLSQP', jac=fjac,
                            constraints=cons, options={'maxiter': 400, 'ftol': 1e-12})
                if res.success and feasible(to_p(res.x), convex_ball_in, convex_half, rev_balls):
                    val = obj_val(to_p(res.x))
                    if val < best_val:
                        best_val, best = val, to_p(res.x)
            return best

        # convenience for the info dict
        def _finish(p, phase, optimal):
            t = p - mu_A
            info = ward_info(t, True, optimal, n_competitors,
                            active(p, rev_balls, index_to_name), phase)
            info["sparsity"] = sparsity
            info["lam"] = lam if sparsity else 0.0
            info["allowed_features"] = None if allowed_features is None else np.asarray(allowed_features).tolist()
            return A_pts + t, info

        # ---------------- Phase 1 ----------------
        p0 = project_K_socp()
        if feasible(p0, convex_ball_in, convex_half, rev_balls):
            # Phase 1 is exact for both objectives (convex program).
            return _finish(p0, "phase1-convex", True)

        # ---------------- Phase 2: violated reverse-convex balls ----------------
        def violated_at(p):
            return [rb for rb in rev_balls
                    if np.linalg.norm(p - rb['kappa']) < rb['R'] - 1e-7]

        # ---------------- Phase 3: greedy deepest-first repair ----------------
        p = p0.copy()
        seen, cycle = set(), False
        for _ in range(len(rev_balls) + 2):
            viol = violated_at(p)
            if not viol:
                break
            rb = max(viol, key=lambda r: r['R'] - np.linalg.norm(p - r['kappa']))
            if rb['nid'] in seen:
                cycle = True
                break
            seen.add(rb['nid'])
            cand = project_spheres_capK([(rb['kappa'], rb['R'])])
            if cand is None:
                cycle = True
                break
            p = cand

        if not cycle and feasible(p, convex_ball_in, convex_half, rev_balls):
            # exact for L2 (radial/SDP, restriction-aware); heuristic for L1
            # (SLSQP) -> optimal flag reflects it
            return _finish(p, "phase3-greedy", not use_sparsity)

        # ---------------- Phase 4: active-set enumeration ----------------
        kmax = min(d, len(rev_balls))
        best_p, best_val = None, np.inf
        for k in range(1, kmax + 1):
            found_this_k = False
            for combo in combinations(range(len(rev_balls)), k):
                spheres = [(rev_balls[i]['kappa'], rev_balls[i]['R']) for i in combo]
                cand = project_spheres_capK(spheres)
                if cand is not None and feasible(cand, convex_ball_in, convex_half, rev_balls):
                    val = obj_val(cand)                 # score on the SAME objective (L2 or L1)
                    if val < best_val:
                        best_val, best_p = val, cand
                    found_this_k = True
            if found_this_k and not exhaustive:
                break

        if best_p is None:
            best_p = mu_B.copy()                        # guaranteed-feasible fallback

        # optimal only if exhaustive AND pure L2 (L1 sphere repair is heuristic)
        opt = exhaustive and not use_sparsity
        phase = ("phase4-exhaustive" if exhaustive else "phase4-earlystop")
        return _finish(best_p, phase, opt)

    def diverseCounterfactuals(self, B_leaves, A_leaves, max_depth=None,
                                both_children=False, **kwargs):
        """
        Hierarchical diverse counterfactuals for Ward-linkage: descend B's own
        dendrogram subtree. The original "under B" domination constraints
        (A must still dominate B's own live structural competitors, built
        once via _build_ward_geometry) are always kept, so every result
        remains a valid counterfactual for the original (A, B) problem. On
        top of those, at each split a direct sibling-preference constraint
        is accumulated to bias the solution toward a specific child:

            d(A(t), child_target) < d(A(t), child_sibling)

        i.e. "closer to the target child's own centroid than to its
        sibling's own centroid" -- literally the condition for A(t) to land
        on the target child's side of the child_target/child_sibling
        Apollonius bisector. This is NOT the same as (and replaces) simply
        adding the sibling as a competitor against B: since B is the parent
        of both children, "closer to the whole of B than to sibling" can
        hold while A(t) is still on the sibling's own side of the two
        children's bisector -- comparing the two children directly is what
        actually forces the split. It is exactly one more Apollonius
        constraint of the same squared quadratic form _add_competitor_
        apollonius already builds for ordinary competitors (near=target,
        far=competitor); here near=child_target, far=child_sibling instead
        of near=B, far=competitor, using child_target's own centroid/weight
        (not B's) -- the same builder handles both, since its derivation
        never actually required one side to be B.

        Level 0 is the base counterfactual (target B, no sibling-preference
        constraints yet -- the objective itself always stays centered on
        A's own original position with the original B-domination
        constraints; only the accumulated sibling-preference constraints
        change what's added as depth increases). At an internal node with
        children (cL, cR), descending into cL adds d(A,cL) < d(A,cR) on top
        of everything accumulated so far (and symmetrically for cR).
        Feasible regions are nested and shrink with depth (F_0 superset F_1
        superset ...), so once a node is infeasible, its descendants are
        not attempted (pruned).

        Parameters
        ----------
        B_leaves, A_leaves : list of int
            Target branch B and moving branch A, as in counterfactual_ward_branch.
        max_depth : int, optional
            Maximum recursion depth (0 = only the base counterfactual). None
            descends all the way to B's leaves.
        both_children : bool
            If False (default), only descend into the child that the
            *current* solution is NOT already closer to (the sibling not yet
            reflected in the current solution), avoiding a near-duplicate of
            the parent CF. If True, descend into both children at every
            internal node, producing one counterfactual per node of B's
            subtree.
        **kwargs
            Forwarded to counterfactual_ward_branch (e.g. eps, sparsity, lam,
            index_to_name, socp_solver, allowed_features).

        Returns
        -------
        results : list of dict, one per generated counterfactual, each with
            'target_node'   : node id of B's subtree this CF specializes to
            'path'          : list of (chosen_child, sibling) node-id pairs
                              accumulated down B's subtree to reach this node
            'A_prime', 'info' : outputs of counterfactual_ward_branch
            'depth'         : recursion depth (0 = base counterfactual)
        """
        # ---- build the base geometry ONCE; it is identical at every node ----
        base = self._build_ward_geometry(
            B_leaves, A_leaves,
            anchor=kwargs.get('anchor', 'min'),
            allowed_features=kwargs.get('allowed_features'),
            sparsity=kwargs.get('sparsity', True),
            lam=kwargs.get('lam', 1.0),
            verbose=kwargs.get('verbose', False),
        )
        if base is None:
            raise ValueError("A_leaves and B_leaves resolve to the same node.")

        socp_solver = kwargs.get('socp_solver')
        if socp_solver is None:
            socp_solver = cp.CLARABEL if hasattr(cp, "CLARABEL") else cp.ECOS
        exhaustive = kwargs.get('exhaustive', False)
        index_to_name = kwargs.get('index_to_name')

        A_pts, mu_A, mu_B, n_A = base["A_pts"], base["mu_A"], base["mu_B"], base["n_A"]
        free_idx, fixed_idx, E, k_free = base["free_idx"], base["fixed_idx"], base["E"], base["k_free"]
        sparsity, lam, use_sparsity = base["sparsity"], base["lam"], base["use_sparsity"]
        allowed_features, d = base["allowed_features"], base["d"]
        n_base_competitors = len(base["competitor_nodes"])
        B_node = base["B_node"]

        # Apollonius data for the current DFS path: the base lists (built
        # once above, never rebuilt) plus whichever sibling competitors are
        # currently "on the stack" -- appended when descending into a child,
        # popped when backtracking out of it (classic DFS backtracking, kept
        # correct because each recursion level appends to, and later pops
        # from, exactly one of these three lists -- see _add_competitor_apollonius).
        convex_ball_in = list(base["convex_ball_in"])
        convex_half = list(base["convex_half"])
        rev_balls = list(base["rev_balls"])

        results = []
        solve_count = 0

        def solve_here(n_extra):
            nonlocal solve_count
            solve_count += 1
            return self._solve_ward_core(
                A_pts, mu_A, mu_B, E, free_idx, fixed_idx, k_free,
                convex_ball_in, convex_half, rev_balls,
                sparsity, lam, use_sparsity, allowed_features,
                n_base_competitors + n_extra, d, socp_solver, exhaustive, index_to_name)

        def new_constraint_satisfied(kind, idx, p, tol=1e-6):
            # Feasibility test for exactly the ONE newly-accumulated sibling
            # constraint (O(1)), mirroring utilityFunctions.feasible's
            # per-constraint predicate/tolerance for the matching kind.
            if kind == "ball_in":
                kappa, R = convex_ball_in[idx]
                return np.linalg.norm(p - kappa) <= R + tol
            if kind == "half":
                a, b = convex_half[idx]
                return a @ p <= b + tol
            kappa, R = rev_balls[idx]["kappa"], rev_balls[idx]["R"]
            return np.linalg.norm(p - kappa) >= R - tol

        def pop_entry(kind):
            if kind == "ball_in":
                convex_ball_in.pop()
            elif kind == "half":
                convex_half.pop()
            else:
                rev_balls.pop()

        def visit(node, depth, path, n_extra, parent_t, parent_optimal, new_entry):
            if new_entry is None:
                # Root: no parent solution to test/reuse, must solve.
                A_prime, info = solve_here(n_extra)
            else:
                kind, idx = new_entry
                p_parent = mu_A + parent_t
                # Descending adds exactly one new competitor, which only
                # shrinks the feasible region (objective unchanged) -- if the
                # parent's (provably optimal) t still satisfies this one new
                # constraint, it remains the min-norm point of the smaller
                # region, so it is still exactly optimal for this node too.
                if parent_optimal and new_constraint_satisfied(kind, idx, p_parent):
                    t = parent_t
                    p = p_parent
                    info = ward_info(t, True, True, n_base_competitors + n_extra,
                                      active(p, rev_balls, index_to_name), "inherited-parent")
                    info["sparsity"] = sparsity
                    info["lam"] = lam if sparsity else 0.0
                    info["allowed_features"] = (None if allowed_features is None
                                                 else np.asarray(allowed_features).tolist())
                    A_prime = A_pts + t
                else:
                    A_prime, info = solve_here(n_extra)

            results.append({
                "target_node": node,
                "path": list(path),
                "A_prime": A_prime,
                "info": info,
                "depth": depth,
            })

            if not info.get("feasible", False):
                # Nested feasible regions shrink with depth: F_0 superset
                # F_1 superset ... , so an infeasible node makes every
                # descendant on this path infeasible too. Do not descend.
                return
            if max_depth is not None and depth >= max_depth:
                return

            children = get_children(self.Z, self.n, node)
            if children is None:
                # Leaf of B's subtree: nothing left to specialize into.
                return

            c_left, c_right = children
            left_leaves = self.get_leaves(c_left)
            right_leaves = self.get_leaves(c_right)
            t_here, optimal_here = info["t"], info.get("optimal", False)

            def descend(child, child_leaves, sibling_node, sibling_leaves, path_pair):
                # Sibling-preference constraint: d(A, child) < d(A, sibling),
                # built with child's OWN centroid/weight as "near" (not B's)
                # -- see diverseCounterfactuals' docstring for why this,
                # rather than framing the sibling as a competitor against B,
                # is what actually forces the split.
                mu_child = np.asarray(self.X[child_leaves], dtype=float).mean(axis=0)
                n_child = len(child_leaves)
                w_child = 2.0 * n_A * n_child / (n_A + n_child)
                kind, idx = self._add_competitor_apollonius(
                    sibling_node, sibling_leaves, mu_A, mu_child, n_A, w_child,
                    convex_ball_in, convex_half, rev_balls)
                try:
                    visit(child, depth + 1, path + [path_pair], n_extra + 1,
                          t_here, optimal_here, (kind, idx))
                finally:
                    pop_entry(kind)

            if both_children:
                descend(c_left, left_leaves, c_right, right_leaves, (c_left, c_right))
                descend(c_right, right_leaves, c_left, left_leaves, (c_right, c_left))
            else:
                mu_A_prime = np.asarray(A_prime, dtype=float).mean(axis=0)
                d_left = np.linalg.norm(mu_A_prime - np.asarray(self.X[left_leaves], dtype=float).mean(axis=0))
                d_right = np.linalg.norm(mu_A_prime - np.asarray(self.X[right_leaves], dtype=float).mean(axis=0))
                # Descend only into the child the current solution is NOT
                # already closer to, to avoid a redundant near-duplicate.
                if d_left <= d_right:
                    descend(c_right, right_leaves, c_left, left_leaves, (c_right, c_left))
                else:
                    descend(c_left, left_leaves, c_right, right_leaves, (c_left, c_right))

        visit(B_node, 0, [], 0, None, False, None)
        return results

    def counterfactual_complete_linkage_branch(self, B_leaves, A_leaves,
                                            metric='euclidean', anchor='min',
                                            eps=1e-6, index_to_name=None,
                                            big_m=None,
                                            t_bound=1e4, time_limit=60, verbose=False,
                                            sparsity=False, lam=1.0,
                                            allowed_features=None, extra_blocks=None):
        """
        Complete-linkage counterfactual that moves an entire branch A 
        so that it would attach under branch B instead of
        wherever it currently sits, per the "minimize ||t||^2 subject to
        complete-linkage domination constraints" formulation.

        Both A and B may be single leaves or full internal subtrees — a
        single-leaf branch is just the n_A=1 / n_B=1 special case of the same
        machinery, so no separate code path is needed for it.

        Parameters
        ----------
        B_leaves : list of int
            Leaf indices of the target branch B (passed first, as requested).
        A_leaves : list of int
            Leaf indices of the branch to move (A). A + t is applied uniformly
            to every point of A (a single rigid translation for the whole
            branch), never to individual leaves independently.
        X : np.ndarray
            Data matrix (rows = leaves).
        Z : np.ndarray
            SciPy linkage matrix.
        metric : {'euclidean', 'hamming'}
            'euclidean' -> continuous t in R^d, solved as a convex MIQP
            (see counterfactual_complete_linkage_branch_gurobi): the
            witness disjunction is encoded with binaries and handed to the
            branch-and-bound solver directly, instead of being enumerated here.
            'hamming'   -> binary t in {0,1}^d (an XOR bit-flip pattern applied
            to every point of A), solved as an exact 0-1 MILP.
        anchor : {'min'}
            How to pick the anchor step s* at which the live partition/competi-
            tor set is read off. Only s* = min(s_A, s_B) is implemented.
        eps : float
            Strict-domination margin (keeps "closer than every competitor"
            unambiguous and the feasible region closed).
        index_to_name : dict, optional
            index -> label, used only for readable printing/diagnostics.
        big_m : float, optional
            Big-M constant for metric='hamming'; defaults to a value large
            enough to make deactivated constraint rows vacuous.
        t_bound, time_limit, verbose
            Forwarded to counterfactual_complete_linkage_branch_gurobi when
            metric='euclidean' (box bound / big-M source, Gurobi time limit,
            and solver log verbosity respectively). Unused for metric='hamming'.

        Returns
        -------
        A_prime : np.ndarray or None
            The moved branch, A + t* (shape n_A x d), or None if infeasible.
        info : dict
            t (the optimal displacement/bit-flip), feasible, distance moved /
            bits flipped, the live competitor count, and which competitor
            witnesses are active (binding) at the optimum.
        """
        
        

        # ---------------------------------------------------------------
        # Step 0 — resolve the two branches as static wholes.
        # From here on A and B are fixed point arrays: no dynamic reassembly
        # as the dendrogram is walked in later steps.
        # ---------------------------------------------------------------
        A_node, A_leaves_sorted, s_A = self._resolve_branch(A_leaves)
        B_node, B_leaves_sorted, s_B = self._resolve_branch(B_leaves)
        if A_node == B_node:
            raise ValueError("A_leaves and B_leaves resolve to the same node.")

        A_pts = np.asarray(self.X[A_leaves_sorted], dtype=float)
        B_pts = np.asarray(self.X[B_leaves_sorted], dtype=float)
        #n_A, n_B = len(A_pts), len(B_pts)

        #def _label(leaves):
        #   return [index_to_name[i] for i in leaves] if index_to_name is not None else leaves

        # ---------------------------------------------------------------
        # Step 1 — anchor step s* = min(s_A, s_B), and the live partition L(s*)
        # obtained by replaying Z rows 0..s* (a union-find over the first
        # s*+1 merges). _alive_clusters_before_row(Z, n, cutoff) applies rows
        # 0..cutoff-1, so cutoff = s*+1 replays exactly rows 0..s* inclusive.
        # If s* == -1 (the anchor is a bare leaf), zero rows are replayed and
        # L(s*) is just the set of singleton leaves, which is correct.
        # ---------------------------------------------------------------
        if anchor != 'min':
            raise NotImplementedError("Only anchor='min' (s* = min(s_A, s_B)) is implemented.")
        s_star = min(s_A, s_B)
        live_partition:Dict[int, Iterable[int]] = self._alive_clusters_before_row(s_star + 1)
        # Returns a dictionary having as keys the id of the nodes and as values the leaves of each node




        # ---------------------------------------------------------------
        # Step 2 — competitor set C: every live block disjoint from BOTH A
        # and B. Because L(s*) is a genuine partition of the leaves, each live
        # block is wholly inside A, wholly inside B, or disjoint from both —
        # never partially overlapping — so this containment test is
        # unambiguous. Blocks inside A are the mover itself; blocks inside B
        # are dropped because B always enters whole, never as fragments.
        # ---------------------------------------------------------------
        A_set, B_set = set(A_leaves_sorted), set(B_leaves_sorted)
        competitor_nodes:Dict[int, Iterable[int]] = {}
        for node_id, leaves in live_partition.items():
            leaf_set = set(leaves)
            if leaf_set <= A_set or leaf_set <= B_set:
                continue
            competitor_nodes[node_id] = leaves

        # Always add A's own unmoved position as a competitor: a counterfactual
        # that leaves A exactly where it already was is not a real
        # counterfactual, even if A happens to already dominate every live
        # structural competitor (e.g. when A is the sibling of the branch the
        # last query moved). This never removes a solution that genuinely
        # requires movement — it only rules out the degenerate t=0 answer.
        competitor_nodes["__A_unmoved__"] = A_leaves_sorted

        #print(f"Branch A                  : {_label(A_leaves_sorted)} (n_A={n_A}, step s_A={s_A})")
        #print(f"Branch B (target)         : {_label(B_leaves_sorted)} (n_B={n_B}, step s_B={s_B})")
        #print(f"Anchor step s*            : {s_star}")
        #print(f"Live competitors at s*    : {len(competitor_nodes)} clusters, "
        #     f"sizes {[len(v) for v in competitor_nodes.values()]}")

        # Guard 1: no competitors at all -> every t is feasible, nothing to beat.
        if not competitor_nodes:
            if verbose:
                print("No live competitors at s* — every t is feasible; returning t*=0.")
            return A_pts.copy(), {
                "t": np.zeros(self.d), "feasible": True, "distance_moved": 0.0,
                "n_competitors": 0, "already_optimal": True,
            }

        # Guard 2: A already dominates B against every competitor at t=0.
        dAB0:float = complete_linkage_max(A_pts, B_pts, metric) # distance between A and B
        dAX0:Dict[int, float] = {nid: complete_linkage_max(A_pts, np.asarray(self.X[leaves], dtype=float), metric)
                for nid, leaves in competitor_nodes.items()} # Distance between A and every competitor X_0
        if all(dAB0 < dx - eps for dx in dAX0.values()):
            if verbose:
                print("A already dominates B against every competitor at t=0; returning t*=0.")
            return A_pts.copy(), {
                "t": np.zeros(self.d), "feasible": True, "distance_moved": 0.0,
                "n_competitors": len(competitor_nodes), "already_optimal": True,
            }

        # ---------------------------------------------------------------
        # Step 3/4 — build the difference sets and solve the program. Same
        # skeleton for both metrics: minimize ||t||^2 subject to, for each
        # competitor, some witness q in Q_X = V_AX dominating every p in
        # P = V_AB. Only the variable type of t and how the domination row
        # linearizes differ between the two metrics, so only the dispatch
        # differs here; each branch builds its own P/Q_X internally.
        # ---------------------------------------------------------------
        if metric == 'euclidean':
            # Delegate to the convex-MIQP solver: same Step 0-5 program,
            # but the disjunction is handed to the B&B as binaries instead
            # of being enumerated combinatorially, so it scales to far more
            # competitors/witnesses than a witness-assignment QP enumeration.
            A_prime, info = self.counterfactual_complete_linkage_branch_gurobi(
                B_leaves, A_leaves, anchor=anchor, eps=eps,
                index_to_name=index_to_name, t_bound=t_bound, time_limit=time_limit,
                verbose=verbose, sparsity=sparsity, lam=lam,
                allowed_features=allowed_features, extra_blocks=extra_blocks)
        elif metric == 'hamming':
            A_prime, info = self._solve_branch_hamming(
                A_pts, B_pts, competitor_nodes, eps, big_m, index_to_name)
        else:
            raise ValueError(f"Unsupported metric: {metric!r}; use 'euclidean' or 'hamming'.")

        # ---------------------------------------------------------------
        # Step 5 — output and checks.
        # ---------------------------------------------------------------
        if not info.get("feasible", False):
            if verbose:
                print("\nFeasible                  : False")
                print("No t satisfies the complete-linkage margin against every live "
                    "competitor — the counterfactual region can be genuinely empty "
                    "for complete linkage (a spatially large/spread B is not "
                    "necessarily dominatable).")
            info["n_competitors"] = len(competitor_nodes)
            return None, info

        moved_norm = info.get("distance_moved", info.get("bits_flipped"))
        if verbose:
            print(f"\nFeasible                  : True")
        #print(f"||t*|| / bits flipped     : {moved_norm}")
        #print(f"Active (binding) witnesses: {info.get('binding_witnesses')}")

        info["n_competitors"] = len(competitor_nodes)
        return A_prime, info

    def _resolve_branch(self, leaves):
        """
        Resolve a set of leaf indices to a *static* branch: its node id in Z,
        its (sorted) leaf list, and its formation step (the row index in Z at
        which it was created by a merge; -1 for a single leaf, which exists
        before any merge / "pre-clustering").

        Used by counterfactual_complete_linkage_branch to turn either a single
        leaf or a full subtree into the same (node_id, leaves, step) shape, so
        the rest of the algorithm never has to special-case "is this a leaf or
        a branch" again — from here on A and B are just fixed point sets.
        """
        leaves_sorted = sorted(int(l) for l in leaves)
        if len(leaves_sorted) == 1:
            # A single leaf is "formed" before any merge happens.
            return leaves_sorted[0], leaves_sorted, -1
        node_id = self.find_node_for_leaves(leaves_sorted)
        if node_id is None:
            raise ValueError(
                "leaves do not form a clean subtree in Z — pass a leaf set that "
                "matches an actual internal node's full leaf set."
            )
        return node_id, leaves_sorted, node_id - self.n


    def _alive_clusters_before_row(self, cutoff_row):
        """
        Union-find over the first `cutoff_row` merges of Z.

        Returns {node_id: [leaf indices]} for every cluster that is still
        "alive" (i.e. has not yet been merged into something else) immediately
        before row `cutoff_row` is applied.
        """
        node_leaves = {i: [i] for i in range(self.n)}
        alive = set(range(self.n))
        for i in range(cutoff_row):
            left, right = int(self.Z[i, 0]), int(self.Z[i, 1])
            node_id = self.n + i
            node_leaves[node_id] = node_leaves[left] + node_leaves[right]
            alive.discard(left)
            alive.discard(right)
            alive.add(node_id)
        return {node_id: node_leaves[node_id] for node_id in alive}


    # ------------------------------------------------------------------
    # Complete-linkage Euclidean solver: witness-disjunction blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _cl_block(key, P_pts, Q_pts, q_labels):
        """
        One witness-disjunction block for the complete-linkage MIQP.

        A block encodes a single complete-linkage domination requirement
        "d(A(t), near) < d(A(t), far)", in the form that complete linkage's
        max-over-pairs forces:

            EXISTS a witness q in V_{A,far} such that
                ||q + t|| >= ||p + t|| + margin   for ALL p in V_{A,near}.

        That existential is exact, not merely sufficient: if the far side's
        max exceeds the near side's, the far side's own argmax witnesses it;
        conversely any such q makes max_far >= max_near. So one block per
        requirement reproduces the requirement exactly.

        Keeping `P` *per block* (rather than one P shared by every block, as
        the original enumeration-era code assumed) is what lets the diversity
        descent reuse this machinery: a base competitor block has
        near = B, far = competitor, while a sibling-preference block has
        near = the target child, far = its sibling -- same algebra, different
        near-set. See counterfactual_complete_linkage_branch_gurobi.

        Parameters
        ----------
        key : hashable
            Block identifier, used in Gurobi constraint names and to report
            which blocks are binding (a competitor node id, or e.g.
            ('sibling', child, sibling)).
        P_pts : (m, d) array
            Difference set V_{A,near}: every p that the chosen witness must
            out-distance.
        Q_pts : (w, d) array
            Difference set V_{A,far}: the candidate witnesses.
        q_labels : list of length w
            Per-witness provenance (leaf index or label), for diagnostics.
        """
        P_pts = np.asarray(P_pts, dtype=float)
        Q_pts = np.asarray(Q_pts, dtype=float)
        return {
            "key": key,
            "P": P_pts,
            "pp": np.einsum('ij,ij->i', P_pts, P_pts),
            "Q": Q_pts,
            "qq": np.einsum('ij,ij->i', Q_pts, Q_pts),
            "labels": list(q_labels),
        }

    def _cl_pair_blocks(self, A_pts, near_pts, far_nodes):
        """
        Build one _cl_block per entry of `far_nodes` (a {key: leaves} dict),
        all sharing the same near-set `near_pts`. This is the base
        "A must stay closer to near than to every competitor" requirement.
        """
        P = (A_pts[:, None, :] - np.asarray(near_pts, dtype=float)[None, :, :]
             ).reshape(-1, A_pts.shape[1])
        blocks = []
        for key, leaves in far_nodes.items():
            C = np.asarray(self.X[leaves], dtype=float)
            Q = (A_pts[:, None, :] - C[None, :, :]).reshape(-1, A_pts.shape[1])
            labels = [leaf for _a in range(len(A_pts)) for leaf in leaves]
            blocks.append(self._cl_block(key, P, Q, labels))
        return blocks

    def counterfactual_complete_linkage_branch_gurobi(
        self, B_leaves, A_leaves,
        metric='euclidean', anchor='min', eps=1e-6,
        index_to_name=None, t_bound=1e4, time_limit=60, verbose=False,
        sparsity=False, lam=1.0, allowed_features=None,
        extra_blocks=None, max_rounds=64, rows_per_round=64):
        """
        Branch-aware complete-linkage counterfactual, solved as a *convex*
        MIQP with Gurobi plus constraint row generation.

        Formulation
        -----------
        The complete-linkage domination inequality ||q + t|| >= ||p + t|| is
        SQUARED, which cancels ||t||^2 and leaves a LINEAR row in t:

            2 <p - q, t>  <=  ||q||^2 - ||p||^2 - eps.

        So the objective stays convex quadratic and every constraint is
        linear -> a convex MIQP, which branch-and-bound solves to certified
        GLOBAL optimality. This is why the squared-linearised form is kept
        rather than an epigraph form on d(A,B): the epigraph version makes
        the row count additive instead of multiplicative, but turns the
        witness rows reverse-convex and forfeits exactly that guarantee.

        The price of linearisation is that the full model has one row per
        (block, witness, p) triple, i.e. |Q| * |P| rows per block. That is
        multiplicative and can reach millions of rows on wide branches
        (n_A enters squared, since it sizes both P and Q), so the rows are
        generated LAZILY:

          1. seed one row per (block, witness) -- the p of largest norm,
             the most likely binding one at t ~ 0;
          2. solve;
          3. for each block, if no selected witness satisfies all of its
             rows, add the violated rows of the least-violated selected
             witness;
          4. repeat until no block is violated.

        On termination with nothing violated the solution is feasible for
        the FULL model, and since the generated model is a relaxation of it
        (a subset of its rows), the relaxation's optimum attained at a
        full-model-feasible point IS the full-model optimum. Global
        optimality therefore survives row generation. If `max_rounds` is
        exhausted with violations outstanding, info['optimal'] is False and
        info['rowgen_converged'] is False.

        Both A and B may be single leaves or full subtrees; A is translated
        rigidly by one shared t, which is why domination must hold against
        the FULL difference set V_AB (every a in A against every b in B),
        not a per-point slice.

        Parameters
        ----------
        B_leaves, A_leaves : list of int
            Target branch B and the branch to move, as elsewhere.
        metric : {'euclidean'}
            Euclidean only; the squared-domination trick is L2 specific.
            Hamming has its own exact MILP (_solve_branch_hamming).
        anchor : {'min'}
            Competitor anchor step s* = min(s_A, s_B).
        eps : float
            Strict-domination margin. NOTE: this margin is on SQUARED
            distances here, whereas metrics.validity / _dominates apply eps
            to UNSQUARED distances. The two are not interchangeable; the
            returned info carries `margin_unsquared` from an independent
            _dominates re-check so a caller can revalidate on the same
            footing it uses for every other method.
        t_bound : float
            Box bound |t_j| <= t_bound; also the source of a valid finite
            big-M. Hitting the box signals a possibly unbounded region.
        time_limit : float
            Gurobi time limit in seconds, applied PER ROUND.
        sparsity : bool
            If True, minimise ||t||^2 + lam * ||t||_1 (elastic net), matching
            the Ward solver's sparsity mode. Still a convex MIQP.
        lam : float
            L1 weight when sparsity=True.
        allowed_features : array of bool, optional
            Per-feature mask; disallowed coordinates are pinned to t_j = 0.
        extra_blocks : list, optional
            Additional witness-disjunction blocks from _cl_block, appended to
            the base competitor blocks. This is the hook the complete-linkage
            diversity descent uses to add sibling-preference constraints
            (near = target child, far = its sibling) without touching this
            method.
        max_rounds : int
            Cap on row-generation rounds.
        rows_per_round : int
            Cap on rows added per violated block per round, so one
            pathological block cannot blow the model up in a single round.

        Returns
        -------
        A_prime : np.ndarray or None
            A + t*, or None if proven infeasible / no incumbent found.
        info : dict
            t, feasible, optimal, distance_moved, n_competitors,
            binding_witnesses, gurobi_status, phase, rowgen rounds/rows,
            rowgen_converged, revalidated, margin_unsquared.
        """
        if metric != 'euclidean':
            raise ValueError("This solver is Euclidean-only; use the Hamming MILP for binary data.")
        if anchor != 'min':
            raise NotImplementedError("Only anchor='min' (s* = min(s_A, s_B)) is implemented.")

        def _say(*a):
            if verbose:
                print(*a)

        # ---- Step 0: resolve A and B as static wholes ----
        A_node, A_leaves_sorted, s_A = self._resolve_branch(A_leaves)
        B_node, B_leaves_sorted, s_B = self._resolve_branch(B_leaves)
        if A_node == B_node:
            raise ValueError("A_leaves and B_leaves resolve to the same node.")

        A_pts = np.asarray(self.X[A_leaves_sorted], dtype=float)
        B_pts = np.asarray(self.X[B_leaves_sorted], dtype=float)
        n_A, d = A_pts.shape

        # ---- Step 1/2: anchor step, live partition, competitor set ----
        s_star = min(s_A, s_B)
        live_partition = self._alive_clusters_before_row(s_star + 1)
        A_set, B_set = set(A_leaves_sorted), set(B_leaves_sorted)
        competitor_nodes = {}
        for node_id, leaves in live_partition.items():
            leaf_set = set(leaves)
            if leaf_set <= A_set or leaf_set <= B_set:
                continue
            competitor_nodes[node_id] = leaves

        # A's OWN UNMOVED POSITION is a competitor, exactly as in
        # _build_ward_geometry and in the complete-linkage wrapper. It must be
        # added HERE and not left to the wrapper: this method rebuilds the
        # competitor set from the live partition itself, so a caller's copy of
        # it is not consulted and the constraint would simply vanish.
        #
        # Without it the model permits a "counterfactual" that barely moves --
        # it can edge out every real competitor while still being closer to
        # where A already was, which metrics.validity (via its A_orig
        # argument) correctly rejects. Observed on wine pair 1: the solver
        # returned a certified-optimal t with ||t|| = 2.84, SMALLER than the
        # competitor-ignoring MEB baseline's 5.32, and independent
        # revalidation marked it invalid. The symptom is distinctive --
        # solver_optimal=True together with valid=False and a suspiciously
        # small displacement.
        competitor_nodes["__A_unmoved__"] = A_leaves_sorted

        extra_blocks = list(extra_blocks or [])

        def _zero_info(extra):
            info = {"t": np.zeros(d), "feasible": True, "optimal": True,
                    "distance_moved": 0.0, "n_competitors": len(competitor_nodes),
                    "already_optimal": True, "phase": "trivial",
                    "rowgen_rounds": 0, "rowgen_rows": 0,
                    "rowgen_converged": True, "binding_witnesses": []}
            info.update(extra)
            return info

        # Guard 1 -- nothing to beat. Only valid when there are no extra
        # blocks either: a sibling-preference block still constrains t even
        # when the competitor set is empty.
        if not competitor_nodes and not extra_blocks:
            _say("No live competitors at s* -- every t is feasible; returning t*=0.")
            return A_pts.copy(), _zero_info({})

        # Guard 2 -- A already dominates at t=0 (again only without extra blocks).
        if not extra_blocks:
            dAB0 = complete_linkage_max(A_pts, B_pts, metric)
            if all(dAB0 < complete_linkage_max(A_pts, np.asarray(self.X[lv], dtype=float),
                                                metric) - eps
                   for lv in competitor_nodes.values()):
                _say("A already dominates B against every competitor at t=0; returning t*=0.")
                return A_pts.copy(), _zero_info({})

        # ---- Step 3: blocks ----
        blocks = self._cl_pair_blocks(A_pts, B_pts, competitor_nodes) + extra_blocks
        if not blocks:
            return A_pts.copy(), _zero_info({})

        # ---- Step 4: model ----
        model = gp.Model("cl_counterfactual_branch")
        model.Params.OutputFlag = 1 if verbose else 0
        model.Params.TimeLimit = time_limit
        # Indicator constraints remove the big-M leakage, but a tight
        # integrality tolerance costs nothing and guards the z > 0.5 reads.
        model.Params.IntFeasTol = 1e-9

        t = model.addVars(d, lb=-t_bound, ub=t_bound, name="t")
        if allowed_features is not None:
            mask = np.asarray(allowed_features, dtype=bool).ravel()
            if mask.shape[0] != d:
                raise ValueError(f"allowed_features must have length d={d}, got {mask.shape[0]}.")
            if not mask.any():
                raise ValueError("allowed_features must allow at least one feature to change.")
            for j in range(d):
                if not mask[j]:
                    t[j].lb = 0.0
                    t[j].ub = 0.0

        obj = gp.quicksum(t[j] * t[j] for j in range(d))
        if sparsity:
            # |t_j| via u_j >= +-t_j; minimised, so u_j tightens to |t_j|.
            u = model.addVars(d, lb=0.0, ub=t_bound, name="u")
            for j in range(d):
                model.addConstr(u[j] >= t[j], name=f"abs_pos_{j}")
                model.addConstr(u[j] >= -t[j], name=f"abs_neg_{j}")
            obj = obj + lam * gp.quicksum(u[j] for j in range(d))
        model.setObjective(obj, GRB.MINIMIZE)

        # one binary per (block, witness) + the existential per block
        zs = []
        for bi, blk in enumerate(blocks):
            zb = model.addVars(len(blk["Q"]), vtype=GRB.BINARY, name=f"z_{bi}")
            model.addConstr(gp.quicksum(zb[wi] for wi in range(len(blk["Q"]))) >= 1,
                            name=f"pick_{bi}")
            zs.append(zb)

        added = set()

        def add_row(bi, wi, pi):
            if (bi, wi, pi) in added:
                return False
            blk = blocks[bi]
            p, q = blk["P"][pi], blk["Q"][wi]
            rhs = float(blk["qq"][wi]) - float(blk["pp"][pi]) - eps
            coeff = 2.0 * (p - q)
            # INDICATOR, not big-M: "z = 1  =>  2<p-q,t> <= rhs".
            #
            # A big-M encoding (rhs + M(1-z), with M derived from the t-box)
            # is numerically unsafe here. t_bound is large (1e4 by default)
            # while the rows have scale ~1 on standardised data, so
            # M ~ 2*||p-q||_1*t_bound reaches 1e5-1e6; Gurobi's integrality
            # tolerance then lets z = 1 - 1e-5 pass as "selected", leaving
            # M*1e-5 ~ 1-10 of slack on a unit-scale constraint. The solver
            # returns a point that satisfies the model as written but
            # violates true complete-linkage domination (observed: a -0.075
            # unsquared margin, and an objective 8% BELOW the dense
            # full-model optimum, i.e. a spuriously cheap "solution").
            # Indicator constraints are enforced by branching rather than by
            # a magic constant, so the leakage class disappears entirely and
            # t_bound reverts to being only a sanity box on t.
            model.addGenConstrIndicator(
                zs[bi][wi], True,
                gp.quicksum(coeff[j] * t[j] for j in range(d)) <= rhs,
                name=f"dom_b{bi}_w{wi}_p{pi}")
            added.add((bi, wi, pi))
            return True

        # seed: for each (block, witness), the largest-norm p
        for bi, blk in enumerate(blocks):
            pi0 = int(np.argmax(blk["pp"]))
            for wi in range(len(blk["Q"])):
                add_row(bi, wi, pi0)

        # ---- row generation ----
        TOL = 1e-7
        stalled_blocks = []
        status = None
        rounds = 0
        converged = False
        for rounds in range(1, max_rounds + 1):
            model.optimize()
            status = model.Status
            if status == GRB.INFEASIBLE:
                # a relaxation being infeasible proves the full model is too
                _say("\nFeasible: False (relaxation infeasible => full model infeasible)")
                return None, {"feasible": False, "optimal": True,
                              "n_competitors": len(competitor_nodes),
                              "gurobi_status": status, "phase": "miqp-rowgen",
                              "rowgen_rounds": rounds, "rowgen_rows": len(added),
                              "rowgen_converged": True}
            if model.SolCount == 0:
                _say("\nFeasible: unknown (no incumbent within time limit)")
                return None, {"feasible": False, "optimal": False,
                              "n_competitors": len(competitor_nodes),
                              "gurobi_status": status, "phase": "miqp-rowgen",
                              "rowgen_rounds": rounds, "rowgen_rows": len(added),
                              "rowgen_converged": False}

            t_val = np.array([t[j].X for j in range(d)])
            any_added = False
            stalled_blocks = []
            for bi, blk in enumerate(blocks):
                sel = [wi for wi in range(len(blk["Q"])) if zs[bi][wi].X > 0.5]
                if not sel:
                    continue
                viol_by_w, satisfied = {}, False
                for wi in sel:
                    # rows for this witness: 2<p-q,t> - (qq_w - pp - eps) <= 0
                    lhs = 2.0 * ((blk["P"] - blk["Q"][wi]) @ t_val)
                    slack = (float(blk["qq"][wi]) - blk["pp"] - eps) - lhs
                    bad = np.nonzero(slack < -TOL)[0]
                    if bad.size == 0:
                        satisfied = True
                        break
                    viol_by_w[wi] = bad
                if satisfied:
                    continue
                # push the closest-to-valid selected witness
                wi_best = min(viol_by_w, key=lambda w: viol_by_w[w].size)
                worst = viol_by_w[wi_best]
                # most-violated rows first
                lhs = 2.0 * ((blk["P"] - blk["Q"][wi_best]) @ t_val)
                slack = (float(blk["qq"][wi_best]) - blk["pp"] - eps) - lhs
                for pi in worst[np.argsort(slack[worst])][:rows_per_round]:
                    any_added = add_row(bi, wi_best, int(pi)) or any_added
                stalled_blocks.append(blk["key"])
            if not any_added:
                # Distinguish "nothing left to violate" (true convergence)
                # from "violated, but every offending row is ALREADY in the
                # model" -- the latter means the solver returned a point that
                # breaks a constraint it was given, i.e. a numerical failure
                # rather than a modelling gap, and adding rows cannot fix it.
                # Silently reporting convergence there is exactly how the
                # big-M leakage above went undetected, so it is surfaced.
                converged = not stalled_blocks
                break

        # ---- Step 5: read out ----
        t_val = np.array([t[j].X for j in range(d)])
        A_prime = A_pts + t_val
        dist_moved = float(np.linalg.norm(t_val))
        is_optimal = (status == GRB.OPTIMAL) and converged

        binding = sorted({
            (index_to_name[lab] if (index_to_name is not None
                                     and lab in index_to_name) else lab)
            for bi, blk in enumerate(blocks)
            for wi in range(len(blk["Q"]))
            if zs[bi][wi].X > 0.5
            for lab in [blk["labels"][wi]]
        }, key=str)

        if np.any(np.abs(np.abs(t_val) - t_bound) < 1e-6):
            _say("NOTE: |t_j| hit t_bound -- region may be unbounded; raise t_bound.")

        # Independent re-check on UNSQUARED distances, the same criterion
        # metrics.validity uses, since this solver's eps is a squared margin.
        comp_pts = [np.asarray(self.X[lv], dtype=float) for lv in competitor_nodes.values()]
        if comp_pts:
            revalid, margin_unsq = _dominates(A_prime, B_pts, comp_pts, 'complete',
                                               0.0, return_margin=True)
        else:
            revalid, margin_unsq = True, float('inf')

        _say(f"\nFeasible: True | globally optimal: {is_optimal} "
             f"(status {status}, {rounds} rowgen rounds, {len(added)} rows)")
        _say(f"||t*||: {dist_moved:.6f} | binding witnesses: {binding}")

        info = {
            "t": t_val,
            "feasible": True,
            "optimal": is_optimal,
            "distance_moved": dist_moved,
            "n_competitors": len(competitor_nodes),
            "binding_witnesses": binding,
            "gurobi_status": status,
            "phase": "miqp-rowgen",
            "rowgen_rounds": rounds,
            "rowgen_rows": len(added),
            "rowgen_converged": converged,
            "rowgen_stalled_blocks": len(stalled_blocks) if not converged else 0,
            "revalidated": bool(revalid),
            "margin_unsquared": float(margin_unsq),
            "sparsity": bool(sparsity),
            "lam": float(lam) if sparsity else 0.0,
        }
        return A_prime, info


    def _solve_branch_hamming(self, A_pts, B_pts, competitor_nodes, eps, big_m, index_to_name):
        """
        Complete-linkage Hamming counterfactual under the RIGID RELABELING move
        (binary OR categorical), consistent with the convention or rigid translation.
    
        Move
        ----
        On each coordinate j a fixed function f_j is applied to every point of A:
        each source value alpha (a value A takes on j) maps to one target value
        beta. No bijection is required (two A-values may collapse onto one target).
        Binary XOR is the special case where the alphabet is {0,1} and f_j is the
        swap; categorical features (alphabet > 2) are handled the same way. This
        preserves the property that keeps A a valid node — equal A-values stay
        equal — because f_j is a well-defined function (same input -> same output).
    
        NOTE: allowing collapses means the move is a general relabeling, not a
        strict isometry (a collapse can merge two distinct A-values, shrinking A's
        internal Hamming distances). If in the future we want to add a true isometry, 
        we can add a per-source "no two sources share a target" (bijection) constraint; 
        not imposed now.
    
        Formulation
        -----------
        Distance is affine, i.e. linear + constant,  in the relabeling indicators y[j, alpha, beta] = 1 iff
        f_j(alpha) = beta:
            d_H(f(a), p) = sum_j 1[f_j(a_j) != p_j] = sum_j (1 - y[j, a_j, p_j]).
        Complete-linkage domination "for competitor X some witness (a_i, c) beats
        every A-B pair (a', b)" becomes, when the witness is selected (z = 1):
            sum_j y[j, a'_j, b_j] - sum_j y[j, a_i_j, c_j] >= eps   for all (a', b),
        deactivated by big-M when z = 0.
    
        Objective
        ---------
        Minimize the number of coordinates actually relabeled (f_j != identity),
        via auxiliary binaries moved_j. Reduces EXACTLY to "bits flipped" for
        binary data. Not also that this way sparsity is enforced naturally as minimizing
        the length of the re-labeling vector equals minimizing the number of variables remapped.
    
        Targets per coordinate are restricted to the values appearing in A or B on
        that coordinate: mapping an A-value to anything B never takes can never
        create a match and is never useful, this is because we
        are trying to to get closer to B and if we take a value that no element of B ever takes
        we are going further away from B which is never helpful. Competitor-only values likewise
        need not be targets, as we wcan get (a witness distance term to such a value is simply the
        "no match" case, contributing its full 1). Identity (beta = alpha) is
        always available (alpha is an A-value, hence a target), so "keep as-is" is
        always an option.
    
        Returns
        -------
        A_prime : np.ndarray (n_A x d) or None
            Moved branch f(A), or None if infeasible (complete linkage can be).
        info : dict
            t = {j: {alpha: beta}} per-coordinate relabeling map,
            coords_changed / distance_moved = number of coordinates relabeled,
            binding_witnesses = competitor leaves whose witness binary is active.
        """
        A_pts = np.asarray(A_pts)
        B_pts = np.asarray(B_pts)
        n_A, d = A_pts.shape
        n_B = len(B_pts)
    
        # value lists (native python scalars) per coordinate
        Acols = [A_pts[:, j].tolist() for j in range(d)]
        Bcols = [B_pts[:, j].tolist() for j in range(d)]
        src_vals = [sorted(set(Acols[j])) for j in range(d)]                 # sources: A's values
        tgt_vals = [sorted(set(Acols[j]) | set(Bcols[j])) for j in range(d)]  # targets: A ∪ B values
    
        # ---- variable layout: y[j,alpha,beta], then moved_j, then z[nid,a_i,c_leaf] ----
        y_col, col = {}, 0 #look-up dictionary that tells us where the variable y[j, a, b] lives
        # in the flat vector (they start at j and they go from col to col+j)
        for j in range(d):
            for a in src_vals[j]:
                for b in tgt_vals[j]:
                    y_col[(j, a, b)] = col
                    col += 1
        moved_col = {j: col + j for j in range(d)}
        col += d
        # competitor part, z[(nid,ai,c_leaf)]=1 means "the A-point a_i paired with competitor point c_leaf is 
        # selected as the witness that dominates B for competitor nid." 
        # The sum_witnesses z >= 1 constraint (per competitor) forces at least one on, 
        # and the big-M domination rows enforce that the selected witness actually 
        # beats every A–B pair.
        competitor_ids = list(competitor_nodes.keys())
        z_col = {}
        for nid in competitor_ids:
            for a_i in range(n_A):
                for c_leaf in competitor_nodes[nid]:
                    z_col[(nid, a_i, c_leaf)] = col
                    col += 1
        n_vars = col
    
        # big-M: the domination LHS lies in [-d, d], so deactivation at z = 0
        # needs eps - M <= -d, i.e. M >= d + eps. Note how SMALL this is --
        # tens, against integer-valued rows -- compared with the 1e5-1e6 the
        # Euclidean solver's t-box produced, where Gurobi's integrality
        # tolerance leaked M*1e-5 of slack and silently returned INVALID
        # solutions. At this scale the same leakage is ~1e-5 against rows that
        # move in unit steps, so it cannot flip a feasible row.
        # Hamming distances are INTEGERS, so "strictly greater" means a gap of
        # at least 1 and a margin of 0.5 is exactly equivalent to the caller's
        # tiny eps -- while being vastly safer numerically. With M ~ d and
        # HiGHS's integrality tolerance ~1e-5, a big-M row leaks ~2e-4 of
        # slack, which is enough to let a TIE (gap 0) pass a row demanding
        # gap >= 1e-6. That is the same leakage class that silently produced
        # invalid Euclidean solutions; here it made row generation stall on a
        # row already in the model and (before this) mis-report the pair as
        # infeasible. 0.5 sits ~2500x above the leakage, so ties cannot pass.
        eps_int = max(float(eps), 0.5)
        M = big_m if big_m is not None else d + eps_int + 1.0

        c_obj = np.zeros(n_vars)
        #c_obj is the vector of coefficients that we will give in input to scipy's milp
        for j in range(d):
            c_obj[moved_col[j]] = 1.0  # minimize number of coordinates relabeled, i.e. small move and sparisty coincide
            # every variable has a weight of 1
    
        # Sparse (row, col, val) triplets instead of dense rows: each row
        # touches ~d+2 of n_vars columns, so a dense matrix wasted virtually
        # all its memory and was the binding limit on dataset size.
        rr, cc, vv, lb, ub = [], [], [], [], []
        _nrow = [0]

        def _row(entries, lo, hi):
            r = _nrow[0]
            for cidx, val in entries:
                rr.append(r); cc.append(cidx); vv.append(val)
            lb.append(lo); ub.append(hi)
            _nrow[0] += 1
    
        # (1) totality: each (j, alpha) maps to exactly one beta
        for j in range(d):
            for a in src_vals[j]:
                _row([(y_col[(j, a, b)], 1.0) for b in tgt_vals[j]], 1.0, 1.0)
                # so basically we have sum_j y[j, a, b] = 1 which means that each value of alpha 
                # is remapped to exactly one beta (which could be alpha itself)
    
        # (2) moved_j detection: moved_j + y[j,alpha,alpha] >= 1
        #     (if any source alpha is remapped to something else, the coordinate is touched)

        # Here we are creating the constraint moved_j + y[j, alpha, alpha] >= 1 which forces 
        # moved_j to be 1 where the variable j is relabelled.
        # Indeed if y[j, alpha, alpha] = 1, i.e. the alpha was kept as itself, the constraint becomes 
        # moved_j >= 0, nothing is enforced.
        # If instead y[j, alpha, alpha] is 0, i.e. alpha was remapped into something else, the constraint
        # vecomes moved_j >=1, which enforces moved_j signal that j was relabelled
        for j in range(d):
            for a in src_vals[j]:
                _row([(moved_col[j], 1.0), (y_col[(j, a, a)], 1.0)],
                     1.0, np.inf)
    
        # (3) at least one witness selected per competitor
        for nid in competitor_ids:
            _row([(z_col[(nid, a_i, c_leaf)], 1.0)
                  for a_i in range(n_A) for c_leaf in competitor_nodes[nid]],
                 1.0, np.inf)
            # number of witnesses selected >= 1
    
        # (4) domination, generated LAZILY.
        #
        # The exact row set is one per (witness, A-B pair) triple:
        #     sum_j y[j,a'_j,b_j] - sum_j y[j,a_i_j,c_j] - M*z >= eps - M
        # which at z = 1 reads d_H(f(a_i), c) >= d_H(f(a'), b) + eps: the
        # selected witness out-distances that pair. Those rows number
        # (n_A * sum_X |X|) * (n_A * n_B) -- MULTIPLICATIVE, with n_A squared
        # since it sizes both factors. For a 1728-row dataset at
        # n_A = n_B = 15 that is ~5.8M rows which, as the dense matrix this
        # used to build, is ~8 GB for ONE pair: not slow but impossible, and
        # it capped this solver to tiny datasets.
        #
        # So rows are generated on demand, exactly as the Euclidean MIQP
        # does: seed one row per witness, solve, add only the rows the
        # incumbent actually violates, repeat. The generated model is a
        # RELAXATION of the full one (a subset of its rows), so its optimum
        # bounds the full optimum from below; terminating with nothing
        # violated means the incumbent is feasible for the full model and
        # therefore optimal for it. The seeded model has n_A * sum_X |X| rows
        # -- the same additive order a single-epigraph reformulation would
        # give, but without altering semantics that are already verified
        # against brute-force enumeration.
        #
        # (An epigraph on d(A,B) was tried and abandoned. It should be
        # equivalent and provably no looser, yet it accepted tie solutions
        # that brute force proves are not strictly dominating, on competitors
        # whose rows were somehow not binding. Not reintroduced until that is
        # understood.)
        comp_witnesses = {nid: [(a_i, c_leaf)
                                for a_i in range(n_A)
                                for c_leaf in competitor_nodes[nid]]
                          for nid in competitor_ids}
        comp_cvals = {nid: {c_leaf: np.asarray(self.X[c_leaf]).tolist()
                            for c_leaf in competitor_nodes[nid]}
                      for nid in competitor_ids}

        base_rr, base_cc, base_vv = list(rr), list(cc), list(vv)
        base_lb, base_ub, base_n = list(lb), list(ub), _nrow[0]
        added = set()
        gen = {'rr': [], 'cc': [], 'vv': [], 'lb': [], 'ub': []}

        def add_dom(nid, a_i, c_leaf, ap_i, b_i):
            key = (nid, a_i, c_leaf, ap_i, b_i)
            if key in added:
                return False
            c = comp_cvals[nid][c_leaf]
            ent = {}
            for j in range(d):
                key_b = (j, Acols[j][ap_i], Bcols[j][b_i])
                if key_b in y_col:
                    ent[y_col[key_b]] = ent.get(y_col[key_b], 0.0) + 1.0
                key_c = (j, Acols[j][a_i], c[j])
                if key_c in y_col:   # absent -> competitor-only value, never matches
                    ent[y_col[key_c]] = ent.get(y_col[key_c], 0.0) - 1.0
            zc = z_col[(nid, a_i, c_leaf)]
            ent[zc] = ent.get(zc, 0.0) - M
            r = base_n + len(gen['lb'])
            for cidx, val in ent.items():
                gen['rr'].append(r); gen['cc'].append(cidx); gen['vv'].append(val)
            gen['lb'].append(eps_int - M); gen['ub'].append(np.inf)
            added.add(key)
            return True

        def _solve_current():
            nrows = base_n + len(gen['lb'])
            A_sp = coo_matrix((base_vv + gen['vv'],
                               (base_rr + gen['rr'], base_cc + gen['cc'])),
                              shape=(nrows, n_vars)).tocsr()
            cons = LinearConstraint(A_sp, np.array(base_lb + gen['lb']),
                                    np.array(base_ub + gen['ub']))
            bnds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
            return milp(c_obj, constraints=cons, bounds=bnds,
                        integrality=np.ones(n_vars))

        def _relabel_of(x):
            rel = {j: {} for j in range(d)}
            for j in range(d):
                for a in src_vals[j]:
                    for b in tgt_vals[j]:
                        if x[y_col[(j, a, b)]] > 0.5:
                            rel[j][a] = b
                            break
                    rel[j].setdefault(a, a)
            out = A_pts.copy()
            for j in range(d):
                out[:, j] = [rel[j].get(int(v), int(v)) for v in A_pts[:, j].tolist()]
            return out

        # seed one row per witness, at the pair farthest apart under the
        # IDENTITY relabeling -- the likeliest binding pair before any solve
        d0 = np.array([[int(np.count_nonzero(A_pts[ap] != B_pts[bi]))
                        for bi in range(n_B)] for ap in range(n_A)])
        ap0, b0 = np.unravel_index(int(np.argmax(d0)), d0.shape)
        for nid in competitor_ids:
            for (a_i, c_leaf) in comp_witnesses[nid]:
                add_dom(nid, a_i, c_leaf, int(ap0), int(b0))

        max_rounds, rows_per_round = 64, 64
        converged, rounds, res = False, 0, None
        for rounds in range(1, max_rounds + 1):
            res = _solve_current()
            if not res.success or res.x is None:
                # a relaxation proven infeasible proves the full model is too
                return None, {"feasible": False, "rowgen_rounds": rounds,
                              "rowgen_rows": len(added), "rowgen_converged": True}
            x = res.x
            A_rel = _relabel_of(x)
            dAB = np.array([[int(np.count_nonzero(A_rel[ap] != B_pts[bi]))
                             for bi in range(n_B)] for ap in range(n_A)])
            any_added, stalled = False, 0
            for nid in competitor_ids:
                sel = [w for w in comp_witnesses[nid]
                       if x[z_col[(nid, w[0], w[1])]] > 0.5]
                if not sel:
                    continue
                satisfied, viol = False, {}
                for (a_i, c_leaf) in sel:
                    dw = int(np.count_nonzero(A_rel[a_i] != np.asarray(self.X[c_leaf])))
                    bad = np.argwhere(dAB + eps_int > dw)
                    if len(bad) == 0:
                        satisfied = True
                        break
                    viol[(a_i, c_leaf)] = bad
                if satisfied:
                    continue
                w_best = min(viol, key=lambda k: len(viol[k]))
                for (ap_i, b_i) in viol[w_best][:rows_per_round]:
                    if add_dom(nid, w_best[0], w_best[1], int(ap_i), int(b_i)):
                        any_added = True
                    else:
                        stalled += 1
            if not any_added:
                # Distinguish true convergence from "violated but every
                # offending row is already present", which would mean the
                # solver broke a row it was given rather than a row being
                # missing -- adding more cannot fix that, so do not call it
                # converged (the Euclidean solver learned this the hard way).
                converged = (stalled == 0)
                break

        # A stall (violated, but every offending row already present) means
        # the solver broke a row it was given -- a numerical failure, not a
        # proof of emptiness. Reporting it as infeasible was wrong: it made
        # this solver claim infeasible on a lymph pair the full dense model
        # solves with 2 coordinates changed. Fall through and return the
        # incumbent with rowgen_converged=False / optimal=False instead, so
        # the caller's independent validity check adjudicates rather than a
        # false "no counterfactual exists".
        stalled_out = not converged

        if not res.success:
            # genuinely possible for complete linkage (a spread-out B may be undominatable)
            return None, {"feasible": False}
    
        x = res.x
    
        # recover the per-coordinate relabeling
        relabel = {j: {} for j in range(d)}
        for j in range(d):
            for a in src_vals[j]:
                for b in tgt_vals[j]:
                    if x[y_col[(j, a, b)]] > 0.5:
                        relabel[j][a] = b
                        break
    
        def apply(A_in):
            A_in = np.asarray(A_in)
            out = A_in.copy()
            for j in range(d):
                out[:, j] = [relabel[j].get(v, v) for v in A_in[:, j].tolist()]
            return out
    
        A_prime = apply(A_pts)
        coords_changed = int(sum(1 for j in range(d)
                                if any(relabel[j][a] != a for a in src_vals[j])))
        binding = sorted({
            (index_to_name[c_leaf] if index_to_name is not None else c_leaf)
            for (nid, a_i, c_leaf), cidx in z_col.items() if x[cidx] > 0.5
        }, key=str)
    
        info = {
            "t": relabel,                      # {j: {alpha: beta}}
            "feasible": True,
            "coords_changed": coords_changed,  # == bits flipped for binary data
            "distance_moved": coords_changed,
            "binding_witnesses": binding,
            "phase": "milp-rowgen",
            "rowgen_rounds": rounds,
            "rowgen_rows": len(added),
            "rowgen_converged": not stalled_out,
            # optimal only on true convergence: no violated row left means the
            # relaxation optimum is full-model feasible, hence full-model
            # optimal. A stall carries no such guarantee.
            "optimal": not stalled_out,
        }
        return A_prime, info

    def get_leaves(self, node_id):
        """Recursively get all leaf indices under a given node."""
        if node_id < self.n:
            return [int(node_id)]
        left, right = int(self.Z[node_id - self.n, 0]), int(self.Z[node_id - self.n, 1])
        return self.get_leaves(left) + self.get_leaves(right)


    def all_internal_nodes(self):
        for i in range(len(self.Z)):
            nid = self.n + i
            yield nid, sorted(self.get_leaves(nid))

    def find_node_for_leaves(self, target_leaves):
        """Find the lowest internal node whose leaf set exactly matches target_leaves."""
        target_set = set(target_leaves)
        for i in range(len(self.Z)):
            node_id = self.n + i
            if set(self.get_leaves(node_id=node_id)) == target_set:
                return node_id
        return None


    def ancestors(self, node):
            path = set()
            for i in range(len(self.Z)):
                left, right = int(self.Z[i, 0]), int(self.Z[i, 1])
                if left == node or right == node:
                    parent = self.n + i
                    path.add(parent)
                    node = parent
            return path


    def find_lca(self, node_a, node_b):
        """Find the lowest common ancestor of two nodes."""
        

        ancestors_a = self.ancestors(node_a)

        # Walk up from node_b until we hit a shared ancestor
        node = node_b
        while node not in ancestors_a:
            for i in range(len(self.Z)):
                left, right = int(self.Z[i, 0]), int(self.Z[i, 1])
                if left == node or right == node:
                    node = self.n + i
                    break
        return node

    def get_merge_height(self, node_id):
        """Return the merge height (linkage distance) of an internal node."""
        return self.Z[node_id - self.n, 2]

    def formation_step(self, leaves_sorted):
        if len(leaves_sorted) == 1:
            return -1
        tgt = set(leaves_sorted)
        for i in range(len(self.Z)):
            if set(self.get_leaves(self.n + i)) == tgt:
                return i
        return None


    def competitor_leaf_sets(self, A_leaves, B_leaves):
        A_sorted = sorted(A_leaves); B_sorted = sorted(B_leaves)
        s_A = self.formation_step(A_sorted); s_B = self.formation_step(B_sorted)
        s_star = min(s_A, s_B)
        live = self._alive_clusters_before_row(s_star + 1)
        A_set, B_set = set(A_sorted), set(B_sorted)
        comps = [leaves for leaves in live.values()
                if not (set(leaves) <= A_set or set(leaves) <= B_set)]
        return comps, s_A, s_B, s_star


    def enumerate_pairs(self, min_leaf, max_leaf, max_pairs, seed=0):
        rng = np.random.default_rng(seed)

        # Collects all internal nodes
        branches = [(nid, lv) for nid, lv in self.all_internal_nodes()
                    if min_leaf <= len(lv) <= max_leaf]

        # A may be a branch OR a single leaf; B must be a branch (never a leaf)
        A_nodes = list(branches)
        B_nodes = list(branches)                      # B: branches only
        if min_leaf <= 1:
            A_nodes += [(j, [j]) for j in range(self.n)]   # leaves added to A only (no leaves in the internal nodes)

        pairs = []
        for _, A_lv in A_nodes:
            A_set = set(A_lv)
            for _, B_lv in B_nodes:
                if set(B_lv) & A_set or set(B_lv) == A_set:
                    continue
                pairs.append((A_lv, B_lv))
        rng.shuffle(pairs)
        return pairs[:max_pairs]

