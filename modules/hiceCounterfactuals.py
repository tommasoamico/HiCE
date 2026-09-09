from typing import List, Dict, Iterable
import numpy as np
from modules.utilityFunctions import project_sphere_closed_form, project_single_sphere_sdp, active, feasible, unit, ward_info, complete_linkage_max
import miniball
from scipy.spatial.distance import cdist
from scipy.optimize import  milp, LinearConstraint, Bounds
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp
from scipy.optimize import minimize
from itertools import combinations


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
                               sparsity=True, lam=1.0, verbose=False):
        """
        Ward-linkage counterfactual: rigidly translate branch A by t so A becomes
        Ward-closer to target branch B than to every live competitor.

        Objective
        ---------
        sparsity=False (default): minimize ||t||^2  (least Euclidean displacement).
        sparsity=True            : minimize ||t||^2 + lam * ||t||_1  (elastic net),
                                which drives coordinates of t to exact zero,
                                yielding a FEATURE-SPARSE counterfactual. lam is
                                the sparsity weight; lam=0 recovers the L2 case.

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
        sparsity : bool     use the L2^2 + lam*L1 elastic-net objective.
        lam : float         L1 weight (only used when sparsity=True).
        exhaustive : bool   Phase-4 behaviour (see below).
        socp_solver : cvxpy solver constant, optional (default CLARABEL, else ECOS).

        Returns
        -------
        A_prime : np.ndarray (n_A x d)
        info : dict  t, feasible, optimal, distance_moved, n_competitors,
                    active_balls, phase, plus sparsity/lam echoed back.
        """
        n = len(self.X)
        d = self.X.shape[1]

        # ---- Step 0: resolve A, B as static wholes ----
        A_node, A_leaves_sorted, s_A = self._resolve_branch(A_leaves)
        B_node, B_leaves_sorted, s_B = self._resolve_branch(B_leaves)
        if A_node == B_node:
            raise ValueError("A_leaves and B_leaves resolve to the same node.")

        A_pts = np.asarray(self.X[A_leaves_sorted], dtype=float)
        B_pts = np.asarray(self.X[B_leaves_sorted], dtype=float)
        n_A, n_B = len(A_pts), len(B_pts)
        mu_A = A_pts.mean(axis=0)
        mu_B = B_pts.mean(axis=0)

        def w(nS):
            return 2.0 * n_A * nS / (n_A + nS)
        w_B = w(n_B)

        # ---- Step 1: anchor + live partition ----
        if anchor != 'min':
            raise NotImplementedError("Only anchor='min' (s* = min(s_A, s_B)) is implemented.")
        s_star = min(s_A, s_B)
        live_partition = self._alive_clusters_before_row(s_star + 1)

        # ---- Step 2: competitor set ----
        A_set, B_set = set(A_leaves_sorted), set(B_leaves_sorted)
        competitor_nodes = {nid: leaves for nid, leaves in live_partition.items()
                            if not (set(leaves) <= A_set or set(leaves) <= B_set)}

        if verbose:
            print(f"n_A={n_A}(s_A={s_A}) n_B={n_B}(s_B={s_B}) s*={s_star} "
                f"competitors={len(competitor_nodes)} sparsity={sparsity} lam={lam}")

        if not competitor_nodes:
            info = ward_info(np.zeros(d), True, True, 0, [], "guard-no-competitors")
            info["sparsity"] = sparsity; info["lam"] = lam if sparsity else 0.0
            return A_pts.copy(), info

        # ---- Apollonius data (objective-independent) ----
        convex_ball_in = []   # (kappa, R)         ||p-kappa|| <= R
        convex_half = []      # (a, b)             a·p <= b
        rev_balls = []        # {kappa, R, nid}    ||p-kappa|| >= R  (reverse-convex)

        for nid, leaves in competitor_nodes.items():
            mu_X = np.asarray(self.X[leaves], dtype=float).mean(axis=0)
            n_X = len(leaves)
            coeff = w_B - w(n_X)
            bvec = w_B * mu_B - w(n_X) * mu_X
            cval = w_B * (mu_B @ mu_B) - w(n_X) * (mu_X @ mu_X)
            if abs(coeff) < 1e-12:
                convex_half.append((-2.0 * bvec, -cval))
            else:
                kappa = bvec / coeff
                R = np.sqrt(max(kappa @ kappa - cval / coeff, 0.0))
                (convex_ball_in if coeff > 0 else rev_balls).append(
                    (kappa, R) if coeff > 0 else {'kappa': kappa, 'R': R, 'nid': nid})

        #rho = float(np.linalg.norm(mu_B - mu_A))

        if socp_solver is None:
            socp_solver = cp.CLARABEL if hasattr(cp, "CLARABEL") else cp.ECOS

        # ---- objective helper (numeric, for scoring / SLSQP) ----
        def obj_val(p):
            t = p - mu_A
            v = float(t @ t)
            if sparsity:
                v += lam * float(np.sum(np.abs(t)))
            return v

        # ---- convex projection onto K (SOCP; + L1 term if sparsity) ----
        def project_K_socp():
            p = cp.Variable(d)
            cons = []
            for (kappa, R) in convex_ball_in:
                cons.append(cp.norm(p - kappa) <= R)
            for (a, b) in convex_half:
                cons.append(a @ p <= b)
            objective = cp.sum_squares(p - mu_A)
            if sparsity:
                objective = objective + lam * cp.norm1(p - mu_A)   # elastic net
            prob = cp.Problem(cp.Minimize(objective), cons)
            prob.solve(solver=socp_solver)
            if p.value is None:
                return mu_A.copy()
            return np.asarray(p.value, dtype=float)

        # ---- sphere ∩ K projection ----
        def project_spheres_capK(active_spheres, n_restarts=8):
            """
            L2 objective, single sphere:
                (1) closed-form radial foot if it lies in K (exact);
                (2) else S-lemma SDP (exact); (3) else SLSQP backstop.
            L1 objective (sparsity=True): the radial foot and the S-lemma SDP are
            NOT valid (radial is L2-only; the Shor SDP is not tight with an L1
            term), so we go straight to multistart SLSQP on L2^2+lam*L1.
            Multiple spheres: multistart SLSQP either way.
            """
            # ---- single sphere, pure L2: exact fast paths ----
            if len(active_spheres) == 1 and not sparsity:
                kappa, R = active_spheres[0]
                p_cf = project_sphere_closed_form(mu_A, kappa, R)
                if feasible(p_cf, convex_ball_in, convex_half, rev_balls):
                    return p_cf
                p_sdp = project_single_sphere_sdp(mu_A, kappa, R, convex_ball_in, convex_half)
                if p_sdp is not None and feasible(p_sdp, convex_ball_in, convex_half, rev_balls):
                    return p_sdp
                # fall through to SLSQP

            # ---- SLSQP on the (possibly L1) objective ----
            eq_cons = [{'type': 'eq',
                        'fun': (lambda p, k=k, R=R: np.linalg.norm(p - k) - R)}
                    for (k, R) in active_spheres]
            ineq_cons = []
            for (kappa, R) in convex_ball_in:
                ineq_cons.append({'type': 'ineq',
                                'fun': (lambda p, k=kappa, R=R: R - np.linalg.norm(p - k))})
            for (a, b) in convex_half:
                ineq_cons.append({'type': 'ineq',
                                'fun': (lambda p, a=a, b=b: b - a @ p)})
            cons = eq_cons + ineq_cons

            rng = np.random.default_rng(0)
            starts = []
            k0, R0 = active_spheres[0]
            dirc = mu_A - k0
            nrm = np.linalg.norm(dirc)
            starts.append(k0 + R0 * (dirc / nrm if nrm > 1e-12 else unit(d, rng)))
            for _ in range(n_restarts - 1):
                starts.append(k0 + R0 * unit(d, rng))

            # objective + gradient; L1 uses a smooth-ish subgradient (sign), which
            # SLSQP tolerates in practice with multistart.
            def fobj(p):
                return obj_val(p)
            def fjac(p):
                t = p - mu_A
                g = 2.0 * t
                if sparsity:
                    g = g + lam * np.sign(t)
                return g

            best, best_val = None, np.inf
            for s in starts:
                res = minimize(fobj, s, method='SLSQP', jac=fjac,
                            constraints=cons, options={'maxiter': 400, 'ftol': 1e-12})
                if res.success and feasible(res.x, convex_ball_in, convex_half, rev_balls):
                    val = obj_val(res.x)
                    if val < best_val:
                        best_val, best = val, res.x
            return best

        # convenience for the info dict
        def _finish(p, phase, optimal):
            t = p - mu_A
            info = ward_info(t, True, optimal, len(competitor_nodes),
                            active(p, rev_balls, index_to_name), phase)
            info["sparsity"] = sparsity
            info["lam"] = lam if sparsity else 0.0
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
            # exact for L2 (radial/SDP); heuristic for L1 (SLSQP) -> optimal flag reflects it
            return _finish(p, "phase3-greedy", not sparsity)

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
        opt = exhaustive and not sparsity
        phase = ("phase4-exhaustive" if exhaustive else "phase4-earlystop")
        return _finish(best_p, phase, opt)

    def counterfactual_complete_linkage_branch(self, B_leaves, A_leaves,
                                            metric='euclidean', anchor='min',
                                            eps=1e-6, index_to_name=None,
                                            big_m=None,
                                            t_bound=1e4, time_limit=60, verbose=False):
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

        #print(f"Branch A                  : {_label(A_leaves_sorted)} (n_A={n_A}, step s_A={s_A})")
        #print(f"Branch B (target)         : {_label(B_leaves_sorted)} (n_B={n_B}, step s_B={s_B})")
        #print(f"Anchor step s*            : {s_star}")
        #print(f"Live competitors at s*    : {len(competitor_nodes)} clusters, "
        #     f"sizes {[len(v) for v in competitor_nodes.values()]}")

        # Guard 1: no competitors at all -> every t is feasible, nothing to beat.
        if not competitor_nodes:
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
                verbose=verbose)
        elif metric == 'hamming':
            A_prime, info = self._solve_branch_hamming(
                A_pts, B_pts, competitor_nodes, eps, big_m, index_to_name)
        else:
            raise ValueError(f"Unsupported metric: {metric!r}; use 'euclidean' or 'hamming'.")

        # ---------------------------------------------------------------
        # Step 5 — output and checks.
        # ---------------------------------------------------------------
        if not info.get("feasible", False):
            print("\nFeasible                  : False")
            print("No t satisfies the complete-linkage margin against every live "
                "competitor — the counterfactual region can be genuinely empty "
                "for complete linkage (a spatially large/spread B is not "
                "necessarily dominatable).")
            info["n_competitors"] = len(competitor_nodes)
            return None, info

        moved_norm = info.get("distance_moved", info.get("bits_flipped"))
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


    '''def counterfactual_complete_linkage_branch_gurobi(
        self, B_leaves, A_leaves,
        metric='euclidean', anchor='min', eps=1e-6,
        index_to_name=None, t_bound=1e4, time_limit=60, verbose=False):
        """
        Branch-aware complete-linkage counterfactual, solved as a *convex* MIQP
        with Gurobi.

        Instead of enumerating one witness per competitor and 
        solving a QP per assignment (exponential in the number of competitors),
        we hand the whole disjunctive problem to a branch-and-bound solver as a
        single convex MIQP. Two deliberate design choices:

        1. SQUARED-AND-LINEARISED domination.  The complete-linkage domination
            inequality  ||q + t|| >= ||p + t||  is squared, which cancels the
            ||t||^2 term and leaves a *linear* row in t:

                2 <p - q, t>  <=  ||q||^2 - ||p||^2 - eps.

            So the objective stays convex quadratic (min ||t||^2) and every
            constraint is linear -> a CONVEX MIQP.  This is exactly the regime
            MIQP branch-and-bound solver is efficient on, and (crucially) a
            convex MIQP is solved to certified GLOBAL optimality.

        Both A and B may be single leaves or full internal subtrees.  A is
        translated rigidly (one shared t applied to every point of A), which is
        why the domination must hold against the FULL difference set V_AB (every
        a in A paired with every b in B), not just a per-point slice.

        The disjunction "some witness q in V_AX dominates every p in V_AB" is
        encoded with one binary z per (competitor, witness pair) and
        sum(z) >= 1 per competitor.  Each domination row is deactivated when its
        binary is 0 via a big-M term (M z on the <= side, so z=0 makes the row
        vacuous and z=1 makes it bind).  A valid finite M is derived per row from
        the t-box, so no manual tuning is needed.

        Parameters
        ----------
        B_leaves : list of int
            Leaf indices of the target branch B (passed first, matching the other
            branch functions).
        A_leaves : list of int
            Leaf indices of the branch A to move.  A single leaf is the n_A = 1
            special case.
        X : np.ndarray
            Data matrix (rows = leaves), continuous features for metric='euclidean'.
        Z : np.ndarray
            SciPy linkage matrix (build it with a Euclidean metric so competitor
            structure is consistent with the L2 geometry solved here).
        metric : {'euclidean'}
            Only Euclidean is supported here; the squared-domination trick is L2
            specific.  (Hamming has its own MILP function.)
        anchor : {'min'}
            Competitor anchor step s* = min(s_A, s_B); only 'min' is implemented.
        eps : float
            Strict-domination margin (keeps "strictly closer than every
            competitor" well-defined and the feasible region closed).
        index_to_name : dict, optional
            index -> label, for readable diagnostics only.
        t_bound : float
            Box bound |t_j| <= t_bound.  Keeps the model finite and supplies the
            big-M.  Hitting the box is a signal the region may be unbounded.
        time_limit : float
            Gurobi time limit (seconds).  If exceeded, the returned point is the
            incumbent and info['optimal'] is False (a bound, not the certified
            optimum).
        verbose : bool
            If True, let Gurobi print its log.

        Returns
        -------
        A_prime : np.ndarray or None
            Moved branch A + t* (shape n_A x d), or None if the model is proven
            infeasible (the complete-linkage counterfactual region can be
            genuinely empty).
        info : dict
            t, feasibility, optimality flag, distance moved, competitor count,
            active (binding) witnesses, and the raw Gurobi status.
        """

        if metric != 'euclidean':
            raise ValueError("This solver is Euclidean-only; use the Hamming MILP for binary data.")
        if anchor != 'min':
            raise NotImplementedError("Only anchor='min' (s* = min(s_A, s_B)) is implemented.")

        

        # ------------------------------------------------------------------
        # Step 0 — resolve A and B as static wholes: node id, sorted leaves,
        # and formation step s (row index in Z; -1 for a bare leaf).
        # ------------------------------------------------------------------
        A_node, A_leaves_sorted, s_A = self._resolve_branch(A_leaves)
        B_node, B_leaves_sorted, s_B = self._resolve_branch(B_leaves)
        if A_node == B_node:
            raise ValueError("A_leaves and B_leaves resolve to the same node.")

        A_pts = np.asarray(self.X[A_leaves_sorted], dtype=float)
        B_pts = np.asarray(self.X[B_leaves_sorted], dtype=float)
        #n_A, n_B = len(A_pts), len(B_pts)

        def _label(leaves):
            return [index_to_name[i] for i in leaves] if index_to_name is not None else leaves

        # ------------------------------------------------------------------
        # Step 1 — anchor step and live partition L(s*).  s* = min(s_A, s_B) is
        # the earliest moment structure in this region is committed; replaying
        # rows 0..s* (cutoff = s*+1) yields the live blocks there.
        # ------------------------------------------------------------------
        s_star = min(s_A, s_B)
        live_partition = self._alive_clusters_before_row( s_star + 1)

        # ------------------------------------------------------------------
        # Step 2 — competitor set: live blocks disjoint from BOTH A and B.
        # L(s*) is a partition, so every block is wholly inside A, wholly inside
        # B, or disjoint — the subset tests are unambiguous.  Blocks inside A are
        # the mover; blocks inside B are dropped because B enters whole (its
        # fragments must never compete against their own parent-as-target).
        # ------------------------------------------------------------------
        A_set, B_set = set(A_leaves_sorted), set(B_leaves_sorted)
        competitor_nodes = {}
        for node_id, leaves in live_partition.items():
            leaf_set = set(leaves)
            if leaf_set <= A_set or leaf_set <= B_set:
                continue
            competitor_nodes[node_id] = leaves

        #print(f"Branch A                  : {_label(A_leaves_sorted)} (n_A={n_A}, step s_A={s_A})")
        #print(f"Branch B (target)         : {_label(B_leaves_sorted)} (n_B={n_B}, step s_B={s_B})")
        #print(f"Anchor step s*            : {s_star}")
        #rint(f"Live competitors at s*    : {len(competitor_nodes)} clusters, "
        #     f"sizes {[len(v) for v in competitor_nodes.values()]}")

        def cl_max(P_pts, Q_pts):
            return cdist(P_pts, Q_pts, metric='euclidean').max()

        # Guard 1 — no competitors: every t feasible, nothing to beat.
        if not competitor_nodes:
            print("No live competitors at s* — every t is feasible; returning t*=0.")
            return A_pts.copy(), {"t": np.zeros(self.d), "feasible": True, "optimal": True,
                                "distance_moved": 0.0, "n_competitors": 0,
                                "already_optimal": True}

        # Guard 2 — A already strictly dominates B against every competitor at t=0.
        dAB0 = cl_max(A_pts, B_pts)
        if all(dAB0 < cl_max(A_pts, np.asarray(self.X[leaves], dtype=float)) - eps
            for leaves in competitor_nodes.values()):
            print("A already dominates B against every competitor at t=0; returning t*=0.")
            return A_pts.copy(), {"t": np.zeros(self.d), "feasible": True, "optimal": True,
                                "distance_moved": 0.0, "n_competitors": len(competitor_nodes),
                                "already_optimal": True}

        # ------------------------------------------------------------------
        # Step 3 — difference sets.
        #   P   = V_AB = {a - b}  : the A-B pairs; the chosen witness must beat
        #                            ALL of these (rigid A => full product, not a
        #                            per-point slice).
        #   Q_X = V_AX = {a - c}  : per competitor, the candidate witnesses; each
        #                            carries its originating competitor leaf so the
        #                            binding witness can be reported by name.
        # No hull reduction (kept exact); scales as n_A*n_B and n_A*|X|.
        # ------------------------------------------------------------------
        P = [a - b for a in A_pts for b in B_pts]
        competitor_Q = {}
        for nid, leaves in competitor_nodes.items():
            C_pts = np.asarray(self.X[leaves], dtype=float)
            competitor_Q[nid] = [(a - c, leaf) for a in A_pts for leaf, c in zip(leaves, C_pts)]

        # ------------------------------------------------------------------
        # Step 4 — the convex MIQP.
        #   min ||t||^2
        #   for each competitor X:  sum_q z_{X,q} >= 1 # for each competitor, at least one witness must dominate
        # i.e. there need to be at least one point of X farther from A than any pair in A-B
        #   for each competitor X, witness q, pair p:
        #       2<p - q, t>  <=  ||q||^2 - ||p||^2 - eps + M_{X,q,p} (1 - z_{X,q})
        # Convex quadratic objective, linear constraints, binary z -> convex MIQP,
        # solved to certified global optimality by Gurobi's B&B.
        # ------------------------------------------------------------------
        model = gp.Model("cl_counterfactual_branch")
        model.Params.OutputFlag = 1 if verbose else 0
        model.Params.TimeLimit = time_limit
        

        t = model.addVars(self.d, lb=-t_bound, ub=t_bound, name="t")
        model.setObjective(gp.quicksum(t[j] * t[j] for j in range(self.d)), GRB.MINIMIZE)

        z = {}  # (competitor_id, witness_row_index) -> binary var
        binding_lookup = {}  # (competitor_id, witness_row_index) -> competitor leaf (for reporting)

        # In low dimesnions finding the hull helps a lot
        # Many other possible speed-ups exist (pruning of witnesses and prioritizing likelihood of being the witness)
        # Eventually test wether to implement them
        for nid, q_list in competitor_Q.items():
            # q_list = The list of candidate witnesses, i.e. a-c, a\inA and C\in X
            # one binary per candidate witness of this competitor
            for wi, (q, leaf) in enumerate(q_list):
                # leaf: leaf index of the competitor point
                z[(nid, wi)] = model.addVar(vtype=GRB.BINARY, name=f"z_{nid}_{wi}")
                binding_lookup[(nid, wi)] = leaf

            # at least one witness selected for this competitor (the existential)
            model.addConstr(gp.quicksum(z[(nid, wi)] for wi in range(len(q_list))) >= 1,
                            name=f"pick_{nid}")

            # domination rows: chosen witness q must beat every p in P (= V_AB)
            for wi, (q, _leaf) in enumerate(q_list):
                qq = float(np.dot(q, q))
                for p in P:
                    # linear-in-t row: 2<p-q, t> <= ||q||^2 - ||p||^2 - eps  (when z=1)
                    coeff = 2.0 * (p - q)                 # length-d vector
                    rhs = qq - float(np.dot(p, p)) - eps
                    # big-M so the row is vacuous when z=0.  On the box |t_j|<=t_bound,
                    # the LHS 2<p-q,t> is at most 2*||p-q||_1 * t_bound in abs value;
                    # M must cover (max LHS - rhs) so that z=0 can't bind the row.
                    lhs_max = 2.0 * float(np.sum(np.abs(p - q))) * t_bound
                    M = lhs_max - rhs + 1.0
                    model.addConstr(
                        gp.quicksum(coeff[j] * t[j] for j in range(self.d))
                        <= rhs + M * (1 - z[(nid, wi)]),
                        name=f"dom_{nid}_{wi}_{id(p) % 100000}")

        model.optimize()

        # ------------------------------------------------------------------
        # Step 5 — read out, distinguishing "certified global optimum" from
        # "incumbent at time limit".
        # ------------------------------------------------------------------
        status = model.Status
        if status == GRB.INFEASIBLE:
            print("\nFeasible                  : False")
            print("No t satisfies the complete-linkage margin against every live "
                "competitor — the region can be genuinely empty (a spread-out B "
                "need not be dominatable).")
            return None, {"feasible": False, "optimal": True,  # infeasibility is a certified result
                        "n_competitors": len(competitor_nodes), "gurobi_status": status}

        if model.SolCount == 0:
            # no incumbent found (e.g. time-limited before any feasible point)
            print("\nFeasible                  : unknown (no incumbent found within time limit)")
            return None, {"feasible": False, "optimal": False,
                        "n_competitors": len(competitor_nodes), "gurobi_status": status}

        t_val = np.array([t[j].X for j in range(self.d)])
        A_prime = A_pts + t_val
        dist_moved = float(np.linalg.norm(t_val))
        is_optimal = (status == GRB.OPTIMAL)

        # active witnesses: competitor leaves whose selection binary is 1
        binding = sorted({
            (index_to_name[leaf] if index_to_name is not None else leaf)
            for key, var in z.items() if var.X > 0.5
            for leaf in [binding_lookup[key]]
        }, key=str)

        if any(abs(abs(v) - t_bound) < 1e-6 for v in t_val):
            print("NOTE: |t_j| hit t_bound — the feasible region may be unbounded; "
                "consider raising t_bound.")

        print(f"\nFeasible                  : True")
        print(f"Globally optimal          : {is_optimal}  (Gurobi status {status})")
        print(f"||t*||                    : {dist_moved:.6f}")
        print(f"Active (binding) witnesses: {binding}")

        info = {
            "t": t_val,
            "feasible": True,
            "optimal": is_optimal,               # True only if status == OPTIMAL
            "distance_moved": dist_moved,
            "n_competitors": len(competitor_nodes),
            "binding_witnesses": binding,
            "gurobi_status": status,
        }
        return A_prime, info
'''


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
    
        # big-M: domination LHS in [-d, d]; deactivation needs eps - M <= -d.
        M = big_m if big_m is not None else d + abs(eps) + 1.0
    
        c_obj = np.zeros(n_vars)
        #c_obj is the vector of coefficients that we will give in input to scipy's milp
        for j in range(d):
            c_obj[moved_col[j]] = 1.0  # minimize number of coordinates relabeled, i.e. small move and sparisty coincide
            # every variable has a weight of 1
    
        A_rows, lb, ub = [], [], []
    
        # (1) totality: each (j, alpha) maps to exactly one beta
        for j in range(d):
            for a in src_vals[j]:
                row = np.zeros(n_vars)
                for b in tgt_vals[j]:
                    row[y_col[(j, a, b)]] = 1.0 # we put at 1 every coefficient
                A_rows.append(row); lb.append(1.0); ub.append(1.0)
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
                row = np.zeros(n_vars)
                row[moved_col[j]] = 1.0 #coefficient of the constraint
                row[y_col[(j, a, a)]] = 1.0 # coefficient of the constraint
                A_rows.append(row) # Add to the constrain matrix
                lb.append(1.0)
                ub.append(np.inf)
    
        # (3) at least one witness selected per competitor
        for nid in competitor_ids:
            row = np.zeros(n_vars)
            for a_i in range(n_A):
                for c_leaf in competitor_nodes[nid]:
                    row[z_col[(nid, a_i, c_leaf)]] = 1.0
            A_rows.append(row); lb.append(1.0); ub.append(np.inf)
            # number of witnesses selected >= 1
    
        # (4) domination: witness (a_i, c) beats every A-B pair (a', b):
        #     sum_j y[j,a'_j,b_j] - sum_j y[j,a_i_j,c_j] - M*z >= eps - M
        # if c is the withness z=1, the difference between the two sums is > eps i.e. 
        # d - d(a',b) > d - d(a, c), d(a, c) > d(a', b). Remember the definition d(a, b) = d - sum_j y[j, a_j, b_j]


        # To speed it up we need to implement lazy constraint generation, consider also that
        # in low dimensions we improve a lot with hull reduction
        for nid in competitor_ids:
            for a_i in range(n_A):
                for c_leaf in competitor_nodes[nid]:
                    c = np.asarray(self.X[c_leaf]).tolist()
                    zc = z_col[(nid, a_i, c_leaf)]
                    for ap_i in range(n_A):
                        for b_i in range(n_B):
                            row = np.zeros(n_vars)
                            for j in range(d):
                                key_b = (j, Acols[j][ap_i], Bcols[j][b_i])
                                if key_b in y_col:
                                    row[y_col[key_b]] += 1.0
                                key_c = (j, Acols[j][a_i], c[j])
                                if key_c in y_col:          # absent -> competitor-only value, no match
                                    row[y_col[key_c]] -= 1.0
                            row[zc] += -M
                            A_rows.append(row); lb.append(eps - M); ub.append(np.inf)
    
        constraints = LinearConstraint(np.array(A_rows), np.array(lb), np.array(ub))
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars)  # all binary (y, moved_j, z)
    
        res = milp(c_obj, constraints=constraints, bounds=bounds, integrality=integrality)
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

