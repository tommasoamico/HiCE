from modules.utilityFunctions import _dominates
import numpy as np
from scipy.spatial.distance import cdist


def validity(A_prime, B_pts, comp_pts, linkage, metric, eps=0.0, A_orig=None,
             return_margin=False):
    """
    Reuse the class's _dominates for euclidean; hamming handled explicitly.

    A_orig, if given, is A's own pre-move position and is folded into the
    competitor set like any other competitor: A_prime must still dominate it,
    i.e. be strictly closer to B than to where A already was. Without this,
    a "counterfactual" that barely moves and still effectively belongs to A
    would count as valid as long as it edges out the real competitors --
    matching HiCE's own solver, which always treats A's unmoved position as
    a competitor (see hiceCounterfactuals._build_ward_geometry).

    return_margin=True additionally returns the tightest competitor's slack
    (see _dominates' docstring / is_degenerate_tie below): positive means
    valid with that much room, non-positive means invalid, and a tiny
    negative value right at the tolerance boundary flags a near-tie rather
    than a clear violation.
    """
    if A_prime is None:
        return (False, np.nan) if return_margin else False
    A_prime = np.asarray(A_prime)
    competitors = list(comp_pts) + ([A_orig] if A_orig is not None else [])
    if metric == 'hamming':
        d = A_prime.shape[1]
        def hmax(P, Q):
            return cdist(P.astype(float), Q.astype(float), metric='hamming').max() * d
        dB = hmax(A_prime, B_pts)
        margins = [(hmax(A_prime, np.asarray(Xp)) - eps) - dB for Xp in competitors]
        margin = min(margins) if margins else np.inf
        ok = margin > 0
        return (ok, margin) if return_margin else ok
    # euclidean -> your predicate (complete uses max, ward uses weighted centroid)
    return _dominates(A_prime, np.asarray(B_pts),
                      [np.asarray(c) for c in competitors], linkage, eps,
                      return_margin=return_margin)


# A margin violation this small is indistinguishable from the kind of exact
# (or near-exact, e.g. duplicate-row) tie diagnosed in breast-w: the solver
# converges to the true (unattainable) infimum of an open strict-domination
# constraint and reports feasible/optimal, while revalidation's strict "<"
# correctly rejects it. Empirically that tie showed up at a ~1e-7 absolute
# violation (see hiceSparsityCoverage investigation); 1e-3 gives >3 orders
# of magnitude of headroom above that while still being far below the scale
# of a genuine, non-degenerate violation on normalized (unit-variance) data.
DEGENERATE_TIE_TOL = 1e-3


def is_degenerate_tie(margin):
    """
    True iff an invalid margin (<=0) is small enough to plausibly be a
    numerically-degenerate tie rather than a real domination violation --
    see DEGENERATE_TIE_TOL. NaN/inf margins (e.g. A_prime is None) are
    never ties.
    """
    return bool(np.isfinite(margin) and -DEGENERATE_TIE_TOL < margin <= 0)


def proximity(info, A_orig, A_prime, metric):
    if metric == 'hamming':
        if A_prime is None:
            return dict(l2=np.nan, l2_sq=np.nan, l1=np.nan, linf=np.nan, n_feat=np.nan)
        A0, A1 = np.asarray(A_orig), np.asarray(A_prime)
        changed = (A0 != A1)
        n_feat = int(np.count_nonzero(changed.any(axis=0)))
        total = int(changed.sum())
        return dict(l2=float(total), l2_sq=float(total), l1=float(total),
                    linf=float(n_feat), n_feat=n_feat)
    # euclidean: use info['t'] if present, else fall back to centroid shift
    t = info.get('t') if isinstance(info, dict) else None
    if t is None or np.ndim(t) == 0 or isinstance(t, dict):
        if A_prime is None:
            return dict(l2=np.nan, l2_sq=np.nan, l1=np.nan, linf=np.nan, n_feat=np.nan)
        t = np.asarray(A_prime).mean(0) - np.asarray(A_orig).mean(0)
    t = np.asarray(t, dtype=float).ravel()
    

    return dict(l2=float(np.linalg.norm(t)), l2_sq=float(t @ t),
                l1=float(np.abs(t).sum()),
                linf=float(np.abs(t).max()) if t.size else np.nan,
                n_feat=int(np.count_nonzero(np.abs(t) > 1e-9)))


def tree_coverage(Z, n, B_node, X, cf_points, metric='euclidean', n_A=None):
    """
    Top-down tree coverage of target branch B's sub-branches by a set of
    counterfactual points.

    A "sub-branch" is any descendant node of B with >=2 leaves (a bare leaf
    doesn't count). Starting at B, at each split the counterfactuals on hand
    are routed to whichever child they are Ward-closer to; a child that is
    itself a sub-branch is "represented" if at least one counterfactual
    routed to it. Recursion into a child reuses only the counterfactuals
    already routed there, so a single counterfactual can represent at most
    one sub-branch per level.

    Routing uses the SAME domination criterion the Ward-linkage solver
    (hiceCounterfactuals._add_competitor_apollonius) actually enforces
    between a moved branch and a candidate cluster: the Ward-weighted
    distance sqrt(w(n_S)) * ||p - mu_S||, with w(n_S) = 2*n_A*n_S/(n_A+n_S)
    and n_A the size of the moving branch. Plain (unweighted) Euclidean
    centroid distance is NOT the same comparison -- Ward's weight scales
    with each side's own leaf count, so a small-vs-large sibling split has
    a skewed decision boundary, not the raw perpendicular bisector.
    Without n_A (or for metric='hamming', which has no Ward weighting),
    falls back to plain distance.

    Returns
    -------
    (n_represented, n_subbranches) : both 0 if B itself is a leaf.
    """
    use_ward_weight = (metric != 'hamming') and (n_A is not None)

    def leaves_of(node):
        if node < n:
            return [int(node)]
        left, right = int(Z[node - n, 0]), int(Z[node - n, 1])
        return leaves_of(left) + leaves_of(right)

    def raw_dist(p, q):
        if metric == 'hamming':
            return float(cdist(p[None, :], q[None, :], metric='hamming')[0, 0] * p.shape[0])
        return float(np.linalg.norm(p - q))

    def side_dist(p, mu_S, n_S):
        d = raw_dist(p, mu_S)
        if use_ward_weight:
            w = 2.0 * n_A * n_S / (n_A + n_S)
            return np.sqrt(w) * d
        return d

    def walk(node, cfs):
        if node < n:
            return 0, 0
        left, right = int(Z[node - n, 0]), int(Z[node - n, 1])
        left_leaves = leaves_of(left)
        right_leaves = leaves_of(right)
        mu_left = np.asarray(X[left_leaves], dtype=float).mean(axis=0)
        mu_right = np.asarray(X[right_leaves], dtype=float).mean(axis=0)
        n_left, n_right = len(left_leaves), len(right_leaves)

        # A counterfactual generated by a sibling-preference constraint
        # d(A, target) <= d(A, sibling) routinely lands with that inequality
        # ACTIVE (the proximity-minimizing objective's optimum sits exactly
        # on the newly-binding constraint boundary), so dl == dr to solver
        # precision is a common, expected outcome here -- not a violation.
        # A strict '<' would drop such a point from BOTH sides and silently
        # lose it from coverage. Use a small relative/absolute tolerance and
        # break exact ties toward the left, so every point is always routed.
        left_cfs, right_cfs = [], []
        for c in cfs:
            dl = side_dist(c, mu_left, n_left)
            dr = side_dist(c, mu_right, n_right)
            tol = 1e-6 * max(1.0, dl, dr)
            if dl <= dr + tol:
                left_cfs.append(c)
            else:
                right_cfs.append(c)

        left_is_branch = left >= n
        right_is_branch = right >= n

        repr_left = int(left_is_branch and len(left_cfs) > 0)
        repr_right = int(right_is_branch and len(right_cfs) > 0)

        lr, lt = walk(left, left_cfs) if left_is_branch else (0, 0)
        rr, rt = walk(right, right_cfs) if right_is_branch else (0, 0)

        return (repr_left + repr_right + lr + rr,
                int(left_is_branch) + int(right_is_branch) + lt + rt)

    if B_node < n:
        return 0, 0
    cf_points = [np.asarray(c, dtype=float) for c in cf_points]
    return walk(B_node, cf_points)


def tree_coverage_from_targets(Z, n, B_node, target_nodes):
    """
    Exact (non-geometric) tree coverage for HiCE's diverseCounterfactuals
    results: a sub-branch node (>=2 leaves) counts as represented iff it is
    literally one of the given target_nodes -- the ground truth of which
    sub-branch each counterfactual was solved for (diverseCounterfactuals'
    own sibling-preference constraint, d(A, target) <= d(A, sibling)).

    tree_coverage's geometric routing (nearest-Ward-distance side, needed
    for methods like DiCE/BayCon that carry no such metadata) cannot be
    used for HiCE here: the proximity-minimizing objective routinely drives
    a specialized CF to land EXACTLY on the newly-active constraint
    boundary (d(target) == d(sibling) to solver precision), which is a
    genuine, unresolvable tie for a purely geometric router -- but not for
    this exact-match check, which already knows which side was intended.

    A "sub-branch" is any descendant node of B with >=2 leaves (matching
    tree_coverage's own definition), so B itself is never counted even if
    it appears in target_nodes (e.g. the depth-0 base counterfactual).

    Returns
    -------
    (n_represented, n_subbranches) : both 0 if B itself is a leaf.
    """
    target_set = set(int(t) for t in target_nodes)

    def walk(node):
        if node < n:
            return 0, 0
        left, right = int(Z[node - n, 0]), int(Z[node - n, 1])
        left_is_branch = left >= n
        right_is_branch = right >= n
        repr_left = int(left_is_branch and left in target_set)
        repr_right = int(right_is_branch and right in target_set)
        lr, lt = walk(left) if left_is_branch else (0, 0)
        rr, rt = walk(right) if right_is_branch else (0, 0)
        return (repr_left + repr_right + lr + rr,
                int(left_is_branch) + int(right_is_branch) + lt + rt)

    if B_node < n:
        return 0, 0
    return walk(B_node)


def sparsity(n_feat, n_total_features):
    """Fraction of features changed (0-1). NaN if n_feat is NaN/None or there
    are no features to change against."""
    if n_feat is None or n_total_features in (None, 0):
        return np.nan
    n_feat = float(n_feat)
    if np.isnan(n_feat):
        return np.nan
    return n_feat / n_total_features
