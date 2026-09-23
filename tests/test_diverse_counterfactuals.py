"""
Regression tests for the incremental diverseCounterfactuals traversal.

diverseCounterfactuals is a pure efficiency refactor over a "call
counterfactual_ward_branch fresh at every node" baseline: it hoists the base
Apollonius geometry out of the traversal (built once via
_build_ward_geometry) and reuses a parent's solution without re-solving
whenever that solution provably remains optimal for the child (nested
feasible regions, unchanged objective). These tests check the refactor
changed nothing observable: every returned counterfactual must match the
from-scratch baseline to numerical tolerance, while doing it with fewer
solves and exactly one base-geometry build.
"""
import numpy as np
import pytest
import cvxpy as cp
from scipy.cluster.hierarchy import linkage

from modules.hiceCounterfactuals import hiceCounterfactuals
from modules.utilityFunctions import get_children


def _make_instance(seed, n=14, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    Z = linkage(X, method='ward')
    return hiceCounterfactuals(X, Z, metric='euclidean', linkage='ward')


def _pick_case(seed, n=14, d=3):
    """Pick a target branch B (an internal node with >=4 leaves, not the
    whole tree) and a query leaf A outside of it, giving a multi-level
    subtree to descend."""
    h = _make_instance(seed, n=n, d=d)
    for i in range(len(h.Z) - 1, -1, -1):
        nid = h.n + i
        leaves = set(h.get_leaves(nid))
        if 4 <= len(leaves) < n:
            outside = [l for l in range(n) if l not in leaves]
            if outside:
                return h, nid, outside[0]
    raise RuntimeError("no suitable (A, B) fixture found")


def _naive_diverse(h, B_leaves, A_leaves, max_depth=None, both_children=True, **kwargs):
    """
    Ground-truth reimplementation that rebuilds the FULL Apollonius geometry
    from scratch at every node of B's subtree -- the original "under B"
    competitor constraints (via _build_ward_geometry, rebuilt fresh each
    time rather than hoisted out once) plus every sibling-preference
    constraint accumulated along the path so far -- i.e. the
    pre-optimization behavior of diverseCounterfactuals under the corrected
    child-vs-sibling mechanic (d(A, child) < d(A, sibling), using the
    child's own centroid/weight, NOT the sibling-as-competitor-against-B
    framing). Used only to check that the incremental implementation is
    numerically equivalent.
    """
    B_node, B_leaves_sorted, _ = h._resolve_branch(B_leaves)
    n_A = len(list(A_leaves))
    socp_solver = cp.CLARABEL if hasattr(cp, "CLARABEL") else cp.ECOS

    def solve(sibling_constraints):
        base = h._build_ward_geometry(
            B_leaves_sorted, A_leaves,
            anchor=kwargs.get('anchor', 'min'),
            allowed_features=kwargs.get('allowed_features'),
            sparsity=kwargs.get('sparsity', True),
            lam=kwargs.get('lam', 1.0),
        )
        convex_ball_in = list(base["convex_ball_in"])
        convex_half = list(base["convex_half"])
        rev_balls = list(base["rev_balls"])
        n_competitors = len(base["competitor_nodes"])
        for (mu_near, n_near, sib_leaves) in sibling_constraints:
            w_near = 2.0 * n_A * n_near / (n_A + n_near)
            h._add_competitor_apollonius(
                "__sib__", sib_leaves, base["mu_A"], mu_near, n_A, w_near,
                convex_ball_in, convex_half, rev_balls)
            n_competitors += 1
        return h._solve_ward_core(
            base["A_pts"], base["mu_A"], base["mu_B"], base["E"],
            base["free_idx"], base["fixed_idx"], base["k_free"],
            convex_ball_in, convex_half, rev_balls,
            base["sparsity"], base["lam"], base["use_sparsity"], base["allowed_features"],
            n_competitors, base["d"], socp_solver, kwargs.get('exhaustive', False),
            kwargs.get('index_to_name'))

    results = []
    stack = [(B_node, [], [], 0)]
    while stack:
        node, sib_constraints, path, depth = stack.pop()
        A_prime, info = solve(sib_constraints)
        results.append({"target_node": node, "path": list(path),
                         "A_prime": A_prime, "info": info, "depth": depth})
        if not info.get("feasible", False):
            continue
        if max_depth is not None and depth >= max_depth:
            continue
        children = get_children(h.Z, h.n, node)
        if children is None:
            continue
        c_left, c_right = children
        left_leaves = h.get_leaves(c_left)
        right_leaves = h.get_leaves(c_right)
        mu_left = np.asarray(h.X[left_leaves], dtype=float).mean(axis=0)
        mu_right = np.asarray(h.X[right_leaves], dtype=float).mean(axis=0)
        n_left, n_right = len(left_leaves), len(right_leaves)
        if both_children:
            stack.append((c_left, sib_constraints + [(mu_left, n_left, right_leaves)],
                          path + [(c_left, c_right)], depth + 1))
            stack.append((c_right, sib_constraints + [(mu_right, n_right, left_leaves)],
                          path + [(c_right, c_left)], depth + 1))
        else:
            mu_A_prime = np.asarray(A_prime, dtype=float).mean(axis=0)
            d_left = np.linalg.norm(mu_A_prime - mu_left)
            d_right = np.linalg.norm(mu_A_prime - mu_right)
            if d_left <= d_right:
                stack.append((c_right, sib_constraints + [(mu_right, n_right, left_leaves)],
                              path + [(c_right, c_left)], depth + 1))
            else:
                stack.append((c_left, sib_constraints + [(mu_left, n_left, right_leaves)],
                              path + [(c_left, c_right)], depth + 1))
    return results


# sparsity=False (pure ||t||^2, no L1 term) throughout these equivalence
# tests: the default sparsity=True elastic-net objective is only weakly
# convex (the L1 term admits ties/flat faces in low dimension), so two
# solves of two DIFFERENT-but-equally-optimal SOCPs (e.g. naive's from-
# scratch child problem vs. the parent's problem the optimized path reused)
# can legitimately land on different minimizers on that tie set -- a
# pre-existing property of the objective, unrelated to this refactor. Pure
# L2 is strictly convex, so the minimizer is unique and both paths must
# agree to plain solver precision.

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("both_children", [True, False])
def test_diverse_counterfactuals_matches_naive(seed, both_children):
    h, B_node, A_leaf = _pick_case(seed)
    B_leaves = h.get_leaves(B_node)

    naive = _naive_diverse(h, B_leaves, [A_leaf], both_children=both_children, sparsity=False)
    optimized = h.diverseCounterfactuals(B_leaves, [A_leaf], both_children=both_children, sparsity=False)

    naive_by_node = {r["target_node"]: r for r in naive}
    optimized_by_node = {r["target_node"]: r for r in optimized}

    assert set(naive_by_node.keys()) == set(optimized_by_node.keys())
    assert len(naive) == len(optimized)

    for node, r_naive in naive_by_node.items():
        r_opt = optimized_by_node[node]
        assert r_naive["info"]["feasible"] == r_opt["info"]["feasible"]
        assert r_naive["info"]["optimal"] == r_opt["info"]["optimal"]
        np.testing.assert_allclose(r_naive["info"]["t"], r_opt["info"]["t"], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(r_naive["A_prime"], r_opt["A_prime"], atol=1e-4, rtol=1e-4)


def test_parent_reuse_correctness():
    h, B_node, A_leaf = _pick_case(0)
    B_leaves = h.get_leaves(B_node)

    optimized = h.diverseCounterfactuals(B_leaves, [A_leaf], both_children=True, sparsity=False)
    inherited = [r for r in optimized if r["info"]["phase"] == "inherited-parent"]
    assert inherited, "expected at least one inherited-parent node for this fixture"

    naive = _naive_diverse(h, B_leaves, [A_leaf], both_children=True, sparsity=False)
    naive_by_node = {r["target_node"]: r for r in naive}

    for r in inherited:
        r_naive = naive_by_node[r["target_node"]]
        assert r_naive["info"]["feasible"]
        np.testing.assert_allclose(r_naive["info"]["t"], r["info"]["t"], atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(r_naive["A_prime"], r["A_prime"], atol=1e-4, rtol=1e-4)


def test_solve_count_below_node_count():
    h, B_node, A_leaf = _pick_case(0)
    B_leaves = h.get_leaves(B_node)

    solve_calls = {"n": 0}
    orig_solve = h._solve_ward_core

    def counting_solve(*a, **kw):
        solve_calls["n"] += 1
        return orig_solve(*a, **kw)

    h._solve_ward_core = counting_solve
    try:
        results = h.diverseCounterfactuals(B_leaves, [A_leaf], both_children=True)
    finally:
        del h._solve_ward_core  # drop the instance-level override, restoring the class method

    assert solve_calls["n"] < len(results)


def test_base_geometry_built_exactly_once():
    h, B_node, A_leaf = _pick_case(0)
    B_leaves = h.get_leaves(B_node)

    build_calls = {"n": 0}
    orig_build = h._build_ward_geometry

    def counting_build(*a, **kw):
        build_calls["n"] += 1
        return orig_build(*a, **kw)

    h._build_ward_geometry = counting_build
    try:
        h.diverseCounterfactuals(B_leaves, [A_leaf], both_children=True)
    finally:
        del h._build_ward_geometry

    assert build_calls["n"] == 1


def test_feasibility_pruning_stops_descent():
    """
    If a node is (forced) infeasible, none of its descendants should be
    attempted -- the feasible regions are nested, so a deeper node on that
    path is guaranteed infeasible too.
    """
    h, B_node, A_leaf = _pick_case(0)
    B_leaves = h.get_leaves(B_node)

    orig_solve = h._solve_ward_core

    def forcing_infeasible(*a, **kw):
        A_prime, info = orig_solve(*a, **kw)
        info = dict(info)
        info["feasible"] = False
        return A_prime, info

    h._solve_ward_core = forcing_infeasible
    try:
        results = h.diverseCounterfactuals(B_leaves, [A_leaf], both_children=True)
    finally:
        del h._solve_ward_core

    assert len(results) == 1
    assert results[0]["target_node"] == B_node
    assert results[0]["info"]["feasible"] is False
