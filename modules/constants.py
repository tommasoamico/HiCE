# ======================================================================
# Per-method presence check (for resumable, append-only runs)
# ======================================================================
FIELDNAMES = [
    'dataset', 'linkage', 'metric', 'pair_idx', 'which', 'base_seed', 'call_seed',
    'n_A', 'n_B', 'n_competitors', 's_A', 's_B', 's_star', 'valid',
    'l2', 'l2_sq', 'l1', 'linf', 'n_features_changed', 'sparsity',
    'solver_feasible', 'solver_optimal', 'ward_phase', 'n_active_balls',
    'used_fallback', 'seconds', 'aborted_timeout', 'error',
]