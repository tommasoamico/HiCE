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

# Validated categorical palette (see dataviz skill, references/palette.md).
# Fixed hue order, assigned by method identity (not by rank/performance).
PALETTE = {
    "method": "#2a78d6",    # blue
    "baseline": "#eb6834",  # orange
    "baseline2": "#e34948", # red
    "dice": "#1baf7a",      # aqua
    "baycon": "#eda100",    # yellow
    "baycon_random": "#c9820a",    # amber (BayCon, random k of its pool)
    "baycon_proximal": "#8a5a00",  # dark amber (BayCon, closest k of its pool)
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e3e2dc"

METHOD_ORDER = ["method", "baseline", "baseline2", "dice", "baycon",
                "baycon_random", "baycon_proximal"]
METHOD_LABEL = {
    "method": "HiCE",
    "baseline": "Baseline",
    "baseline2": "Baseline 2",
    "dice": "DiCE",
    "baycon": "BayCon",
    "baycon_random": "BayCon (random)",
    "baycon_proximal": "BayCon (proximal)",
}

# ======================================================================
# Complete-linkage single-CF comparison (runCompleteLinkage.py). Same base
# schema as FIELDNAMES so its rows stay directly comparable to the Ward
# single-CF runs (the `linkage` column is what distinguishes them), plus the
# complete-linkage solver's own diagnostics. `ward_phase` carries the generic
# solver-phase string here too ('miqp-rowgen', 'trivial', ...), and
# `n_active_balls` stays blank -- it is a Ward sphere-repair notion with no
# complete-linkage analogue.
# ======================================================================
COMPLETE_FIELDNAMES = FIELDNAMES + [
    'rowgen_rounds', 'rowgen_rows', 'rowgen_converged', 'rowgen_stalled_blocks',
    'revalidated', 'margin_unsquared', 'one_class_pool', 'seed_valid',
]

# ======================================================================
# Diversity experiment (runHierarchical.py): produce k counterfactuals per
# (A, B) branch pair per method, instead of one -- schemas below.
# ======================================================================
DIVERSITY_RAW_FIELDNAMES = [
    'dataset', 'linkage', 'metric', 'pair_idx', 'which', 'base_seed', 'call_seed',
    'n_A', 'n_B', 'cf_idx', 'valid', 'l2', 'seconds', 'error',
]
DIVERSITY_SUMMARY_FIELDNAMES = [
    'dataset', 'linkage', 'metric', 'pair_idx', 'which', 'base_seed', 'call_seed',
    'n_A', 'n_B', 'k_requested', 'n_returned', 'n_valid', 'n_degenerate_ties',
    'scale', 'avg_proximity', 'avg_pairwise_distance',
    'avg_proximity_norm', 'avg_pairwise_distance_norm',
    'n_subbranches', 'n_subbranches_represented',
    'tree_coverage', 'tree_coverage_norm',
    'seconds', 'error',
]

# Colors handed out (in order) to methods not covered by PALETTE above --
# e.g. per-sparsity-mode HiCE variants coming from runHiceSparsity.py.
EXTRA_COLORS = ["#8456ce", "#4a3aa7", "#c14fa5", "#0b8fa3", "#8a8f00", "#c96f1e"]