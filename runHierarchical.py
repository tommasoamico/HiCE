"""
Experiment code for the hiceCounterfactuals class.

Matches the actual API:
    cf = hiceCounterfactuals(X, Z, metric, linkage, anchor='min')
    A_prime, info = cf.findCounterfactual(B_leaves, A_leaves)
    A_prime, info = cf.baseline(B_leaves, A_leaves)

For each (A, B) branch pair it runs the full method, the baseline, and any
enabled competitor baselines (DiCE, BayCon), and records — computed
INDEPENDENTLY of what each solver reports:
  - validity under our domination definition (recomputed from scratch),
  - proximity to the original (L2 / L2^2 / L1 / Linf / #features changed;
    coordinate-count for the Hamming relabeling case),
  - for Ward: which algorithm phase fired (info['phase']) and #active balls,
  - the solver's own feasible/optimal flags, competitor count, timing.

Dataset is taken from OpenML by name; column types are auto-detected to pick
euclidean vs hamming, with guards (Ward requires continuous; mixed -> error).

Resumable runs
--------------
The CSV is APPENDED to, and the skip check is PER-METHOD: if some methods for a
(dataset, linkage) are already in the CSV, only the missing ones are run and
appended. This lets you add a new competitor (e.g. BayCon) to an existing
results file without recomputing method/baseline/baseline2/dice. Use the SAME --seed as
the original run so the (A,B) pairs (and hence pair_idx) line up across methods.

Usage
-----
    python runHierarchical.py --dataset iris --linkage ward --out results.csv
    python runHierarchical.py --dataset iris --linkage ward --out results.csv --no-dice
    python runHierarchical.py --dataset iris wine breast-w --linkage ward --out results.csv
    python runHierarchical.py --linkage ward --synthetic          # offline

Multiple datasets
-----------------
--dataset accepts one or more names; all methods are run for each dataset in
turn and appended to the SAME --out file (resumable per-(dataset, method) as
usual).
"""

import argparse, time, csv, sys, os, random, hashlib
import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage

from modules.hiceCounterfactuals import hiceCounterfactuals
from modules.utilityFunctions import _dominates, detect_column_types, load_synthetic
from modules.metrics import proximity, validity, sparsity
from modules.dice import run_dice_baseline
from modules.baycon import run_baycon_baseline
from tqdm import tqdm


# ======================================================================
# Column-type detection + metric resolution
# ======================================================================
def resolve_metric_and_validate(df, linkage):
    types = detect_column_types(df)
    kinds = set(types.values())
    if kinds == {'continuous'}:
        return 'euclidean', types
    if kinds == {'categorical'}:
        if linkage == 'ward':
            raise ValueError(
                "Ward linkage requires continuous (Euclidean) features, but all "
                "columns were detected as categorical. Use linkage='complete' "
                "for categorical data.")
        return 'hamming', types
    raise ValueError(
        f"Mixed-type data is not supported (detected kinds {kinds}). The method "
        "requires a single ground metric (all-continuous -> Euclidean, or "
        "all-categorical -> Hamming). Mixed data would need a heterogeneous "
        f"metric such as Gower's, not implemented. Per-column types: {types}")


def load_openml(name, linkage):
    import openml, pandas as pd
    ds = openml.datasets.get_dataset(name)
    target = ds.default_target_attribute
    Xdf, _, _, _ = ds.get_data(dataset_format="dataframe", target=target)
    Xdf = Xdf.dropna(axis=1, how='any')
    metric, types = resolve_metric_and_validate(Xdf, linkage)
    if metric == 'euclidean':
        X = Xdf.to_numpy(dtype=float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    else:
        X = np.column_stack([pd.Categorical(Xdf[c]).codes.astype(int)
                             for c in Xdf.columns]).astype(int)
    return X, metric, types


# ======================================================================
# Reproducibility helpers
# ======================================================================
def _seed_everything(seed):
    """Seed every global RNG a competitor baseline might reach for (both
    baycon and DiCE-genetic draw from the GLOBAL numpy/python random state,
    not a local Generator), so a call is reproducible on its own."""
    random.seed(seed)
    np.random.seed(seed)


def call_seed(dataset_label, linkage, base_seed, pair_idx, which):
    """
    Deterministic per-(dataset, linkage, base_seed, pair_idx, which) seed in
    [0, 2**32). Depends ONLY on these five values (not on run order, other
    methods, or anything upstream), so re-deriving it later reproduces the
    exact seed used originally -- e.g. to replay just the BayCon run for one
    pair in isolation: recompute the pair with the same (dataset, linkage,
    base_seed, min_leaf, max_leaf, max_pairs) via enumerate_pairs, take
    pair_idx, then call `_seed_everything(call_seed(...))` followed by
    `run_baycon_baseline(..., seed=call_seed(...))`.
    """
    key = f"{dataset_label}|{linkage}|{base_seed}|{pair_idx}|{which}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:8], 16)


# ======================================================================
# Per-method presence check (for resumable, append-only runs)
# ======================================================================
def _methods_in_csv(out_csv, dataset_label, linkage):
    """
    Set of `which` (method names) already present for this (dataset, linkage)
    in the CSV. Empty set if the file/rows don't exist.
    """
    present = set()
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return present
    with open(out_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('dataset') == dataset_label and row.get('linkage') == linkage:
                present.add(row.get('which'))
    return present


# ======================================================================
# Main loop
# ======================================================================
def run(dataset, linkage, out_csv, min_leaf, max_leaf, max_pairs,
        synthetic=False, seed=0, eps=0.0, run_hice=True, run_dice=True, run_baycon=True):
    dataset_label = dataset if not synthetic else 'synthetic'

    if synthetic:
        X = load_synthetic(seed=seed); metric, types = 'euclidean', {}
        print(f"[data] synthetic: {X.shape} (metric=euclidean)")
    else:
        X, metric, types = load_openml(dataset, linkage)
        nc = sum(v == 'continuous' for v in types.values())
        ncat = sum(v == 'categorical' for v in types.values())
        print(f"[data] OpenML '{dataset}': {X.shape} (metric={metric}; "
              f"{nc} continuous / {ncat} categorical)")

    n = len(X)
    Z = (scipy_linkage(X.astype(float), method=linkage, metric='hamming')
         if metric == 'hamming' else scipy_linkage(X, method=linkage))
    print(f"[tree] linkage={linkage}, metric={metric}, {n} leaves")

    # one class instance, reused across all pairs
    cf = hiceCounterfactuals(X=X, Z=Z, metric=metric, linkage=linkage, anchor='min')

    # full set of methods we'd run for this config
    which_list = []
    if run_hice:
        which_list.append('method')
    which_list.append('baseline')
    if linkage == 'ward':
        which_list.append('baseline2')
    if run_dice and linkage == 'ward' and metric == 'euclidean':
        which_list.append('dice')
    if run_baycon and linkage == 'ward' and metric == 'euclidean':
        which_list.append('baycon')

    # per-method skip: only run methods NOT already present for this config
    already = _methods_in_csv(out_csv, dataset_label, linkage)
    to_run = [w for w in which_list if w not in already]
    if not to_run:
        print(f"[skip] all methods {which_list} already present for "
              f"'{dataset_label}'/{linkage} in {out_csv}; nothing to do.")
        return []
    if already:
        print(f"[resume] already present: {sorted(already)}; running only: {to_run}")

    pairs = cf.enumerate_pairs(min_leaf, max_leaf, max_pairs, seed=seed)
    print(f"[pairs] testing {len(pairs)} (A,B) pairs; methods={to_run}")

    rows = []
    for idx, (A_lv, B_lv) in enumerate(tqdm(pairs)):
        A_pts = X[sorted(A_lv)]; B_pts = X[sorted(B_lv)]
        comps_leaves, s_A, s_B, s_star = cf.competitor_leaf_sets(A_lv, B_lv)
        comp_pts = [X[sorted(c)] for c in comps_leaves]

        for which in to_run:
            # deterministic per-(dataset, linkage, seed, pair, which) seed,
            # independent of run order -- reseed every global RNG a
            # competitor baseline might touch right before calling it, so
            # the call is reproducible standalone later (see call_seed()).
            cseed = call_seed(dataset_label, linkage, seed, idx, which)
            _seed_everything(cseed)

            err = ''; A_prime = None; info = {}; secs = np.nan
            try:
                t0 = time.perf_counter()
                if which == 'method':
                    A_prime, info = cf.findCounterfactual(B_leaves=B_lv, A_leaves=A_lv)
                elif which == 'baseline':
                    A_prime, info = cf.baseline(B_leaves=B_lv, A_leaves=A_lv)
                elif which == 'baseline2':
                    A_prime, info = cf.baseline2(B_leaves=B_lv, A_leaves=A_lv)
                elif which == 'dice':
                    A_prime, info = run_dice_baseline(
                        A_pts, B_pts, comp_pts, linkage, X_all=X,
                        mu_A=A_pts.mean(0), eps=eps, seed=cseed)
                elif which == 'baycon':
                    A_prime, info = run_baycon_baseline(
                        A_pts, B_pts, comp_pts, linkage, X_all=X,
                        mu_A=A_pts.mean(0), eps=eps, seed=cseed)
                secs = time.perf_counter() - t0
            except Exception as e:
                err = f"{type(e).__name__}: {e}"

            valid = validity(A_prime, B_pts, comp_pts, linkage, metric, eps=eps)
            prox = proximity(info if isinstance(info, dict) else {}, A_pts, A_prime, metric)
            spars = sparsity(prox['n_feat'], X.shape[1])

            rows.append(dict(
                dataset=dataset_label,
                linkage=linkage, metric=metric,
                pair_idx=idx, which=which,
                base_seed=seed, call_seed=cseed,
                n_A=len(A_lv), n_B=len(B_lv), n_competitors=len(comps_leaves),
                s_A=s_A, s_B=s_B, s_star=s_star,
                valid=int(valid),
                l2=prox['l2'], l2_sq=prox['l2_sq'], l1=prox['l1'],
                linf=prox['linf'], n_features_changed=prox['n_feat'],
                sparsity=spars,
                solver_feasible=int(bool(info.get('feasible', False))) if isinstance(info, dict) else 0,
                solver_optimal=int(bool(info.get('optimal', False))) if isinstance(info, dict) else 0,
                ward_phase=info.get('phase', '') if isinstance(info, dict) else '',
                n_active_balls=(len(info.get('active_balls', []))
                                if isinstance(info, dict) and info.get('active_balls') else 0),
                used_fallback=int(bool(info.get('used_fallback', False))) if isinstance(info, dict) else 0,
                seconds=secs, error=err,
            ))

        if (idx + 1) % 25 == 0:
            print(f"  ... {idx+1}/{len(pairs)}")

    # ---- append to CSV (header only if the file is new/empty) ----
    if rows:
        file_exists = os.path.exists(out_csv) and os.path.getsize(out_csv) > 0
        # if appending to an existing file, align to its header to stay consistent
        fieldnames = list(rows[0].keys())
        existing_header = None
        existing_rows = None
        if file_exists:
            with open(out_csv, newline='') as f:
                reader = csv.DictReader(f)
                existing_header = reader.fieldnames
                if existing_header and any(k not in existing_header for k in fieldnames):
                    # new columns introduced: need to rewrite the whole file
                    # with a widened header, so the existing rows stay aligned
                    existing_rows = list(reader)
            if existing_header:
                # union: keep existing columns, add any new ones at the end
                for k in fieldnames:
                    if k not in existing_header:
                        existing_header.append(k)
                fieldnames = existing_header

        if existing_rows is not None:
            # header widened -> rewrite the file from scratch
            with open(out_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                w.writeheader()
                for r in existing_rows:
                    for k in fieldnames:
                        r.setdefault(k, '')
                    w.writerow(r)
                for r in rows:
                    for k in fieldnames:
                        r.setdefault(k, '')
                    w.writerow(r)
        else:
            with open(out_csv, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                if not file_exists:
                    w.writeheader()
                for r in rows:
                    for k in fieldnames:
                        r.setdefault(k, '')
                    w.writerow(r)
        print(f"[out] appended {len(rows)} rows to {out_csv}")

    _summary(rows, linkage)
    return rows


def _summary(rows, linkage):
    from collections import Counter
    def agg(which):
        sub = [r for r in rows if r['which'] == which and not r['error']]
        if not sub:
            errs = [r for r in rows if r['which'] == which and r['error']]
            if errs:
                print(f"\n[{which}] ALL ERRORED e.g.: {errs[0]['error'][:80]}")
            return
        vrate = np.mean([r['valid'] for r in sub])
        vr = [r for r in sub if r['valid']]
        l2 = np.nanmean([r['l2'] for r in vr]) if vr else np.nan
        nf = np.nanmean([r['n_features_changed'] for r in vr]) if vr else np.nan
        line = (f"\n[{which}] valid={vrate:.1%}  mean L2(valid)={l2:.4f}  "
                f"mean #feat(valid)={nf:.2f}  n={len(sub)}")
        if which in ('dice', 'baycon'):
            own = 1.0 - np.mean([int(r.get('used_fallback', 0) or 0) for r in sub])
            line += f"  found-own-CF={own:.1%}"
        print(line)
        if linkage == 'ward' and which == 'method':
            print(f"    phase activation: {dict(Counter(r['ward_phase'] for r in sub))}")
            print(f"    #active balls:    {dict(Counter(r['n_active_balls'] for r in sub))}")
    for w in ('method', 'baseline', 'baseline2', 'dice', 'baycon'):
        agg(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, nargs='+', default=['iris'],
                     help="one or more OpenML dataset names; all methods are "
                          "run for each, appended to the same --out file")
    ap.add_argument('--linkage', choices=['complete', 'ward'], required=True)
    ap.add_argument('--out', type=str, default='results.csv')
    ap.add_argument('--min-leaf', type=int, default=1)
    ap.add_argument('--max-leaf', type=int, default=15)
    ap.add_argument('--max-pairs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eps', type=float, default=0.0)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--no-hice', action='store_true', help="disable the HiCE method itself")
    ap.add_argument('--no-dice', action='store_true', help="disable the DiCE baseline")
    ap.add_argument('--no-baycon', action='store_true', help="disable the BayCon baseline")
    a = ap.parse_args()

    if a.synthetic:
        run(None, a.linkage, a.out, a.min_leaf, a.max_leaf, a.max_pairs,
            synthetic=True, seed=a.seed, eps=a.eps,
            run_hice=not a.no_hice, run_dice=not a.no_dice, run_baycon=not a.no_baycon)
        return

    for dataset in a.dataset:
        print(f"\n===== dataset: {dataset} =====")
        run(dataset, a.linkage, a.out, a.min_leaf, a.max_leaf, a.max_pairs,
            synthetic=False, seed=a.seed, eps=a.eps,
            run_hice=not a.no_hice, run_dice=not a.no_dice, run_baycon=not a.no_baycon)


if __name__ == '__main__':
    main()