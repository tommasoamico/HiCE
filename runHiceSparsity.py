"""
Standalone replay of the HiCE (Ward) method across all sparsity modes, driven
by a results CSV that runHierarchical.py already produced.

What it does
------------
Given a --in-csv in the exact format runHierarchical.py writes (see
modules/constants.py:FIELDNAMES), for every ward-linkage (dataset, base_seed,
pair_idx) that has a 'method' or 'baycon' row in that CSV, this script:

  1. Re-derives the EXACT same dataset (X, Z) and the EXACT same (A, B, live
     competitor) pair that the original run used, by replaying
     load_openml/load_synthetic + scipy linkage + cf.enumerate_pairs with the
     same (dataset, linkage='ward', base_seed, min_leaf, max_leaf, max_pairs).
     These four enumeration parameters are NOT stored in the source CSV (only
     base_seed and pair_idx are), so you MUST pass --min-leaf/--max-leaf/
     --max-pairs matching the ORIGINAL runHierarchical.py invocation, or the
     regenerated pairs won't line up with pair_idx. This mirrors the
     reproducibility contract documented in utilityFunctions.call_seed.

  2. Re-runs BayCon for that exact pair (same call_seed as the original run),
     to recover which features it changed -- the source CSV only stores
     aggregate proximity metrics, not the translation vector itself.

  3. Runs counterfactual_ward_branch (HiCE) for that pair under every
     sparsity mode:
       'l2'           sparsity=False           (plain least-displacement)
       'l1'           sparsity=True,  lam=--lam (elastic-net feature sparsity)
       'adaptive'     sparsity='adaptive'       (data-driven lam)
       'baycon_mask'  allowed_features = the binary mask of features BayCon
                      changed for this pair (features BayCon left untouched
                      are locked to 0 movement). Skipped for a pair if BayCon
                      found no counterfactual (nothing to restrict to).

  4. Appends one row per (pair, sparsity_mode) to --out, independently
     validity-checked and proximity-scored exactly like runHierarchical.py
     does (same modules.metrics functions), so the output is directly
     comparable to the original CSV's 'method'/'baycon' rows.

Resumable, like runHierarchical.py: rows already present for a
(dataset, base_seed, pair_idx, sparsity_mode) key are skipped.

Usage
-----
    python runHiceSparsity.py --in-csv results.csv --out hice_sparsity.csv \\
        --min-leaf 1 --max-leaf 15 --max-pairs 100

    # restrict to specific datasets already present in results.csv
    python runHiceSparsity.py --in-csv results.csv --dataset iris wine
"""

import argparse
import csv
import os
import time
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from tqdm import tqdm

from modules.hiceCounterfactuals import hiceCounterfactuals
from modules.utilityFunctions import (
    MethodTimeout, time_limit, call_seed, _seed_everything,
    load_openml, load_synthetic,
)
from modules.metrics import proximity, validity, sparsity
from modules.baycon import run_baycon_baseline


SPARSITY_MODES = ['l2', 'l1', 'adaptive', 'baycon_mask']

# Revalidation numerical tolerance -- see runHierarchical.py's REVALIDATION_TOL
# for why: absorbs solver floating-point ties (~1e-11) at near-zero optimal
# displacement without weakening the actual validity definition.
REVALIDATION_TOL = 1e-9

FIELDNAMES = [
    'dataset', 'linkage', 'metric', 'pair_idx', 'sparsity_mode',
    'base_seed', 'call_seed', 'lam',
    'n_A', 'n_B', 'n_competitors', 's_A', 's_B', 's_star', 'valid',
    'l2', 'l2_sq', 'l1', 'linf', 'n_features_changed', 'sparsity',
    'solver_feasible', 'solver_optimal', 'ward_phase', 'n_active_balls',
    'allowed_features', 'n_baycon_features_changed', 'baycon_found',
    'seconds', 'aborted_timeout', 'error',
]


# ======================================================================
# Rebuilding the exact original (dataset, pairs) from the source CSV
# ======================================================================
def _read_source_csv(in_csv, linkage_filter='ward'):
    """
    Group source-CSV rows into {(dataset, base_seed): {pair_idx, ...}},
    restricted to linkage_filter and to rows that came from 'method' or
    'baycon' (sparsity modes only apply to Ward/HiCE, and we need the
    'baycon' rows' pairs to know which ones to replay BayCon for).
    """
    groups = defaultdict(set)
    with open(in_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('linkage') != linkage_filter:
                continue
            if row.get('which') not in ('method', 'baycon'):
                continue
            try:
                pair_idx = int(row['pair_idx'])
                base_seed = int(row['base_seed'])
            except (KeyError, ValueError, TypeError):
                continue
            groups[(row['dataset'], base_seed)].add(pair_idx)
    return groups


def _load_dataset_and_pairs(dataset_label, min_leaf, max_leaf, max_pairs, seed):
    synthetic = (dataset_label == 'synthetic')
    if synthetic:
        X = load_synthetic(seed=seed)
        metric = 'euclidean'
    else:
        X, metric, _ = load_openml(dataset_label, 'ward')

    Z = scipy_linkage(X, method='ward')
    cf = hiceCounterfactuals(X=X, Z=Z, metric=metric, linkage='ward', anchor='min')
    pairs = cf.enumerate_pairs(min_leaf, max_leaf, max_pairs, seed=seed)
    return X, metric, cf, pairs


# ======================================================================
# Output CSV bookkeeping (resumable, like runHierarchical.py)
# ======================================================================
def _prepare_out_csv(out_csv):
    file_exists = os.path.exists(out_csv) and os.path.getsize(out_csv) > 0
    if not file_exists:
        with open(out_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
    return file_exists


def _already_done(out_csv, file_exists):
    done = set()
    if not file_exists:
        return done
    with open(out_csv, newline='') as f:
        for row in csv.DictReader(f):
            try:
                key = (row['dataset'], int(row['base_seed']), int(row['pair_idx']),
                       row['sparsity_mode'])
            except (KeyError, ValueError):
                continue
            done.add(key)
    return done


# ======================================================================
# Main loop
# ======================================================================
def run(in_csv, out_csv, min_leaf, max_leaf, max_pairs, timeout=300, eps=0.0,
        lam=1.0, datasets_filter=None):
    groups = _read_source_csv(in_csv)
    if datasets_filter:
        groups = {k: v for k, v in groups.items() if k[0] in datasets_filter}
    if not groups:
        print(f"[skip] no ward-linkage 'method'/'baycon' rows found in {in_csv}"
              + (f" for datasets {datasets_filter}" if datasets_filter else ""))
        return []

    file_exists = _prepare_out_csv(out_csv)
    done = _already_done(out_csv, file_exists)

    csv_file = open(out_csv, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, extrasaction='ignore')

    rows = []
    try:
        for (dataset_label, base_seed), pair_idxs in groups.items():
            print(f"\n===== dataset={dataset_label} base_seed={base_seed} =====")
            try:
                X, metric, cf, pairs = _load_dataset_and_pairs(
                    dataset_label, min_leaf, max_leaf, max_pairs, base_seed)
            except ValueError as e:
                print(f"[skip] '{dataset_label}': {e}")
                continue

            for idx in tqdm(sorted(pair_idxs)):
                if idx >= len(pairs):
                    print(f"[warn] pair_idx={idx} out of range for '{dataset_label}' "
                          f"(only {len(pairs)} pairs regenerated with "
                          f"min_leaf={min_leaf}, max_leaf={max_leaf}, max_pairs={max_pairs}) "
                          f"-- these probably differ from the original run; skipping.")
                    continue
                A_lv, B_lv = pairs[idx]
                A_pts = X[sorted(A_lv)]
                B_pts = X[sorted(B_lv)]
                comps_leaves, s_A, s_B, s_star = cf.competitor_leaf_sets(A_lv, B_lv)
                comp_pts = [X[sorted(c)] for c in comps_leaves]

                # ---- replay the original BayCon call for this exact pair ----
                baycon_cseed = call_seed(dataset_label, 'ward', base_seed, idx, 'baycon')
                _seed_everything(baycon_cseed)
                allowed_mask = None
                n_baycon_feat = np.nan
                baycon_found = 0
                try:
                    with time_limit(timeout):
                        _, baycon_info = run_baycon_baseline(
                            A_pts, B_pts, comp_pts, 'ward', X_all=X,
                            mu_A=A_pts.mean(0), eps=eps, seed=baycon_cseed)
                    baycon_found = int(bool(baycon_info.get('found', False)))
                    t_baycon = baycon_info.get('t')
                    if t_baycon is not None:
                        t_baycon = np.asarray(t_baycon, dtype=float)
                        allowed_mask = (np.abs(t_baycon) > 1e-9).astype(int)
                        n_baycon_feat = int(allowed_mask.sum())
                except Exception as e:
                    print(f"[warn] BayCon replay failed for '{dataset_label}' pair {idx}: {e}")

                for mode in SPARSITY_MODES:
                    key = (dataset_label, base_seed, idx, mode)
                    if key in done:
                        continue
                    if mode == 'baycon_mask' and (allowed_mask is None or allowed_mask.sum() == 0):
                        # BayCon found nothing / changed no feature -- no mask to restrict to
                        continue

                    sparsity_kw, lam_kw, allowed_kw = {
                        'l2': (False, 0.0, None),
                        'l1': (True, lam, None),
                        'adaptive': ('adaptive', lam, None),
                        'baycon_mask': (False, 0.0, allowed_mask),
                    }[mode]

                    hice_cseed = call_seed(dataset_label, 'ward', base_seed, idx, f'hice_{mode}')
                    _seed_everything(hice_cseed)

                    err = ''; A_prime = None; info = {}; secs = np.nan
                    aborted_timeout = False
                    t0 = time.perf_counter()
                    try:
                        with time_limit(timeout):
                            A_prime, info = cf.counterfactual_ward_branch(
                                B_leaves=B_lv, A_leaves=A_lv, anchor='min',
                                sparsity=sparsity_kw, lam=lam_kw,
                                allowed_features=allowed_kw)
                        secs = time.perf_counter() - t0
                    except MethodTimeout as e:
                        secs = time.perf_counter() - t0
                        aborted_timeout = True
                        A_prime, info = None, {}
                        err = f"MethodTimeout: {e}"
                    except Exception as e:
                        secs = time.perf_counter() - t0
                        err = f"{type(e).__name__}: {e}"

                    valid = validity(A_prime, B_pts, comp_pts, 'ward', metric,
                                      eps=eps - REVALIDATION_TOL, A_orig=A_pts)
                    prox = proximity(info if isinstance(info, dict) else {}, A_pts, A_prime, metric)
                    spars = sparsity(prox['n_feat'], X.shape[1])

                    row = dict(
                        dataset=dataset_label, linkage='ward', metric=metric,
                        pair_idx=idx, sparsity_mode=mode,
                        base_seed=base_seed, call_seed=hice_cseed, lam=lam_kw,
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
                        allowed_features=('' if allowed_kw is None
                                           else ','.join(map(str, np.nonzero(allowed_kw)[0]))),
                        n_baycon_features_changed=n_baycon_feat,
                        baycon_found=baycon_found,
                        seconds=secs, aborted_timeout=int(aborted_timeout), error=err,
                    )
                    rows.append(row)
                    for k in FIELDNAMES:
                        row.setdefault(k, '')
                    writer.writerow(row)
                    csv_file.flush()
    finally:
        csv_file.close()

    if rows:
        print(f"[out] appended {len(rows)} rows to {out_csv}")
    _summary(rows)
    return rows


def _summary(rows):
    for mode in SPARSITY_MODES:
        sub = [r for r in rows if r['sparsity_mode'] == mode and not r['error']]
        if not sub:
            continue
        vrate = np.mean([r['valid'] for r in sub])
        vr = [r for r in sub if r['valid']]
        l2 = np.nanmean([r['l2'] for r in vr]) if vr else np.nan
        nf = np.nanmean([r['n_features_changed'] for r in vr]) if vr else np.nan
        print(f"[{mode}] valid={vrate:.1%}  mean L2(valid)={l2:.4f}  "
              f"mean #feat(valid)={nf:.2f}  n={len(sub)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-csv', required=True,
                     help="results CSV produced by runHierarchical.py")
    ap.add_argument('--out', type=str, default='hice_sparsity_results.csv')
    ap.add_argument('--dataset', type=str, nargs='+', default=None,
                     help="restrict to these dataset labels (default: every "
                          "ward-linkage dataset present in --in-csv)")
    ap.add_argument('--min-leaf', type=int, default=1,
                     help="MUST match the original runHierarchical.py --min-leaf")
    ap.add_argument('--max-leaf', type=int, default=15,
                     help="MUST match the original runHierarchical.py --max-leaf")
    ap.add_argument('--max-pairs', type=int, default=100,
                     help="MUST match the original runHierarchical.py --max-pairs")
    ap.add_argument('--lam', type=float, default=1.0,
                     help="L1 weight for sparsity_mode='l1' (mirrors runHierarchical's default)")
    ap.add_argument('--eps', type=float, default=0.0)
    ap.add_argument('--timeout', type=float, default=600,
                     help="per-call wall-clock limit in seconds (0/negative disables)")
    a = ap.parse_args()

    timeout = a.timeout if a.timeout and a.timeout > 0 else None
    run(a.in_csv, a.out, a.min_leaf, a.max_leaf, a.max_pairs,
        timeout=timeout, eps=a.eps, lam=a.lam, datasets_filter=a.dataset)


if __name__ == '__main__':
    main()
