"""
Experiment code for the hiceCounterfactuals class.

Matches the actual API:
    cf = hiceCounterfactuals(X, Z, metric, linkage, anchor='min')
    A_prime, info = cf.findCounterfactual(B_leaves, A_leaves)
    A_prime, info = cf.baseline(B_leaves, A_leaves)

For each (A, B) branch pair it runs BOTH the full method and the baseline and
records, computed INDEPENDENTLY of what the solver reports:
  - validity under our domination definition (recomputed from scratch, using
    the same `_dominates` predicate from utilityFunctions),
  - proximity to the original (L2 / L2^2 / L1 / Linf / #features changed;
    coordinate-count for the Hamming relabeling case),
  - for Ward: which algorithm phase fired (info['phase']) and #active balls,
  - the solver's own feasible/optimal flags, competitor count, timing.

Dataset is taken from OpenML by name; column types are auto-detected to pick
euclidean vs hamming, with guards (Ward requires continuous; mixed -> error).

Usage
-----
    python run_experiments.py --dataset iris --linkage ward --out results.csv
    python run_experiments.py --dataset kr-vs-kp --linkage complete --out r.csv
    python run_experiments.py --linkage ward --synthetic          # offline
"""

import argparse, time, csv, sys
import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage


from modules.hiceCounterfactuals import hiceCounterfactuals          
from modules.utilityFunctions import _dominates, detect_column_types, load_synthetic        
from modules.metrics import proximity, validity
from modules.dice import run_dice_baseline

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
    Xdf, _, _, _ = ds.get_data(
        dataset_format="dataframe",
        target=target,                           
    )
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
# Main loop
# ======================================================================
def run(dataset, linkage, out_csv, min_leaf, max_leaf, max_pairs,
        synthetic=False, seed=0, eps=0.0, run_dice=True):
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

    # which methods to run: DiCE is Ward-only for now (complete seed not implemented)
    which_list = ['method', 'baseline']
    if run_dice and linkage == 'ward' and metric == 'euclidean':
        which_list.append('dice')

    pairs = cf.enumerate_pairs(min_leaf, max_leaf, max_pairs, seed=seed)
    print(f"[pairs] testing {len(pairs)} (A,B) pairs; methods={which_list}")

    rows = []
    for idx, (A_lv, B_lv) in enumerate(pairs):
        A_pts = X[sorted(A_lv)]; B_pts = X[sorted(B_lv)]
        comps_leaves, s_A, s_B, s_star = cf.competitor_leaf_sets(A_lv, B_lv)
        comp_pts = [X[sorted(c)] for c in comps_leaves]

        for which in which_list:
            err = ''; A_prime = None; info = {}; secs = np.nan
            try:
                t0 = time.perf_counter()
                if which == 'method':
                    A_prime, info = cf.findCounterfactual(B_leaves=B_lv, A_leaves=A_lv)
                elif which == 'baseline':
                    A_prime, info = cf.baseline(B_leaves=B_lv, A_leaves=A_lv)
                elif which == 'dice':
                    A_prime, info = run_dice_baseline(
                        A_pts, B_pts, comp_pts, linkage, X_all=X,
                        mu_A=A_pts.mean(0), eps=eps)
                secs = time.perf_counter() - t0
            except Exception as e:
                err = f"{type(e).__name__}: {e}"

            
            valid = validity(A_prime, B_pts, comp_pts, linkage, metric, eps=eps)

            '''
            solver_says = (bool(info.get('feasible', False))
                                       and info.get('optimal', False)) if isinstance(info, dict) else False
            # diagnostic: solver certified optimal but harness says invalid -> mismatch
            if which == 'method' and solver_says and not valid and A_prime is not None:
                from scipy.spatial.distance import cdist
                nA = len(A_prime); muA = A_prime.mean(0)
                def wd(P):
                    nS = len(P); c = (2 * nA * nS / (nA + nS)) ** 0.5
                    return c * np.linalg.norm(muA - P.mean(0))
                dB = wd(B_pts)
                offenders = [(sorted(c), wd(Xc)) for c, Xc in zip(comps_leaves, comp_pts)
                             if not (dB < wd(Xc) - eps)]
                print(f"[MISMATCH pair {idx}] n_comp harness={len(comps_leaves)} "
                      f"solver={info.get('n_competitors')} dB={dB:.6f} "
                      f"closest_offender={min(offenders, key=lambda o: o[1]) if offenders else None}")
            '''
            prox = proximity(info if isinstance(info, dict) else {}, A_pts, A_prime, metric)

            rows.append(dict(
                dataset=dataset if not synthetic else 'synthetic',
                linkage=linkage, metric=metric,
                pair_idx=idx, which=which,
                n_A=len(A_lv), n_B=len(B_lv), n_competitors=len(comps_leaves),
                s_A=s_A, s_B=s_B, s_star=s_star,
                valid=int(valid),
                l2=prox['l2'], l2_sq=prox['l2_sq'], l1=prox['l1'],
                linf=prox['linf'], n_features_changed=prox['n_feat'],
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

    if rows:
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[out] wrote {len(rows)} rows to {out_csv}")
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
        print(f"\n[{which}] valid={vrate:.1%}  mean L2(valid)={l2:.4f}  "
              f"mean #feat(valid)={nf:.2f}  n={len(sub)}")
        if linkage == 'ward' and which == 'method':
            print(f"    phase activation: {dict(Counter(r['ward_phase'] for r in sub))}")
            print(f"    #active balls:    {dict(Counter(r['n_active_balls'] for r in sub))}")
    agg('method'); agg('baseline')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='iris')
    ap.add_argument('--linkage', choices=['complete', 'ward'], required=True)
    ap.add_argument('--out', type=str, default='results.csv')
    ap.add_argument('--min-leaf', type=int, default=1)
    ap.add_argument('--max-leaf', type=int, default=15)
    ap.add_argument('--max-pairs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eps', type=float, default=0.0)
    ap.add_argument('--synthetic', action='store_true')
    a = ap.parse_args()
    run(a.dataset, a.linkage, a.out, a.min_leaf, a.max_leaf, a.max_pairs,
        synthetic=a.synthetic, seed=a.seed, eps=a.eps)


if __name__ == '__main__':
    main()