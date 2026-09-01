from modules.utilityFunctions import _dominates
import numpy as np
from scipy.spatial.distance import cdist


def validity(A_prime, B_pts, comp_pts, linkage, metric, eps=0.0):
    """Reuse the class's _dominates for euclidean; hamming handled explicitly."""
    if A_prime is None:
        return False
    A_prime = np.asarray(A_prime)
    if metric == 'hamming':
        d = A_prime.shape[1]
        def hmax(P, Q):
            return cdist(P.astype(float), Q.astype(float), metric='hamming').max() * d
        dB = hmax(A_prime, B_pts)
        return all(dB < hmax(A_prime, np.asarray(Xp)) - eps for Xp in comp_pts)
    # euclidean -> your predicate (complete uses max, ward uses weighted centroid)
    return _dominates(A_prime, np.asarray(B_pts),
                      [np.asarray(c) for c in comp_pts], linkage, eps)


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
