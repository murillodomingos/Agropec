import numpy as np
import pandas as pd
from typing import Optional, Sequence, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
from scipy.stats import energy_distance, wasserstein_distance
from ot.sliced import sliced_wasserstein_distance

# ---------- D1: Energy Distance (High vs Low regimes of x) ----------

def _split_high_low(x: pd.Series, q: float) -> Tuple[pd.Index, pd.Index, float, float]:
    q_hi = x.quantile(q); q_lo = x.quantile(1 - q)
    high_idx = x.index[x >= q_hi]
    low_idx = x.index[x <= q_lo]
    return high_idx, low_idx, float(q_lo), float(q_hi)

def _energy_distance_mv(X: np.ndarray, Y: np.ndarray) -> float:
    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return np.nan
    s_xy = cdist(X, Y, metric="euclidean").mean()
    s_xx = 0.0 if n < 2 else (2.0 / (n * n)) * pdist(X, metric="euclidean").sum()
    s_yy = 0.0 if m < 2 else (2.0 / (m * m)) * pdist(Y, metric="euclidean").sum()
    return 2.0 * s_xy - s_xx - s_yy

def _perm_pvalue_labels(obs: float,
                        A: np.ndarray,
                        B: np.ndarray,
                        n_perm: int,
                        mv: bool,
                        rng: np.random.Generator) -> float:
    X = np.vstack([A, B])
    n, m = len(A), len(B)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(X))
        A2 = X[perm[:n]]
        B2 = X[perm[n:]]
        stat = _energy_distance_mv(A2, B2) if mv else energy_distance(A2.ravel(), B2.ravel())
        if stat >= obs - 1e-15:
            cnt += 1
    return (cnt + 1.0) / (n_perm + 1.0)

@dataclass
class D1EnergyHLResult:
    """
    D1 | Energy distance between future curves under High vs Low regimes of x.
    High/Low sets defined by thresholds (x ≥ Q_q, x ≤ Q_{1−q}); default q=0.7 (top/bottom 30%).
    Energy distance (multivariate): D_E = 2 E||X−Y|| − E||X−X'|| − E||Y−Y'|| with Euclidean norm.
    Outputs: global curve-level D_E with permutation p-value; per-horizon 1D D_E and p-values.
    Decision: High–Low (not deciles) to keep groups reasonably large and separated; label-permutations for p-values.
    """
    name: str
    q: float
    thr_low: float
    thr_high: float
    n_high: int
    n_low: int
    horizons: List[int]
    energy_full: float
    pval_full: float
    energy_by_h: np.ndarray
    pval_by_h: np.ndarray

def d1_energy_highlow(
    x: pd.Series,
    Y: pd.DataFrame,
    q: float = 0.7,
    n_perm: int = 499,
    max_rows: Optional[int] = None,
    random_state: Optional[int] = 0,
    name: Optional[str] = None
) -> D1EnergyHLResult:
    """
    Compute energy distance between Y-curves when x is High (≥ Q_q) vs Low (≤ Q_{1−q}).
    Global multivariate D_E with permutation p-value; per-horizon 1D D_E and p-values.
    """
    x2 = x.reindex(Y.index).dropna()
    Y2 = Y.reindex(x2.index).dropna()
    x2 = x2.reindex(Y2.index)
    hi_idx, lo_idx, thr_lo, thr_hi = _split_high_low(x2, q)
    Y_hi = Y2.loc[hi_idx]
    Y_lo = Y2.loc[lo_idx]
    if max_rows is not None:
        rng = np.random.default_rng(random_state)
        if len(Y_hi) > max_rows:
            Y_hi = Y_hi.iloc[rng.choice(len(Y_hi), size=max_rows, replace=False)]
        if len(Y_lo) > max_rows:
            Y_lo = Y_lo.iloc[rng.choice(len(Y_lo), size=max_rows, replace=False)]
    A = Y_hi.values.astype(float)
    B = Y_lo.values.astype(float)
    rng = np.random.default_rng(random_state)
    de_full = _energy_distance_mv(A, B)
    p_full = _perm_pvalue_labels(de_full, A, B, n_perm=n_perm, mv=True, rng=rng)
    H = Y2.shape[1]
    de_h = np.zeros(H); pv_h = np.zeros(H)
    for j in range(H):
        a = A[:, j]
        b = B[:, j]
        de_h[j] = energy_distance(a, b)
        pv_h[j] = _perm_pvalue_labels(de_h[j], a.reshape(-1,1), b.reshape(-1,1), n_perm=n_perm, mv=False, rng=rng)
    return D1EnergyHLResult(
        name=name or (x.name if x.name else "EnergyHL"),
        q=q,
        thr_low=thr_lo,
        thr_high=thr_hi,
        n_high=len(A),
        n_low=len(B),
        horizons=list(range(1, H+1)),
        energy_full=de_full,
        pval_full=p_full,
        energy_by_h=de_h,
        pval_by_h=pv_h
    )

def d1_energy_report(res: D1EnergyHLResult, top_k: int = 12) -> None:
    """
    Markdown summary and visuals for D1 results.
    """
    print(f"# D1: Energy Distance (High–Low) — {res.name}")
    print(f"- q: {res.q:.2f} | thresholds: [{res.thr_low:.3f}, {res.thr_high:.3f}] | sizes: high={res.n_high}, low={res.n_low}")
    print(f"- Global curve D_E: {res.energy_full:.4f} (perm p={res.pval_full:.4g})")
    df = pd.DataFrame({"horizon": res.horizons, "D_E": res.energy_by_h, "pval": res.pval_by_h})
    print("\n## Per-horizon energy distance (top)")
    print(df.sort_values("D_E", ascending=False).head(top_k).to_markdown(index=False))
    plt.figure(figsize=(8,4))
    plt.bar(df["horizon"], df["D_E"])
    plt.xlabel("horizon (months)"); plt.ylabel("energy distance (1D)")
    plt.title("Per-horizon energy distance (High vs Low)"); plt.tight_layout(); plt.show()



