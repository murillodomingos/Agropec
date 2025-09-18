import numpy as np
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression


from src.sim import plot_grid_by_taxonomy, simulate_future_curve_data
X, Y, _ , _ = simulate_future_curve_data(T=1500, H=24, start="2010-01-01", seed=0)
x = X["x_LINp_hi_L"]


# ---------- L3-simplified: PLS1 (feature window -> curve) ----------
def window_stack_series(s: pd.Series, window: int) -> pd.DataFrame:
    cols = {f"{s.name}_t-{k}": s.shift(k) for k in range(window - 1, -1, -1)}
    df = pd.DataFrame(cols, index=s.index)
    return df.dropna()

@dataclass
class L3PLSResult:
    name: str
    window: int
    n_components: int
    x_feature_names: List[str]
    horizons: List  # keep whatever Y columns are (e.g., ["h1",..., "h24"])
    coef: np.ndarray           # (window, H)
    x_scores: pd.DataFrame     # (T_eff, K)
    y_scores: pd.DataFrame     # (T_eff, K)
    r2_per_h: pd.Series        # length H
    r2_mean: float

def l3_pls1_feature(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    feature: str,
    window: int,
    n_components: int = 1,
    scale: bool = True,
    name: Optional[str] = None,
) -> L3PLSResult:
    """
    PLS with a single feature's lag window -> full curve Y (all horizons present in Y).
    """
    s = X[feature].astype(float)
    Xw = window_stack_series(s, window)

    # Use ALL horizons available in Y
    Y2 = Y.reindex(Xw.index).dropna()
    Xw = Xw.reindex(Y2.index).dropna()

    H = Y2.shape[1]
    K = min(n_components, window, H)

    pls = PLSRegression(n_components=K, scale=scale)
    pls.fit(Xw.values, Y2.values)
    Yhat = pls.predict(Xw.values)

    resid = Y2.values - Yhat
    sse = (resid**2).sum(axis=0)
    ycen = Y2.values - Y2.values.mean(axis=0, keepdims=True)
    sst = (ycen**2).sum(axis=0)
    r2_h = 1 - sse / np.maximum(sst, 1e-12)

    xs = pd.DataFrame(pls.x_scores_, index=Y2.index, columns=[f"t{k+1}" for k in range(K)])
    ys = pd.DataFrame(pls.y_scores_, index=Y2.index, columns=[f"u{k+1}" for k in range(K)])

    return L3PLSResult(
        name=name or f"PLS1[{feature}]",
        window=window,
        n_components=K,
        x_feature_names=list(Xw.columns),
        horizons=list(Y2.columns),        # e.g., ["h1",...,"h24"]
        coef=pls.coef_,                   # shape = (window, H)
        x_scores=xs,
        y_scores=ys,
        r2_per_h=pd.Series(r2_h, index=Y2.columns, name="R2"),
        r2_mean=float(r2_h.mean()),
    )

def l3_pls1_panel(
    ax: plt.Axes,
    feature: str,
    *,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    window: int,
    n_components: int = 1,
    scale: bool = True,
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    xtick_count: int = 8,
    ytick_count: int = 6,
):
    """
    Grid-friendly panel: heatmap of coefficients (lags × horizons) for one feature.
    Rows: lags (0..window-1, bottom→top). Cols: ALL horizons found in Y (e.g., 24).
    Returns the mappable for optional colorbar.
    """
    if feature not in X.columns:
        ax.text(0.5, 0.5, f"{feature} not in X", ha="center", va="center")
        ax.axis("off")
        return None

    res = l3_pls1_feature(X, Y, feature, window, n_components, scale=scale, name=feature)
    M = np.asarray(res.coef, dtype=float)          # (window, H)
    w, H = M.shape

    # symmetric color limits by default
    if vmin is None or vmax is None:
        a = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
        if vmin is None: vmin = -a
        if vmax is None: vmax =  a

    im = ax.imshow(M, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    xlocs = np.linspace(0, H-1, min(H, xtick_count), dtype=int)
    ax.set_xticks(xlocs)
    ax.set_xticklabels([str(res.horizons[i]) for i in xlocs])

    ylocs = np.linspace(0, w-1, min(w, ytick_count), dtype=int)
    ax.set_yticks(ylocs)
    ax.set_yticklabels([f"L{int(l)}" for l in ylocs])

    ax.set_title(f"{feature} — PLS1 coef")
    ax.set_xlabel("horizon (months)")
    ax.set_ylabel("lag")
    ax.tick_params(labelsize=8)
    return im



if __name__ == "__main__":
    fig, _ = plot_grid_by_taxonomy(
        X.columns,
        plot_fn=l3_pls1_panel,
        grouping=("family",),
        max_cols=3,
        figsize_per_cell=(3.0, 2.5),
        title="L3-simplified: PLS1 (feature window → curve)",
        colorbar_policy="group_right",
        plot_fn_kwargs=dict(X=X, Y=Y, window=24, n_components=1, scale=True, cmap="coolwarm"),
        families_order=["LIN","LAG","MON","NMON","INT","REG","VOL","SEAS","COL","HUM","NULL"],
    )
    plt.show()
