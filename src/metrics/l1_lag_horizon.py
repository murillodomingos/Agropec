import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import lfilter
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


from src.sim import plot_grid_by_taxonomy, simulate_future_curve_data
X, Y, _ , _ = simulate_future_curve_data(T=1500, H=24, start="2010-01-01", seed=0)
x = X["x_LINp_hi_L"]


# ---------- L1: Lag–Horizon CCF with AR prewhitening ----------
@dataclass
class L1CCFResult:
    """
    Lag–Horizon cross-correlation surface with AR(p) prewhitening.
    ρ_{ℓ,h} = Corr( x_{t-ℓ}^pw , y_{t,h}^pw ), ℓ=0..L, h=1..H.
    AR order p chosen by AIC over p=0..p_max on x_t; same AR filter applied to y_{·,h}.
    """
    name: str
    lags: int
    horizons: Sequence[int]
    ar_order: int
    ar_params: np.ndarray
    rho: np.ndarray  # (lags+1, H)
    columns: Sequence[str]

def _best_ar_order_aic(x: np.ndarray, p_max: int) -> Tuple[int, np.ndarray]:
    best_p, best_aic, best_params = 0, np.inf, np.array([])
    for p in range(p_max + 1):
        try:
            res = ARIMA(x, order=(p, 0, 0), trend='n').fit(method='statespace', disp=0)
            if res.aic < best_aic:
                best_aic, best_p = res.aic, p
                best_params = getattr(res, "arparams", np.array([]))
        except Exception:
            continue
    return best_p, best_params

def l1_ccf_prewhiten(
    x: pd.Series,
    Y: pd.DataFrame,
    max_lag: int,
    ar_maxlag: int = 12,
    name: Optional[str] = None
) -> L1CCFResult:
    """
    L1 | Prewhiten x_t via AR(p) (AIC over p=0..ar_maxlag), apply same AR filter to each y_{·,h},
    then compute ρ_{ℓ,h} = Corr(x_{t-ℓ}^pw, y_{t,h}^pw) for ℓ=0..max_lag and all horizons h.
    """
    df = Y.join(x.rename("x"), how="inner").dropna()
    y_mat = df[Y.columns].values.astype(float)
    x_vec = df["x"].values.astype(float)
    p, arparams = _best_ar_order_aic(x_vec, ar_maxlag)
    if p > 0:
        b = np.r_[1.0, -arparams]
        xp = lfilter(b, [1.0], x_vec)
        yp = lfilter(b, [1.0], y_mat, axis=0)
    else:
        xp = x_vec.copy()
        yp = y_mat.copy()
    xp = xp[p:]
    yp = yp[p:, :]
    n = xp.shape[0]
    H = yp.shape[1]
    L = min(max_lag, n - 2)
    rho = np.full((L + 1, H), np.nan, dtype=float)
    for h in range(H):
        ycol = yp[:, h]
        xm = xp - xp.mean(); ym = ycol - ycol.mean()
        for ell in range(L + 1):
            if ell == 0:
                x0, y0 = xm, ym
            else:
                x0, y0 = xm[:-ell], ym[ell:]
            if len(x0) > 1 and np.std(x0) > 0 and np.std(y0) > 0:
                rho[ell, h] = np.corrcoef(x0, y0)[0, 1]
    return L1CCFResult(
        name=name or (x.name if x.name else "x"),
        lags=L,
        horizons=list(range(1, H + 1)),
        ar_order=p,
        ar_params=arparams,
        rho=rho,
        columns=list(Y.columns)
    )

def l1_plot_panel(
    ax: plt.Axes,
    feature: str,
    *,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    max_lag: int = 30,
    ar_maxlag: int = 12,
    cmap: str = "viridis",      # change if you like
    vmin: Optional[float] = -1, # fix to [-1,1] to make panels comparable
    vmax: Optional[float] =  1,
):
    """
    Draw a single L1-CCF heatmap for `feature` onto `ax`.
    Returns the 'mappable' from imshow for optional colorbar placement.
    """
    if feature not in X.columns:
        ax.text(0.5, 0.5, f"{feature} not in X", ha="center", va="center")
        ax.axis("off")
        return None

    res = l1_ccf_prewhiten(
        x=X[feature], Y=Y, max_lag=max_lag, ar_maxlag=ar_maxlag, name=feature
    )
    im = ax.imshow(res.rho, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    # Light axis labels to keep grid tidy
    ax.set_xlabel("h", fontsize=8)
    ax.set_ylabel("lag", fontsize=8)
    # Coarser ticks
    H = len(res.horizons)
    L = res.lags
    ax.set_xticks(range(0, H, max(1, H // 6)))
    ax.set_yticks(range(0, L + 1, max(1, L // 5)))
    return im



if __name__ == "__main__":


    fig, _ = plot_grid_by_taxonomy(
        X.columns,
        plot_fn=l1_plot_panel,
        grouping=("family",),                  # rows = family
        order_within=("target","sign","strength","lag"),
        max_cols=4,
        figsize_per_cell=(3.1, 2.5),
        title="L1: Lag–Horizon CCF (prewhitened) — grouped by family",
        colorbar_policy="group_right",
        plot_fn_kwargs=dict(X=X, Y=Y, max_lag=30, ar_maxlag=12, vmin=-1, vmax=1),
        families_order=["LIN","LAG","MON","NMON","INT","REG","VOL","SEAS","COL","HUM","NULL"],
    )
    plt.show()


