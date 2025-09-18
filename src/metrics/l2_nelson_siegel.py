import numpy as np
import pandas as pd
from typing import Optional, Sequence, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt


from src.sim import plot_grid_by_taxonomy, simulate_future_curve_data
X, Y, _ , _ = simulate_future_curve_data(T=1500, H=24, start="2010-01-01", seed=0)
x = X["x_LINp_hi_L"]


# ---------- L2: Nelson–Siegel curve factors (Level/Slope/Curvature) ----------

def _ns_basis(horizons_months: Sequence[int], lam: float) -> np.ndarray:
    tau = np.asarray(horizons_months, dtype=float) / 12.0
    f1 = np.ones_like(tau)
    f2 = (1 - np.exp(-lam * tau)) / (lam * tau)
    f3 = f2 - np.exp(-lam * tau)
    return np.column_stack([f1, f2, f3])  # (H,3)

def _fit_ns_for_lambda(Y: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    pinv = np.linalg.pinv(B)             # (3,H)
    betas = Y @ pinv.T                   # (T,3)
    Yhat = betas @ B.T                   # (T,H)
    resid = Y - Yhat
    sse = np.sum(resid**2, axis=1)
    sst = np.sum((Y - Y.mean(axis=1, keepdims=True))**2, axis=1)
    r2 = 1 - sse / np.maximum(sst, 1e-12)
    return betas, r2, float(np.nanmean(r2)), float(np.nanmedian(r2))

@dataclass
class L2NSResult:
    """
    L2 | Nelson–Siegel factorization of the curve.
    Basis: B(τ,λ) = [1, (1-e^{-λτ})/(λτ), (1-e^{-λτ})/(λτ) - e^{-λτ}], τ in years.
    For fixed λ, betas_t = argmin ||y_t - B beta_t||_2 via least squares.
    Outputs Level/Slope/Curvature series and fit quality; optionally correlations with a driver x at given lags.
    Decision: λ chosen by grid-search maximizing mean R^2 across t (if not provided).
    """
    name: str
    lambda_: float
    horizons: Sequence[int]
    betas: pd.DataFrame         # columns: ["level","slope","curvature"]
    r2_by_time: pd.Series
    r2_mean: float
    r2_median: float
    corr_by_lag: Optional[pd.DataFrame]   # index = lag, columns = L/S/C
    best_by_factor: Optional[pd.DataFrame]

def l2_ns_factors(
    Y: pd.DataFrame,
    x: Optional[pd.Series] = None,
    lags: Sequence[int] = (0,),
    lam: Optional[float] = None,
    lam_grid: Optional[Sequence[float]] = None,
    name: Optional[str] = None
) -> L2NSResult:
    """
    Fit Nelson–Siegel factors to each daily curve and (optionally) correlate x_{t-ℓ} with factors_t.
    If λ not given, choose λ from lam_grid (default 30 values in [0.05,1.50]) maximizing mean R^2.
    """
    Y2 = Y.dropna()
    idx = Y2.index
    H = Y2.shape[1]
    horizons = list(range(1, H + 1))
    Ynp = Y2.to_numpy(dtype=float)

    if lam is None:
        grid = np.linspace(0.05, 1.50, 30) if lam_grid is None else np.asarray(lam_grid, dtype=float)
        best_lam, best_mean = grid[0], -np.inf
        for l in grid:
            B = _ns_basis(horizons, l)
            _, _, r2_mean, _ = _fit_ns_for_lambda(Ynp, B)
            if r2_mean > best_mean:
                best_mean, best_lam = r2_mean, l
        lam = float(best_lam)

    B = _ns_basis(horizons, lam)
    betas_np, r2, r2_mean, r2_median = _fit_ns_for_lambda(Ynp, B)
    betas = pd.DataFrame(betas_np, index=idx, columns=["level","slope","curvature"])
    r2_s = pd.Series(r2, index=idx, name="r2")

    corr_by_lag = None
    best_by_factor = None
    if x is not None:
        z = x.reindex(idx).astype(float)
        rows = []
        for ell in lags:
            xs = z.shift(ell)
            df = pd.concat([xs.rename("x"), betas], axis=1).dropna()
            vals = [df["x"].corr(df[c]) for c in ["level","slope","curvature"]]
            rows.append([ell] + vals)
        corr_by_lag = pd.DataFrame(rows, columns=["lag","level","slope","curvature"]).set_index("lag")
        best = []
        for c in ["level","slope","curvature"]:
            k = corr_by_lag[c].abs().idxmax()
            best.append((c, int(k), float(corr_by_lag.loc[k, c])))
        best_by_factor = pd.DataFrame(best, columns=["factor","lag_of_max_abs","rho_at_max_abs"])

    return L2NSResult(
        name=name or (x.name if x is not None and x.name else "NS"),
        lambda_=lam,
        horizons=horizons,
        betas=betas,
        r2_by_time=r2_s,
        r2_mean=r2_mean,
        r2_median=r2_median,
        corr_by_lag=corr_by_lag,
        best_by_factor=best_by_factor
    )

# ---------- Plots as reusable functions (for grid or standalone) ----------

def l2_plot_r2_timeseries_panel(
    ax: plt.Axes,
    feature: str,                     # unused (kept for grid compatibility)
    *,
    Y: pd.DataFrame,
    lam: Optional[float] = None,
    lam_grid: Optional[Sequence[float]] = None,
):
    """
    Panel that draws the per-date R^2 time series of the NS fit (independent of any x).
    IO kept grid-compatible: (ax, feature, **kwargs). 'feature' is ignored.
    """
    res = l2_ns_factors(Y=Y, x=None, lam=lam, lam_grid=lam_grid)
    res.r2_by_time.plot(ax=ax)
    ax.set_title("Per-date R$^2$ of NS fit")
    ax.set_xlabel("date"); ax.set_ylabel("R$^2$")
    ax.figure.tight_layout()
    return None  # no colorbar

def l2_plot_factor_series_panel(
    ax: plt.Axes,
    feature: str,                     # unused (kept for grid compatibility)
    *,
    Y: pd.DataFrame,
    lam: Optional[float] = None,
    lam_grid: Optional[Sequence[float]] = None,
):
    """
    Panel that draws the level/slope/curvature time series (independent of any x).
    IO kept grid-compatible: (ax, feature, **kwargs). 'feature' is ignored.
    """
    res = l2_ns_factors(Y=Y, x=None, lam=lam, lam_grid=lam_grid)
    ax.plot(res.betas.index, res.betas["level"], label="level")
    ax.plot(res.betas.index, res.betas["slope"], label="slope")
    ax.plot(res.betas.index, res.betas["curvature"], label="curvature")
    ax.legend(); ax.set_title("NS factor scores")
    ax.figure.tight_layout()
    return None  # no colorbar

def l2_plot_corr_heatmap(
    ax: plt.Axes,
    feature: str,
    *,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    lags: Sequence[int] = range(0, 15),
    lam: Optional[float] = None,
    lam_grid: Optional[Sequence[float]] = None,
    cmap: str = "coolwarm",
    vmin: Optional[float] = -1.0,
    vmax: Optional[float] =  1.0,
):
    """
    Heatmap panel: rows = lags ℓ, columns = factors k∈{L,S,C}, values = Corr(x_{t-ℓ}, factor_t).
    Returns the mappable for optional colorbar in grouped grids.
    """
    if feature not in X.columns:
        ax.text(0.5, 0.5, f"{feature} not in X", ha="center", va="center")
        ax.axis("off")
        return None

    res = l2_ns_factors(Y=Y, x=X[feature], lags=lags, lam=lam, lam_grid=lam_grid, name=feature)
    if res.corr_by_lag is None or res.corr_by_lag.empty:
        ax.text(0.5, 0.5, "no correlations", ha="center", va="center")
        ax.axis("off")
        return None

    M = res.corr_by_lag[["level","slope","curvature"]].to_numpy(dtype=float)
    im = ax.imshow(M, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["level","slope","curvature"])
    ax.set_yticks(range(len(res.corr_by_lag.index)))
    ax.set_yticklabels(list(res.corr_by_lag.index))
    ax.set_xlabel("factor k"); ax.set_ylabel("lag ℓ")
    ax.set_title(f"ρ(x_{{t-ℓ}}, k_t) — {feature}")
    return im

# ---------- Text+plots report (kept same 2 plots, removed basis, added heatmap) ----------

def l2_ns_report(res: L2NSResult) -> None:
    """
    Markdown tables and simple plots: λ, fit quality, correlations by lag,
    factor series, and correlation heatmap (lags × factors).
    """
    print(f"# L2: Nelson–Siegel factors — {res.name}")
    print(f"- lambda: {res.lambda_:.4f}")
    print(f"- R^2 mean: {res.r2_mean:.3f} | median: {res.r2_median:.3f}")

    print("\n## Sample of factor scores (head)")
    print(res.betas.head(10).to_markdown())

    if res.corr_by_lag is not None:
        print("\n## Correlation with factors by lag")
        print(res.corr_by_lag.to_markdown())
    if res.best_by_factor is not None:
        print("\n## Best lag per factor")
        print(res.best_by_factor.to_markdown(index=False))

    # --- Plot 1: Per-date R^2 (unchanged) ---
    plt.figure(figsize=(9,4))
    res.r2_by_time.plot()
    plt.title("Per-date R^2 of NS fit")
    plt.xlabel("date"); plt.ylabel("R^2"); plt.tight_layout(); plt.show()

    # --- Plot 2: Factor series (unchanged) ---
    plt.figure(figsize=(9,4))
    plt.plot(res.betas.index, res.betas["level"], label="level")
    plt.plot(res.betas.index, res.betas["slope"], label="slope")
    plt.plot(res.betas.index, res.betas["curvature"], label="curvature")
    plt.legend(); plt.title("NS factor scores"); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    res_ns = l2_ns_factors(Y=Y, x=x, lags=range(0,15), lam=None)
    l2_ns_report(res_ns)


    fig, _ = plot_grid_by_taxonomy(
        X.columns,
        plot_fn=l2_plot_corr_heatmap,
        grouping=("family",),   # rows by family
        max_cols=3,
        figsize_per_cell=(3.0, 2.4),
        title="L2: Corr(x_{t-ℓ}, NS factors)",
        colorbar_policy="group_right",
        plot_fn_kwargs=dict(X=X, Y=Y, lags=range(0,15), vmin=-1, vmax=1, cmap="coolwarm"),
        families_order=["LIN","LAG","MON","NMON","INT","REG","VOL","SEAS","COL","HUM","NULL"],
    )
    plt.show()


