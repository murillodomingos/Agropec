import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


from src.sim import plot_grid_by_taxonomy, simulate_future_curve_data
X, Y, _ , _ = simulate_future_curve_data(T=1500, H=24, start="2010-01-01", seed=0)
x = X["x_LINp_hi_L"]


def window_stack_series(s: pd.Series, window: int) -> pd.DataFrame:
    """
    For a single feature series s, build lag-window columns:
      ... s_t-(w-1), ..., s_t-1, s_t-0
    """
    cols = {f"{s.name}_t-{k}": s.shift(k) for k in range(window - 1, -1, -1)}
    df = pd.DataFrame(cols, index=s.index)
    return df.dropna()


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean distances (O(n^2))."""
    G = X @ X.T
    sq = np.sum(X * X, axis=1, keepdims=True)
    D2 = sq + sq.T - 2.0 * G
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2, dtype=float)

def _pairwise_abs(y: np.ndarray) -> np.ndarray:
    """|y_i - y_j| for a 1D array."""
    return np.abs(y[:, None] - y[None, :])

def _double_center(D: np.ndarray) -> np.ndarray:
    """Biased double-centering: A = D - row_mean - col_mean + grand_mean."""
    r = D.mean(axis=1, keepdims=True)
    c = D.mean(axis=0, keepdims=True)
    g = D.mean()
    A = D - r - c + g
    np.fill_diagonal(A, 0.0)
    return A



@dataclass
class N2MarginalFeatureResult:
    """Per-feature marginal distance correlation vs horizon."""
    feature: str
    window: int
    horizons: List[int]          # [1..H]
    n_used: int                  # samples used after alignment/subsample
    dcor_by_h: np.ndarray        # shape (H,)
    index: pd.Index              # aligned time index for this feature

@dataclass
class N2MarginalCollection:
    """Convenience container to pass into the grid."""
    window: int
    results: Dict[str, N2MarginalFeatureResult]  # feature -> result



def n2_marginal_feature(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    feature: str,
    window: int,
    *,
    max_samples: Optional[int] = 1500,
    random_state: int = 0,
    compute_dtype: str = "float32",
) -> N2MarginalFeatureResult:
    """
    Compute marginal dCor(feature_window, y_h) for all horizons h,
    using *only* this feature's window and its own alignment with Y.

    No global caches are used.
    """
    if feature not in X.columns:
        raise KeyError(f"Feature '{feature}' not found in X.columns")

    rng = np.random.default_rng(random_state)

 
    Xw = window_stack_series(X[feature], window)      # (n, window)
    Y2 = Y.reindex(Xw.index).dropna()
    Xw = Xw.reindex(Y2.index).dropna()
    if len(Xw) == 0:
        H = Y.shape[1]
        return N2MarginalFeatureResult(
            feature=feature, window=window,
            horizons=list(range(1, H + 1)),
            n_used=0, dcor_by_h=np.full(H, np.nan), index=pd.Index([])
        )

    if max_samples is not None and len(Xw) > max_samples:
        take = rng.choice(len(Xw), size=max_samples, replace=False)
        Xw = Xw.iloc[take].sort_index()
        Y2 = Y2.iloc[take].sort_index()

    n, H = len(Xw), Y2.shape[1]
    Xn = Xw.values.astype(compute_dtype, copy=False)
    Yn = Y2.values.astype(compute_dtype, copy=False)

    AX = _double_center(_pairwise_euclidean(Xn))
    n2 = float(n) * float(n)
    den_x = float(np.sum(AX * AX)) / n2

    dcor = np.zeros(H, dtype=float)
    for h in range(H):
        DY = _pairwise_abs(Yn[:, h].astype(compute_dtype, copy=False))
        BY = _double_center(DY)
        num = float(np.sum(AX * BY)) / n2
        den_y = float(np.sum(BY * BY)) / n2
        dcor[h] = 0.0 if (den_x <= 0.0 or den_y <= 0.0) else num / np.sqrt(den_x * den_y)

    return N2MarginalFeatureResult(
        feature=feature,
        window=window,
        horizons=list(range(1, H + 1)),
        n_used=n,
        dcor_by_h=dcor,
        index=Xw.index,
    )

def n2_marginal_all_features(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    window: int,
    *,
    max_samples: Optional[int] = 1500,
    random_state: int = 0,
    compute_dtype: str = "float32",
) -> N2MarginalCollection:
    """
    Compute standalone marginal results for *each* feature in X.
    Each feature is processed independently (no shared caches).
    """
    results: Dict[str, N2MarginalFeatureResult] = {}
    for j, f in enumerate(X.columns):
        res = n2_marginal_feature(
            X, Y, f, window,
            max_samples=max_samples,
            random_state=random_state,
            compute_dtype=compute_dtype,
        )
        results[f] = res
    return N2MarginalCollection(window=window, results=results)


def n2_plot_panel_from_result(
    ax: plt.Axes,
    res: N2MarginalFeatureResult,
    *,
    kind: str = "line",          # "line" or "bar"
    y_min: float = 0.0,
    y_max: Optional[float] = None,
    show_max_label: bool = True,
    xlabel: str = "horizon (months)",
    ylabel: str = "dCor",
    title_prefix: Optional[str] = None,
) -> None:
    """Draw a single feature’s marginal dCor profile."""
    if res.n_used == 0 or res.dcor_by_h.size == 0:
        ax.set_axis_off()
        return

    x = res.horizons
    y = res.dcor_by_h

    if kind == "bar":
        ax.bar(x, y, width=0.9)
    else:
        ax.plot(x, y, marker="o", linewidth=1.5)

    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    y_cap = float(np.nanmax(y)) if np.isfinite(y).any() else 1.0
    ax.set_ylim(y_min, y_max if y_max is not None else max(0.01, 1.05 * y_cap))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    tprefix = f"{title_prefix}: " if title_prefix else ""
    ax.set_title(f"{tprefix}{res.feature} (n={res.n_used})", fontsize=10)

    if show_max_label and np.isfinite(y).any():
        k = int(np.nanargmax(y))
        ax.annotate(f"max@{x[k]} = {y[k]:.3f}",
                    xy=(x[k], y[k]),
                    xytext=(0.98, 0.9),
                    textcoords="axes fraction",
                    ha="right", va="top", fontsize=8)

def n2_plot_panel(
    ax: plt.Axes,
    feature: str,
    *,
    collection: N2MarginalCollection,
    kind: str = "line",
    y_min: float = 0.0,
    y_max: Optional[float] = None,
    show_max_label: bool = True,
    xlabel: str = "horizon (months)",
    ylabel: str = "dCor",
    title_prefix: Optional[str] = None,
) -> None:
    """
    Adapter for plot_grid_by_taxonomy:
      plot_fn(ax, key, **plot_fn_kwargs)
    Looks up the feature's dataclass in the collection and plots it.
    """
    res = collection.results.get(feature, None)
    if res is None:
        ax.set_axis_off()
        return
    n2_plot_panel_from_result(
        ax, res,
        kind=kind, y_min=y_min, y_max=y_max,
        show_max_label=show_max_label,
        xlabel=xlabel, ylabel=ylabel, title_prefix=title_prefix
    )


if __name__ == "__main__":
    collection = n2_marginal_all_features(
        X, Y, window=10, max_samples=1500, random_state=0, compute_dtype="float32"
    )

    fig, _ = plot_grid_by_taxonomy(
        X.columns,
        plot_fn=n2_plot_panel,
        grouping=("family",),                 # rows = family (yours)
        order_within=("target","sign","strength","lag"),
        max_cols=4,
        figsize_per_cell=(3.1, 2.5),
        title=f"N2: Marginal dCor (stand-alone per feature) — window={collection.window}",
        colorbar_policy=None,
        plot_fn_kwargs=dict(
            collection=collection,
            kind="line",
            y_min=0.0,
            y_max=None,
            show_max_label=True,
            title_prefix=None
        ),
        families_order=["LIN","LAG","MON","NMON","INT","REG","VOL","SEAS","COL","HUM","NULL"],
    )
    plt.show()
