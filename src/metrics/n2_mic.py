import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Callable
import matplotlib.pyplot as plt
from minepy import MINE


from src.sim import plot_grid_by_taxonomy, simulate_future_curve_data
X, Y, _ , _ = simulate_future_curve_data(T=1500, H=24, start="2010-01-01", seed=0)
x = X["x_LINp_hi_L"]


def build_lag_matrix(x: pd.Series, lags: Sequence[int]) -> pd.DataFrame:
    """
    Columns: x_t-ℓ for ℓ in lags. We do NOT align with Y here.
    """
    cols = {f"{x.name}_t-{ell}": x.shift(ell) for ell in lags}
    return pd.DataFrame(cols, index=x.index).dropna(how="all")

def align_with_horizon(Xlag: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Joint alignment for one horizon: drop rows with any NaNs (in lag cols or y).
    Returns (aligned_Xlag, aligned_y).
    """
    df = pd.concat([Xlag, y.rename("y")], axis=1).dropna()
    if df.empty:
        return Xlag.iloc[0:0], y.iloc[0:0]
    return df.drop(columns=["y"]), df["y"]

def screen_scores(
    Xlag_h: pd.DataFrame, y_h: pd.Series, method: str = "spearman"
) -> np.ndarray:
    """
    Cheap proxy: absolute Spearman/Pearson for each lag column.
    Returns array of shape (n_lags,).
    """
    meth = "spearman" if method == "spearman" else ("pearson" if method == "pearson" else "pearson")
    s = Xlag_h.corrwith(y_h, method=meth).abs()
    return s.to_numpy(dtype=float, copy=False)

def select_indices(scores: np.ndarray, keep: str, topk: Optional[int], frac: float) -> np.ndarray:
    """
    Keep a subset of lag indices given proxy scores.
    keep: "topk" | "frac" | "all"
    """
    valid = np.where(np.isfinite(scores))[0]
    if valid.size == 0:
        return np.empty(0, dtype=int)
    if keep == "all":
        return valid
    if keep == "frac":
        k = max(1, int(np.ceil(frac * valid.size)))
    else:  # "topk"
        k = valid.size if topk is None else min(topk, valid.size)
    return valid[np.argpartition(scores[valid], -k)[-k:]]


@dataclass
class N2MICFeatureResult:
    """
    Per-feature MIC grid on (lag × horizon).
    mic: shape (L, H); NaN where not computed (screened out or insufficient data).
    """
    feature: str
    lags: List[int]
    horizons: List[int]
    alpha: float
    c: int
    mic: np.ndarray
    n_used_by_h: List[int]

@dataclass
class N2MICCollection:
    """
    Container to pass into plot_grid_by_taxonomy; also handy for lookups.
    """
    lags: List[int]
    horizons: List[int]
    alpha: float
    c: int
    results: Dict[str, N2MICFeatureResult]   # feature -> result


def n2_mic_feature_screened(
    x: pd.Series,
    Y: pd.DataFrame,
    lags: Sequence[int],
    *,
    alpha: float = 0.6,
    c: int = 15,
    max_samples: Optional[int] = None,
    random_state: Optional[int] = 0,
    min_aligned: int = 5,
    screen_method: str = "spearman",   # "spearman" | "pearson"
    keep_rule: str = "topk",           # "topk" | "frac" | "all"
    topk_per_h: Optional[int] = None,  # defaults to ≈20% of lags if None
    frac_per_h: float = 0.25,
) -> N2MICFeatureResult:
    """
    Compute MIC(x_{t-ℓ}, y_{t,h}) only for screened (ℓ,h) pairs.
    """
    rng = np.random.default_rng(random_state)
    lags = list(lags)
    H = Y.shape[1]
    horizons = list(range(1, H + 1))
    L = len(lags)

    mic = np.full((L, H), np.nan, dtype=float)
    n_used_by_h = [0] * H

    # Precompute shifts once for this feature
    Xlag_all = build_lag_matrix(x, lags)

    # Default top-k ~20% of lags
    if topk_per_h is None and keep_rule == "topk":
        topk_per_h = max(3, L // 5)

    # Reuse one MINE object (compute_score resets its state)
    mine = MINE(alpha=alpha, c=c)

    for hj, col in enumerate(Y.columns):
        Xlag_h, y_h = align_with_horizon(Xlag_all, Y[col])
        n = len(y_h)
        if n < min_aligned:
            continue

        # Subsample once per horizon (same rows for all lags)
        if max_samples is not None and n > max_samples:
            idx = rng.choice(n, size=max_samples, replace=False)
            Xlag_h = Xlag_h.iloc[idx].reset_index(drop=True)
            y_h    = y_h.iloc[idx].reset_index(drop=True)
            n = len(y_h)
        n_used_by_h[hj] = n

        # 1) Screen lags cheaply
        scores = screen_scores(Xlag_h, y_h, method=screen_method)
        keep_idx = select_indices(scores, keep=keep_rule, topk=topk_per_h, frac=frac_per_h)
        if keep_idx.size == 0:
            continue

        # 2) Run MIC only on selected lags
        yv = y_h.to_numpy(dtype=float, copy=False)
        for li in keep_idx:
            xv = Xlag_h.iloc[:, li].to_numpy(dtype=float, copy=False)
            # Skip trivial/degenerate vectors
            if n < min_aligned or np.all(xv == xv[0]) or np.all(yv == yv[0]):
                continue
            mine.compute_score(xv, yv)
            mic[li, hj] = mine.mic()

    return N2MICFeatureResult(
        feature=x.name or "x",
        lags=lags,
        horizons=horizons,
        alpha=alpha,
        c=c,
        mic=mic,
        n_used_by_h=n_used_by_h,
    )

def n2_mic_all_features_screened(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    lags: Sequence[int],
    *,
    alpha: float = 0.6,
    c: int = 15,
    max_samples: Optional[int] = None,
    random_state: Optional[int] = 0,
    min_aligned: int = 5,
    screen_method: str = "spearman",
    keep_rule: str = "topk",
    topk_per_h: Optional[int] = None,
    frac_per_h: float = 0.25,
) -> N2MICCollection:
    """
    Run the screened MIC per feature independently and collect results.
    """
    lags = list(lags)
    H = Y.shape[1]
    horizons = list(range(1, H + 1))
    results: Dict[str, N2MICFeatureResult] = {}
    for feat in X.columns:
        res = n2_mic_feature_screened(
            X[feat], Y, lags,
            alpha=alpha, c=c,
            max_samples=max_samples, random_state=random_state, min_aligned=min_aligned,
            screen_method=screen_method, keep_rule=keep_rule,
            topk_per_h=topk_per_h, frac_per_h=frac_per_h,
        )
        results[feat] = res
    return N2MICCollection(lags, horizons, alpha, c, results)


def n2_mic_plot_panel_from_result(
    ax: plt.Axes,
    res: N2MICFeatureResult,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
    xlabel: str = "horizon (months)",
    ylabel: str = "lag",
    title_prefix: Optional[str] = None,
) -> None:
    """
    Draw a single feature's MIC heatmap. (Colorbar handled by the grid.)
    """
    M = res.mic
    if M.size == 0 or np.all(np.isnan(M)):
        ax.set_axis_off()
        return

    ax.imshow(M, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)

    L, H = M.shape
    y_pos = np.arange(L) if L <= 20 else np.arange(0, L, max(1, int(np.ceil(L / 10))))
    ax.set_yticks(y_pos)
    ax.set_yticklabels([res.lags[i] for i in y_pos])

    x_pos = np.arange(H) if H <= 16 else np.arange(0, H, max(1, int(np.ceil(H / 8))))
    ax.set_xticks(x_pos)
    ax.set_xticklabels([res.horizons[i] for i in x_pos])

    nmin = min([n for n in res.n_used_by_h if n > 0], default=0)
    nmax = max(res.n_used_by_h) if res.n_used_by_h else 0
    tprefix = f"{title_prefix}: " if title_prefix else ""
    ax.set_title(f"{tprefix}{res.feature} (n∈[{nmin},{nmax}])", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def n2_mic_plot_panel(
    ax: plt.Axes,
    feature: str,
    *,
    collection: N2MICCollection,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
    xlabel: str = "horizon (months)",
    ylabel: str = "lag",
    title_prefix: Optional[str] = None,
) -> None:
    """
    Adapter for plot_grid_by_taxonomy: plot_fn(ax, key, **kwargs).
    Looks up the feature's dataclass and draws it.
    """
    res = collection.results.get(feature)
    if res is None:
        ax.set_axis_off()
        return
    n2_mic_plot_panel_from_result(
        ax, res, vmin=vmin, vmax=vmax, cmap=cmap,
        xlabel=xlabel, ylabel=ylabel, title_prefix=title_prefix
    )


if __name__ == "__main__":

    mic_collection = n2_mic_all_features_screened(
        X, Y, lags=T.shape[1],
        alpha=0.6, c=15,
        max_samples=3000,
        random_state=0,
        min_aligned=5,
        screen_method="spearman",
        keep_rule="topk",
        topk_per_h=5,        # ~5 lags/horizon
        # frac_per_h=0.25,   # use if keep_rule="frac"
    )

    fig, _ = plot_grid_by_taxonomy(
        X.columns,
        plot_fn=n2_mic_plot_panel,
        grouping=("family",),
        order_within=("target","sign","strength","lag"),
        max_cols=4,
        figsize_per_cell=(3.1, 2.5),
        title=f"N2: MIC (screened, per-feature) — lags=0..{max(lags)}",
        colorbar_policy="group_right",
        plot_fn_kwargs=dict(
            collection=mic_collection,
            vmin=0.0, vmax=1.0, cmap="viridis",
            xlabel="horizon (months)", ylabel="lag"
        ),
        families_order=["LIN","LAG","MON","NMON","INT","REG","VOL","SEAS","COL","HUM","NULL"],
    )
    plt.show()
