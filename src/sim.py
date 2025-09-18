import re
import numpy as np
import pandas as pd
from typing import *
import matplotlib.pyplot as plt
from matplotlib import gridspec



# --------- Taxonomy parsing (fits names you used above) ---------
_FAMILIES = ["LIN", "LAG", "MON", "NMON", "INT", "REG", "VOL", "SEAS", "COL", "NULL", "HUM"]
_SIGNS    = ["p", "n"]
_STRENGTH = ["lo", "mid", "hi"]
_TARGETS  = ["L", "S", "C", "front", "mid", "back"]
_SHAPES   = ["sat", "quad", "hump", "midonly", "amp"]


def parse_feature_taxonomy(name: str) -> Dict[str, Optional[str]]:
    """
    Parse a feature name like 'x_LAGp_hi_S_L3' into components:
    family, sign, shape, strength, target, lag (int or None).
    It’s robust to missing parts; everything defaults to None if absent.
    """
    out = dict(family=None, sign=None, shape=None, strength=None, target=None, lag=None)
    base = name.split("x_", 1)[-1] if name.startswith("x_") else name
    tokens = base.split("_")

    t0 = tokens[0] if tokens else ""
    fam = None
    for F in sorted(_FAMILIES, key=len, reverse=True):
        if t0.startswith(F):
            fam = F
            rest = t0[len(F):]
            break
    out["family"] = fam

    if rest and len(rest) > 0 and rest[0] in _SIGNS:
        out["sign"] = rest[0]
        rest = rest[1:]
    for sh in _SHAPES:
        if rest.startswith(sh):
            out["shape"] = sh
            rest = rest[len(sh):]
            break

    for tk in tokens[1:]:
        if tk in _STRENGTH:
            out["strength"] = tk
        elif tk in _TARGETS:
            out["target"] = tk
        elif re.fullmatch(r"L\d+", tk):          # e.g., L3
            out["lag"] = int(tk[1:])
        elif tk in _SHAPES and out["shape"] is None:
            out["shape"] = tk
    return out




def plot_grid_by_taxonomy(
    features: Sequence[str],
    plot_fn: Callable[..., Optional[object]],
    *,
    grouping: Tuple[str, ...] = ("family",),
    order_within: Tuple[str, ...] = ("target","sign","strength","lag"),
    max_cols: int = 3,
    figsize_per_cell: Tuple[float, float] = (3.0, 2.4),
    title: Optional[str] = None,
    colorbar_policy: str = "group_right",         # 'none' | 'group_right'
    colorbar_width: float = 0.06,                  # width ratio for cbar column
    colorbar_kwargs: Optional[dict] = None,
    parse_fn: Callable[[str], Dict[str, Optional[str]]] = parse_feature_taxonomy,
    families_order: Optional[List[str]] = None,
    plot_fn_kwargs: Optional[dict] = None,
    fig_xlabel: str = "h",
    fig_ylabel: str = "lag",
    title_height_ratio: float = 0.075,            # dedicated title row height (relative)
):
    """
    Creates a tidy, aligned grid:
      - Each row = group (e.g., family) with N panels + 1 narrow cbar column.
      - Uniform panel widths; figure-level x/y labels; dedicated title row.
    """
    plot_fn_kwargs = plot_fn_kwargs or {}
    colorbar_kwargs = colorbar_kwargs or {}

    # ---- parse + group ----
    meta = {f: parse_fn(f) for f in features}
    def gkey(f): return tuple(meta[f].get(k) for k in grouping)
    groups: Dict[Tuple, List[str]] = {}
    for f in features:
        groups.setdefault(gkey(f), []).append(f)

    if families_order and len(grouping)==1 and grouping[0]=="family":
        fam_to_key = {g[0]: g for g in groups.keys()}
        ordered = [fam_to_key[x] for x in families_order if x in fam_to_key]
        leftovers = [k for k in groups.keys() if k not in ordered]
        group_order = ordered + leftovers
    else:
        def none_last(t): return tuple(("_" if v is None else str(v)) for v in t)
        group_order = sorted(groups.keys(), key=none_last)

    def sort_key(f):
        m = meta[f]
        key=[]
        for k in order_within:
            v = m.get(k)
            if v is None: key.append(("~", 9999))
            elif k=="lag": key.append(("", int(v)))
            else:          key.append((str(v), 0))
        key.append((f,0))
        return tuple(key)

    for g in groups:
        groups[g] = sorted(groups[g], key=sort_key)

    # ---- figure geometry ----
    rows_info = []
    max_used_cols = 1
    for g in group_order:
        n = len(groups[g])
        ncols = min(max_cols, max(1, n))
        nrows = int(np.ceil(n / ncols))
        rows_info.append((g, nrows, ncols, n))
        max_used_cols = max(max_used_cols, ncols)

    cell_w, cell_h = figsize_per_cell
    fig_w = cell_w * max_used_cols * (1.0 + (colorbar_policy=="group_right")*colorbar_width/max(1e-6,1.0))
    fig_h = sum(nrows * cell_h + 0.35 for _, nrows, _, _ in rows_info)

    # --- master GridSpec with a dedicated title row ---
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs_master = gridspec.GridSpec(
        nrows=len(rows_info) + 1, ncols=1, figure=fig,
        height_ratios=[title_height_ratio] + [1] * len(rows_info),
        hspace=0.45
    )
    ax_title = fig.add_subplot(gs_master[0, 0])
    ax_title.axis("off")
    if title:
        ax_title.text(0.5, 0.2, title, ha="center", va="bottom",
                      fontsize=12, fontweight="bold")

    group_axes_map: Dict[Tuple, List[plt.Axes]] = {}

    # ---- draw each group row ----
    for gi, (g, nrows, ncols, n) in enumerate(rows_info):
        add_cbar_col = (colorbar_policy == "group_right")
        total_cols = ncols + (1 if add_cbar_col else 0)
        wr = [1.0]*ncols + ([colorbar_width] if add_cbar_col else [])

        sub = gridspec.GridSpecFromSubplotSpec(
            nrows=nrows, ncols=total_cols, subplot_spec=gs_master[gi+1, 0],
            wspace=0.10, hspace=0.12, width_ratios=wr
        )

        feats = groups[g]
        mapp_for_cbar = None
        axes_this_group: List[plt.Axes] = []

        row_label = " • ".join(f"{k}={v if v is not None else '—'}" for k,v in zip(grouping,g))

        for i, f in enumerate(feats):
            r, c = divmod(i, ncols)
            ax = fig.add_subplot(sub[r, c])
            axes_this_group.append(ax)

            mm = plot_fn(ax=ax, feature=f, **plot_fn_kwargs)  # panel draw
            if mm is not None:
                mapp_for_cbar = mm

            ax.set_title(f, fontsize=9)
            if c != 0:  # hide inner y tick labels
                ax.set_yticklabels([]); ax.set_ylabel("")
            if r != nrows - 1:  # hide inner x tick labels
                ax.set_xticklabels([]); ax.set_xlabel("")
            ax.tick_params(labelsize=8)

        # fill empties
        total_slots = nrows * ncols
        for j in range(len(feats), total_slots):
            r, c = divmod(j, ncols)
            ax = fig.add_subplot(sub[r, c]); ax.axis("off")
            axes_this_group.append(ax)

        if feats:
            ax0 = axes_this_group[0]
            ax0.annotate(row_label, xy=(0, 1.02), xycoords="axes fraction",
                         ha="left", va="bottom", fontsize=10, fontweight="bold")

        if add_cbar_col and mapp_for_cbar is not None and feats:
            cax = fig.add_subplot(sub[:, -1])
            cbar = fig.colorbar(mapp_for_cbar, cax=cax, **colorbar_kwargs)
            cbar.ax.tick_params(labelsize=8)

        group_axes_map[g] = axes_this_group

    if fig_xlabel:
        fig.supxlabel(fig_xlabel)
    if fig_ylabel:
        fig.supylabel(fig_ylabel)

    fig.tight_layout()
    return fig, group_axes_map





# ---------- simulation helpers ----------
def ar1(n, phi=0.9, sigma=1.0, rng=None):
    rng = np.random.default_rng(rng)
    x = np.zeros(n)
    eps = rng.normal(0, sigma, size=n)
    for t in range(n):
        x[t] = (phi * x[t-1] if t > 0 else 0.0) + eps[t]
    return x

def arma11(n, phi=0.7, theta=0.4, sigma=1.0, rng=None):
    rng = np.random.default_rng(rng)
    e = rng.normal(0, sigma, n)
    x, e_lag = np.zeros(n), 0.0
    for t in range(n):
        x[t] = (phi * x[t-1] if t>0 else 0) + e[t] + theta * e_lag
        e_lag = e[t]
    return x

def markov_2state(n, p00=0.9, p11=0.9, rng=None):
    rng = np.random.default_rng(rng)
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        p = p00 if s[t-1]==0 else (1 - p11)
        s[t] = 0 if rng.uniform() < p else 1
    return s  # {0,1}

def smooth_noise(n, sigma=1.0, k=5, rng=None):
    # simple moving-average smoothing
    rng = np.random.default_rng(rng)
    z = rng.normal(0, sigma, n)
    kernel = np.ones(k)/k
    return np.convolve(z, kernel, mode='same')

def make_seasonal_amp(n, rng=None):
    rng = np.random.default_rng(rng)
    t = np.arange(n)
    amp = ar1(n, phi=0.8, sigma=0.5, rng=rng)
    return amp, np.sin(2*np.pi*t/12), np.cos(2*np.pi*t/12)

# Local horizon bases (front/mid/back) shaped like gentle bumps
def local_bases(H):
    h = np.arange(1, H+1)
    B_front = np.exp(-0.5*((h-3)/2.0)**2)
    B_mid   = np.exp(-0.5*((h-12)/3.5)**2)
    B_back  = np.exp(-0.5*((h-21)/3.0)**2)
    return B_front, B_mid, B_back

# ---------- main ----------
def simulate_future_curve_data(
    T=1500, H=24, start="2010-01-01", seed=0
):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=T)

    # Nelson–Siegel bases
    taus = np.arange(1, H+1)/12.0
    lam  = 0.8
    f1 = np.ones(H)
    f2 = (1 - np.exp(-lam*taus)) / (lam*taus)
    f3 = f2 - np.exp(-lam*taus)
    B_front, B_mid, B_back = local_bases(H)

    # ===== Drivers X =====
    X = {}

    # Linear / Lagged to NS
    X["x_LINp_lo_L"]    = ar1(T, phi=0.9, sigma=1.0, rng=rng)
    X["x_LINp_hi_L"]    = ar1(T, phi=0.9, sigma=1.0, rng=rng)
    X["x_LINn_mid_S"]   = arma11(T, phi=0.7, theta=0.5, sigma=1.0, rng=rng)
    X["x_LAGp_hi_L_L3"] = ar1(T, phi=0.85, sigma=1.0, rng=rng)
    X["x_LAGp_mid_S_L5"]= ar1(T, phi=0.85, sigma=1.0, rng=rng)

    # Nonlinear (monotonic / non-monotonic)
    base_mon            = ar1(T, phi=0.9, sigma=1.0, rng=rng)
    X["x_MONsat_mid_S"] = np.tanh(base_mon / 1.5)
    base_quad           = ar1(T, phi=0.92, sigma=1.0, rng=rng)
    X["x_NMONquad_C"]   = (base_quad**2 - np.var(base_quad))

    # Local horizon bumps (direct to Y)
    X["x_HUM_midonly"]  = smooth_noise(T, sigma=1.0, k=7, rng=rng)

    # Interaction (front)
    X["x_INT_lin_sat_front"] = X["x_LINp_hi_L"] * X["x_MONsat_mid_S"]

    # Regime and volatility
    s = markov_2state(T, p00=0.9, p11=0.9, rng=rng)             # {0,1}
    X["x_REG_thr_L"]   = s.astype(float)
    vol_base           = np.exp(ar1(T, phi=0.95, sigma=0.4, rng=rng))  # positive
    X["x_VOL_back"]    = vol_base

    # Seasonal amplitude
    amp, sin12, cos12  = make_seasonal_amp(T, rng=rng)
    X["x_SEAS_amp"]    = amp

    # Collinear + Null
    X["x_COL_lin"]     = X["x_LINp_hi_L"] + rng.normal(0, 0.05, T)
    X["x_NULL1"]       = ar1(T, phi=0.8, sigma=1.0, rng=rng)

    X_df = pd.DataFrame(X, index=idx)

    # ===== Factor dynamics (NS) =====
    phi_L, phi_S, phi_C = 0.97, 0.94, 0.92
    sL, sS, sC = 0.8, 0.6, 0.5
    L = np.zeros(T); S = np.zeros(T); C = np.zeros(T)

    # Coefficients (tune strengths)
    β = {
        "x_LINp_lo_L":  0.10,
        "x_LINp_hi_L":  0.35,
        "x_LAGp_hi_L_L3": 0.45,
        "x_LINn_mid_S": -0.30,
        "x_LAGp_mid_S_L5": 0.40,
        "x_MONsat_mid_S": 0.60,
        "x_NMONquad_C":  0.50,
        "regime_shift":  0.80,
    }

    for t in range(T):
        # lagged lookups
        L_lag3 = X_df["x_LAGp_hi_L_L3"].iloc[t-3] if t-3 >= 0 else 0.0
        S_lag5 = X_df["x_LAGp_mid_S_L5"].iloc[t-5] if t-5 >= 0 else 0.0

        # Level
        exo_L = (
            β["x_LINp_lo_L"]   * X_df["x_LINp_lo_L"].iloc[t] +
            β["x_LINp_hi_L"]   * X_df["x_LINp_hi_L"].iloc[t] +
            β["x_LAGp_hi_L_L3"]* L_lag3 +
            β["regime_shift"]  * X_df["x_REG_thr_L"].iloc[t]
        )
        L[t] = (phi_L * L[t-1] if t>0 else 0.0) + exo_L + np.random.normal(0, sL)

        # Slope
        exo_S = (
            β["x_LINn_mid_S"]  * X_df["x_LINn_mid_S"].iloc[t] +
            β["x_LAGp_mid_S_L5"]* S_lag5 +
            β["x_MONsat_mid_S"]* X_df["x_MONsat_mid_S"].iloc[t]
        )
        S[t] = (phi_S * S[t-1] if t>0 else 0.0) + exo_S + np.random.normal(0, sS)

        # Curvature
        exo_C = β["x_NMONquad_C"] * X_df["x_NMONquad_C"].iloc[t]
        C[t] = (phi_C * C[t-1] if t>0 else 0.0) + exo_C + np.random.normal(0, sC)

    # ===== Seasonal month-of-delivery (scaled by x_SEAS_amp) =====
    rng2 = np.random.default_rng(seed+1)
    base_month = rng2.normal(0.0, 1.0, size=12)

    # ===== Build Y =====
    Y = np.zeros((T, H))
    for t in range(T):
        ns_part = L[t]*f1 + S[t]*f2 + C[t]*f3

        # Local horizon effects (direct)
        local = (
            0.80 * X_df["x_HUM_midonly"].iloc[t]        * B_mid +
            0.60 * X_df["x_INT_lin_sat_front"].iloc[t]  * B_front
        )

        # Seasonality amplitude
        seasonal_amp = 1.0 + 0.5 * X_df["x_SEAS_amp"].iloc[t]

        for h in range(H):
            deliv_m = (idx[t] + pd.DateOffset(months=h+1)).month - 1
            seas = seasonal_amp * base_month[deliv_m]

            # Heteroskedastic noise: back end depends on x_VOL_back
            sigma_h = 0.6 + 0.03*h + 0.4 * (B_back[h]) * X_df["x_VOL_back"].iloc[t]
            noise = rng.normal(0, sigma_h)

            Y[t, h] = ns_part[h] + local[h] + seas + noise

    Y_df = pd.DataFrame(Y, index=idx, columns=[f"h{h}" for h in range(1, H+1)])

    # Optionally return ground-truth mapping for evaluation
    truth = {
        "to_L": ["x_LINp_lo_L", "x_LINp_hi_L", "x_LAGp_hi_L_L3", "x_REG_thr_L"],
        "to_S": ["x_LINn_mid_S", "x_LAGp_mid_S_L5", "x_MONsat_mid_S"],
        "to_C": ["x_NMONquad_C"],
        "direct_front": ["x_INT_lin_sat_front"],
        "direct_mid": ["x_HUM_midonly"],
        "variance_back": ["x_VOL_back"],
        "seasonal_amp": ["x_SEAS_amp"],
        "collinear": [("x_LINp_hi_L", "x_COL_lin")],
        "null": ["x_NULL1"],
        "lags": {"x_LAGp_hi_L_L3": 3, "x_LAGp_mid_S_L5": 5},
    }

    return X_df, Y_df, idx, truth
