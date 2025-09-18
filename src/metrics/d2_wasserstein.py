# ---------- D2: Sliced Wasserstein Distance (High vs Low regimes of x) ----------

@dataclass
class D2SWResult:
    """
    D2 | Sliced Wasserstein distance between future curves under High vs Low x.
    We compare empirical curve distributions: SW_p (default p=2 via random 1D projections).
    Also report 1D Wasserstein distances per horizon for interpretability.
    Decision: High–Low split with quantile q; sliced version for scalability in ℝ^H.
    """
    name: str
    q: float
    thr_low: float
    thr_high: float
    n_high: int
    n_low: int
    horizons: List[int]
    sw_full: float
    n_projections: int
    w1_by_h: np.ndarray

def d2_sliced_wasserstein_highlow(
    x: pd.Series,
    Y: pd.DataFrame,
    q: float = 0.7,
    n_projections: int = 128,
    max_rows: Optional[int] = None,
    random_state: Optional[int] = 0,
    name: Optional[str] = None
) -> D2SWResult:
    """
    Compute sliced Wasserstein distance between Y-curves for High vs Low x; plus 1D W1 per horizon.
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
    sw = float(sliced_wasserstein_distance(A, B, n_projections=n_projections, seed=random_state))
    H = Y2.shape[1]
    w1 = np.zeros(H)
    for j in range(H):
        w1[j] = wasserstein_distance(A[:, j], B[:, j])
    return D2SWResult(
        name=name or (x.name if x.name else "SW"),
        q=q,
        thr_low=thr_lo,
        thr_high=thr_hi,
        n_high=len(A),
        n_low=len(B),
        horizons=list(range(1, H+1)),
        sw_full=sw,
        n_projections=n_projections,
        w1_by_h=w1
    )

def d2_sliced_wasserstein_report(res: D2SWResult, top_k: int = 12) -> None:
    """
    Markdown summary and visuals for D2 results.
    """
    print(f"# D2: Sliced Wasserstein (High–Low) — {res.name}")
    print(f"- q: {res.q:.2f} | thresholds: [{res.thr_low:.3f}, {res.thr_high:.3f}] | sizes: high={res.n_high}, low={res.n_low}")
    print(f"- Global sliced-W distance (n_proj={res.n_projections}): {res.sw_full:.4f}")
    df = pd.DataFrame({"horizon": res.horizons, "W1": res.w1_by_h})
    print("\n## Per-horizon 1D Wasserstein (top)")
    print(df.sort_values("W1", ascending=False).head(top_k).to_markdown(index=False))
    plt.figure(figsize=(8,4))
    plt.bar(df["horizon"], df["W1"])
    plt.xlabel("horizon (months)"); plt.ylabel("W1 distance")
    plt.title("Per-horizon Wasserstein distance (High vs Low)"); plt.tight_layout(); plt.show()


if __name__ == "__main__":

    x = X.iloc[:,0]
    res_d1 = d1_energy_highlow(x=x, Y=Y, q=0.7, n_perm=199, max_rows=2000, random_state=0, name=f"{x.name}_EnergyHL")
    d1_energy_report(res_d1, top_k=12)

    res_d2 = d2_sliced_wasserstein_highlow(x=x, Y=Y, q=0.7, n_projections=128, max_rows=2000, random_state=0, name=f"{x.name}_SW")
    d2_sliced_wasserstein_report(res_d2, top_k=12)