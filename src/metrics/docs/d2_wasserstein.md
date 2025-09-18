#### D2) Sliced Wasserstein Distance (High–Low)

##### What it measures
- **Sliced Wasserstein** approximates Wasserstein distance in $\mathbb{R}^H$ by averaging **1D Wasserstein** distances over random projections:
  - Project curves $\mathbf{y}$ onto many random directions $u$ (unit vectors),
  - Compute 1D $W_p$ between projected samples for High vs Low,
  - Average over projections.

**Interpretation:** a **geometric** notion of how much **mass must move** to morph one curve distribution into the other.

##### How to read the outputs
- **Global sliced-$W$:** one number summarizing **multivariate** shift of curve distributions.
- **Per-horizon $W_1$ (1D):** interpretable in **price units**; larger $W_1$ means larger typical value shift for horizon $h$.
  - A flat high profile ⇒ widespread level shift.
  - A profile increasing with $h$ ⇒ stronger **long-end** shift.
  - Hump-shaped ⇒ **belly** deformation.

##### Practical narratives
- “**Rainfall Low** (drought) increases **long-end** futures (per-horizon $W_1$ grows with $h$); sliced-$W$ is large.”
- “**Spot High** mainly lifts **near end** (largest $W_1$ for $h \le 3$).”

##### Caveats
- No $p$-values are reported by default here; if needed, use permutations like in D1.
- The number of projections trades off stability vs speed (defaults are typically fine).

---

#### Choosing between D1 and D2
- **D1 (Energy):** nonparametric, **test-friendly** with straightforward label permutations; good general detector of distribution shifts.
- **D2 (Sliced Wasserstein):** **geometry-aware** and unit-interpretability per horizon via $W_1$; excellent for visual/narrative summaries of where mass moves.

---

#### Common pitfalls and tips
- **Multiple testing:** if screening many drivers/horizons, apply FDR to per-horizon $p$-values (D1).
- **Scaling:** distances are sensitive to units; compare horizons on consistent scales (e.g., all in prices or logs).
- **Seasonality:** strong delivery-month effects may dominate differences; consider conditioning or residualizing.
- **Sample selection:** High/Low thresholds (e.g., 30/30%) balance **contrast** vs **sample size**; avoid extreme splits that leave too few dates.

---

#### When to use these methods
- To detect **regime-driven distribution shifts** (beyond mean effects).
- To complement correlation-based screens with **shape-level evidence** (“the curve behaves differently when X is high”).
- To prioritize drivers that cause **broad** or **targeted** distributional changes in the futures curve.

---
