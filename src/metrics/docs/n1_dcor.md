### Interpreting N1: Distance Correlation (dCor) — Windowed Drivers vs Future Curve

#### 1) What the method does
We test for **any statistical dependence** (linear or non-linear) between:
- A **windowed vector of drivers** $\mathbf{X}_t \in \mathbb{R}^{w \cdot m}$, built from $w$ most recent values of $m$ explanatory series.
- The **futures curve vector** $\mathbf{Y}_t \in \mathbb{R}^H$, containing $H$ horizons.

The **distance correlation** statistic is:

$$
\mathsf{dCor}(X,Y) = \frac{\mathsf{dCov}(X,Y)}{\sqrt{\mathsf{dVar}(X)\,\mathsf{dVar}(Y)}}
$$

where $\mathsf{dCov}$ and $\mathsf{dVar}$ come from double-centered pairwise distance matrices.

**Key property:**  
$\mathsf{dCor}(X,Y)=0 \iff X$ and $Y$ are statistically independent (in the population).

---

#### 2) How the test is run
- **Global test:** $\mathsf{dCor}(\mathbf{X}_t, \mathbf{Y}_t)$ across all horizons together.  
- **Per-horizon test:** $\mathsf{dCor}(\mathbf{X}_t, y_{t,h})$ for each horizon $h=1,\dots,H$.  
- **Permutation p-values:**  
  Shuffle $\mathbf{Y}_t$ relative to $\mathbf{X}_t$ and recompute statistic.  
  $p=\tfrac{1+\#\{s: \mathsf{dCor}^{(s)} \ge \mathsf{dCor}^{obs}\}}{1+n_{\text{perm}}}$

---

#### 3) How to read the outputs
- **Global dCor:** one number summarizing if the driver window and the *whole curve* are dependent.  
  - If significant, the driver explains some structure in the curve (could be level, slope, non-linear).
- **Per-horizon dCor:** profile across $h=1..H$ shows which maturities are most affected.  
  - A flat high profile ⇒ broad level effect.  
  - Strong only at near horizons ⇒ short-term link.  
  - Strong only at long horizons ⇒ long-run association.
- **Permutation p-values:** guard against spurious correlations.  
  - $p < 0.05$ (or after FDR correction across many horizons) ⇒ statistically robust.

---

#### 4) Interpreting the report tables
- **Per-horizon distance correlations (top):**  
  Sorted by magnitude; shows the horizons where dependence is strongest.
- **Bar chart of dCor vs horizon:**  
  Quick visualization of dependence structure across the curve.

---

#### 5) Practical interpretation examples
- **“FX driver (window 10 days) has global dCor=0.42 (p<0.01)”**  
  → FX dynamics are strongly dependent on the entire futures curve.
- **“Per-horizon profile: strongest dCor at $h=1..3$ months”**  
  → FX mainly explains short-end futures.
- **“Feed cost driver (corn) shows dCor only for $h=12..24$ months”**  
  → Corn prices affect long-term expectations of cattle prices.

---

#### 6) Thresholds and caveats
- **Effect size:** dCor $\approx 0.1-0.2$ may already be meaningful in economics, if stable across time.  
- **Permutation stability:** use enough permutations (e.g. 199/499) for reliable $p$-values.  
- **Multiple testing:** if many horizons and drivers are tested, apply FDR correction.  
- **Sample size:** dCor involves $O(n^2)$ pairwise distances; use subsampling for very large $T$.

---

#### 7) When to use
- When you suspect **non-linear associations** between drivers and curves.  
- As a **robust screening tool**: variables with significant dCor are strong candidates for deeper modeling (e.g. NS factors, PLS).

---
