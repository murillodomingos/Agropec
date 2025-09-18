### Interpreting N2: MIC (Maximal Information Coefficient) — Lag–Horizon Grid

#### 1) What the method does
We screen for **non-linear (not-necessarily monotonic) associations** between a single driver $x$ and each point on the futures curve by scanning **lags** $\ell$ and **horizons** $h$:
- Pair $(x_{t-\ell},\, y_{t,h}) \in \mathbb{R}^2$ across time $t$.
- Compute **MIC**$(x_{t-\ell}, y_{t,h}) \in [0,1]$ for each $(\ell,h)$.
- Higher MIC $\Rightarrow$ stronger relationship *of any functional shape* (linear, exponential, periodic, thresholded, etc.).

MIC searches over many grid partitions of the scatter $(x,y)$ to approximate the **maximal normalized mutual information**. It is **model-free** and shape-agnostic.

---

#### 2) Key settings (reported in the header)
- **`alpha`** (default 0.6): controls the grid resolution search; larger values allow finer grids.
- **`c`** (default 15): caps grid size relative to sample size; balances sensitivity vs. overfitting.
- **`lags`**: the set $\{\ell\}$ of driver lags tested (e.g., 0..20).
- **`horizons`**: the set $\{h\}$ of curve maturities (e.g., 1..24 months).
- **Subsampling (`max_samples`)** (optional): limits sample size per $(\ell,h)$ pair to speed up.

---

#### 3) How to read the outputs

##### A) “Max MIC per horizon” table
For each horizon $h$, the table lists:
- **`lag_of_max`**: lag $\ell^\star$ where MIC is highest for that horizon.
- **`MIC_at_max`**: the MIC value at $(\ell^\star, h)$.

**Interpretation:**  
- If multiple horizons share the same $\ell^\star$, the driver has a **consistent lead time** across the curve.  
- If short horizons peak at small lags while long horizons peak at larger lags, there is a **lag–maturity gradient** (diagonal influence).

##### B) “Top cells by MIC” table
Shows the globally strongest $(\ell,h)$ pairs. Use this as a **shortlist** for follow-up plots or confirmatory tests (e.g., distance correlation or partial plots).

##### C) Heatmap (MIC over lag × horizon)
- **Vertical bands** (same lag, many horizons): a broad impact at a specific lead time (often a **level**-like effect).
- **Horizontal bands** (varies by horizon, same lag): driver targets a **specific segment** (short or long end).
- **Localized blobs**: **selective** influence on certain maturities at specific lags (e.g., belly-only effects).

---

#### 4) Practical narratives you can extract
- “**FX (BRL/USD)** shows MIC $\approx 0.35$ at $\ell=5$ across $h=1\ldots 6$”  
  → Short-end futures respond to FX moves with a one-week lead.
- “**Corn (feed cost)** has highest MIC for $h=12\ldots 24$ at $\ell=15$”  
  → Long-end expectations are non-linearly tied to past feed costs with ~3-week delay.
- “**Spot price** yields a vertical band at $\ell=2$ across most horizons”  
  → A common **level** driver with a 2-day lead.

---

#### 5) Thresholds and caveats
- **Magnitude:** MIC $\in [0,1]$; in noisy economic data, values **0.2–0.4** can be meaningful if pattern is **stable across adjacent lags/horizons**.
- **No p-values by default:** MIC is a scan statistic; if you need significance:
  - Use **permutation tests** (heavy) or
  - Treat MIC as a **ranking** and then confirm top pairs with secondary tests (e.g., distance correlation with permutations).
- **Multiple testing:** The grid $(\ell,h)$ is large. Apply **FDR** (Benjamini–Hochberg) across tested cells if converting to tests.
- **Sample size sensitivity:** MIC needs enough points per $(\ell,h)$ to be stable; beware of tiny effective samples after shifts and missing-data drops.
- **Scale/units:** MIC is rank/grid based and fairly scale-robust; still ensure reasonable preprocessing (e.g., outlier handling) to avoid artifacts.

---

#### 6) Recommended workflow with MIC
1. **Scan MIC** over $(\ell,h)$ for each driver.
2. **Aggregate** by horizon (max MIC and argmax-lag) and across horizons (e.g., mean MIC over a short-end set or long-end set).
3. **Shortlist** top drivers/horizons.
4. **Validate** with a second metric (distance correlation, HSIC) or visualize scatter for selected $(\ell,h)$.
5. **Summarize** findings in plain language (lead time + targeted maturities).

---

#### 7) When to use MIC vs. other tools
- Use **MIC** for **broad shape discovery** (unknown non-linear forms).
- Use **distance correlation (N1)** for a **global dependence** decision (and permutation $p$-values).
- Use **PLS (L3)** or **NS factors (L2)** to **explain curve structure** once a driver is flagged.

---
