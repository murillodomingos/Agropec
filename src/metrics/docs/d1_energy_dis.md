### Interpreting D1–D2: Distributional Differences Under Driver Regimes (High vs Low)

We ask whether the **entire distribution of the futures curve** (or of each horizon) **changes** when a driver $x$ is **High** vs **Low**. This captures effects beyond mean correlation (variance, skew, shape).

---

#### 0) Setup (shared)
- Split dates into **High** and **Low** by quantiles of $x_t$:  
  High: $x_t \ge Q_q$, Low: $x_t \le Q_{1-q}$ (default $q=0.7$ ⇒ top/bottom 30%).
- For each date $t$, the curve is a vector $\mathbf{y}_t \in \mathbb{R}^H$ (e.g., $H=24$ months).
- We compute:
  - a **global, multivariate** distance between $\{\mathbf{y}_t\}_{t\in \text{High}}$ and $\{\mathbf{y}_t\}_{t\in \text{Low}}$,
  - and **per-horizon** distances between $\{y_{t,h}\}$ in High vs Low, for each $h=1,\dots,H$.

---

#### D1) Energy Distance (High–Low)

##### What it measures
- **Energy distance** between distributions $F$ (High) and $G$ (Low):

$$
D_E^2(F,G)\;=\; 2\,\mathbb{E}\|X-Y\| \;-\; \mathbb{E}\|X-X'\|\;-\;\mathbb{E}\|Y-Y'\|,\quad X,X'\!\sim F,\;Y,Y'\!\sim G.
$$

- In our report:
  - **Global $D_E$ (multivariate):** $X=\mathbf{y}_t \in \mathbb{R}^H$,
  - **Per-horizon $D_E$ (1D):** $X=y_{t,h} \in \mathbb{R}$.

**Interpretation:** $D_E$ increases when High and Low distributions are **further apart** (location, scale, or shape differences).

##### Significance
- **Permutation $p$-values** are computed by re-labeling High/Low and recomputing the statistic.
  - Small $p$ ⇒ the observed separation is unlikely by chance.

##### How to read the outputs
- **Header:** quantiles used, thresholds, and group sizes (balance matters).
- **Global curve $D_E$ + $p$-value:** “Is the **entire curve distribution** different when $x$ is High vs Low?”
- **Per-horizon table:** horizons with largest $D_E$ pinpoint where the curve differs most.
  - Large $D_E$ at **short horizons only** ⇒ short-end regime shift.
  - Broadly large $D_E$ ⇒ **level**-like shifts across the curve.
  - Peaks in mid-horizons ⇒ **curvature/belly** differences.

##### Practical narratives
- “**Corn High** shifts the **whole curve** upward (global $D_E$ significant); strongest gaps at $h=9$–$12$.”
- “**FX High** affects **near maturities** (1–3M) with little change beyond 12M.”

##### Caveats
- Use enough samples per group; extremely unbalanced splits weaken power.
- Consider seasonality: if season binds $y$ strongly, stratify or deseasonalize before comparing.

---
