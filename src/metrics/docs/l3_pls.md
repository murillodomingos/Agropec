### Interpreting L3: Partial Least Squares (PLS) — Window of X vs Full Future Curve Y

#### 1) What this method does
We relate a **lag window of drivers** to the **entire futures curve** on each date.

- Input on date $t$: a stacked window $\mathbf{X}_t \in \mathbb{R}^{w\cdot m}$ built from the last $w$ observations of $m$ driver series, and a curve vector $\mathbf{Y}_t \in \mathbb{R}^{H}$ (e.g., 24 horizons).
- **PLS** finds latent components $(\mathbf{T}, \mathbf{U})$ such that the covariance of scores is maximized:
  $$
  \max_{\mathbf{w},\,\mathbf{c}}\ \mathrm{Cov}\big(\mathbf{X}\mathbf{w},\,\mathbf{Y}\mathbf{c}\big)\,,
  $$
  iteratively extracting $K$ components and yielding a linear map $\widehat{\mathbf{B}}$ with
  $$
  \widehat{\mathbf{Y}} \approx \mathbf{X}\,\widehat{\mathbf{B}}\,.
  $$

**Intuition:** PLS finds a small number of **temporal driver patterns** (in the window) that best explain **joint movements across horizons** (curve shape/level).

---

#### 2) Key choices
- **Window size $w$:** how much recent history of each driver enters (e.g., 10 days). Larger $w$ increases feature count ($w\cdot m$).
- **Number of components $K$:** upper-bounded by $\min(w\cdot m, H)$. Start small (e.g., 1–3); increase if residual structure remains.
- **Scaling/Normalization:** by default here, **no automatic scaling**; apply z-scores upstream if drivers are on vastly different scales.

---

#### 3) Outputs to read
- **Coefficients** $\widehat{\mathbf{B}} \in \mathbb{R}^{(w\cdot m)\times H}$  
  Rows = windowed driver-features; columns = horizons. Entry $(i,h)$ quantifies linear effect of feature $i$ on horizon $h$.
- **Weights**:
  - $\mathbf{W}_X \in \mathbb{R}^{(w\cdot m)\times K}$: feature weights per component (which **time-lags and drivers** define each component).
  - $\mathbf{W}_Y \in \mathbb{R}^{H\times K}$: horizon weights per component (which **parts of the curve** each component targets).
- **Scores**:
  - $\mathbf{T}=\mathbf{X}\mathbf{W}_X \in \mathbb{R}^{T\times K}$ and $\mathbf{U}=\mathbf{Y}\mathbf{W}_Y \in \mathbb{R}^{T\times K}$.
  - Reported as time series $T_k$ and $U_k$.
- **Score correlation** $\mathrm{Corr}(T_k,U_k)$: how strongly component $k$ links X-window variation to Y-curve variation.
- **Per-horizon $R^2$** and **mean $R^2$**: in-sample explanatory power of the linear map for each horizon and on average.

---

#### 4) How to interpret each artifact

##### A) Per-horizon $R^2$
- High $R^2$ at **short horizons** ⇒ drivers explain near-end forecasts well (short-run influences).
- High $R^2$ at **long horizons** ⇒ drivers capture long-run level/slope movements.
- A **U-shape** across horizons can indicate belly/shape effects.

##### B) Score correlations $\mathrm{Corr}(T_k,U_k)$
- Values near 1 indicate component $k$ captures a **strong shared factor** between windowed drivers and the curve.
- If only the first component is strong, the relationship is effectively **low-rank** (single dominant driver pattern).

##### C) X-weights (top features per component)
- Sort $|\mathbf{W}_X[:,k]|$ to see **which driver-lags** define component $k$.
- Example: many large weights on `FX_t-1..t-5` ⇒ a **recent FX shock** component.

#### D) Y-weights (horizon profile per component)
- Inspect $\mathbf{W}_Y[:,k]$ to see which horizons move under component $k$.
- Flat positive profile ⇒ **level** effect; decreasing with horizon ⇒ **slope** (short > long); hump-shaped ⇒ **curvature/belly**.

##### E) Coefficient heatmap (features → horizons)
- Columns with uniformly signed coefficients across features ⇒ **parallel shifts** (level).
- Columns with sign flips across near vs far features ⇒ **tilts** (slope).
- Localized patterns around mid-horizons ⇒ **curvature** responses.

---

#### 5) Turning results into narratives
- **“Spot price window drives short end:”** high $R^2$ for $h\le3$, $\mathrm{Corr}(T_1,U_1)\approx 0.8$, X-weights concentrate on `spot_t..t-5`.  
- **“FX shock tilts curve (slope):”** Y-weights for comp 1 decrease with horizon; coefficients near-end >> far-end.  
- **“Feed cost affects belly (curvature):”** Y-weights hump at mid-horizons; coefficients strongest for $h\in[6,12]$.

---

#### 6) Stability and sanity checks
- **Component stability:** similar top X-features and Y-profiles across adjacent subsamples ⇒ robust factor.  
- **Sign stability:** consistent signs across neighboring lags/horizons reduce chance of overfitting.  
- **Scale sensitivity:** if one driver dominates due to units, standardize X before PLS.  
- **Collinearity:** PLS is robust relative to OLS, but extremely redundant windows may still overfit without cross-validation.

---

#### 7) Practical thresholds
- $\mathrm{Corr}(T_1,U_1)\gtrsim 0.6$ is a strong shared mode; components beyond the first often show diminishing returns.  
- Per-horizon $R^2$ in $[0.05,0.20]$ can be meaningful in finance if **stable** and **interpretable** (short data-generating processes).

---

#### 8) When to extend
- If coefficients clearly exhibit **level/slope/curvature** structure, compare with **Nelson–Siegel** factors for parsimony.  
- If non-linear effects suspected, combine with **distance correlation / HSIC** on $(\mathbf{X}_t,\mathbf{Y}_t)$ for dependence beyond linear maps.  
- For many drivers, consider **regularized PLS** or reduce window size to maintain interpretability and speed.

---
