### Interpreting L2: Nelson–Siegel (NS) Curve Factors

#### 1) What this method does
We approximate each daily 24-month futures curve $y_t(\tau)$ (delivery in $\tau$ months/years) with **three loadings** (Level, Slope, Curvature):

$$
y_t(\tau)\;\approx\; \beta_{1,t}\cdot 1
\;+\; \beta_{2,t}\cdot \frac{1-e^{-\lambda \tau}}{\lambda \tau}
\;+\; \beta_{3,t}\cdot\!\Big(\frac{1-e^{-\lambda \tau}}{\lambda \tau}-e^{-\lambda \tau}\Big)
$$

- $\beta_{1,t}$ = **Level** (lifts the whole curve)  
- $\beta_{2,t}$ = **Slope** (tilts short vs long end)  
- $\beta_{3,t}$ = **Curvature** (belly vs ends)  
- $\lambda > 0$ controls *where* the curve bends.

For each date $t$ we solve least squares for $(\beta_{1,t},\beta_{2,t},\beta_{3,t}) \in \mathbb{R}^3$.

---

#### 2) How $\lambda$ is chosen
- If not given, $\lambda$ is selected by a **grid search** maximizing the **mean $R^2$** of the NS fit over time.
- Interpretation: a higher mean $R^2$ means the 3-factor NS basis explains more of your observed curve variance.

**Report items:**
- **$\lambda$**: the final bend parameter  
- **Per-date $R^2$ series** and its **mean/median**: goodness-of-fit over time

---

#### 3) What the outputs mean
**Factor scores (time series):**
- $\text{level\_t} = β_{1,t} \in \mathbb{R}$
- $\text{slope\_t} = β_{2,t} \in \mathbb{R}$
- $\text{curvature\_t} = β_{3,t} \in \mathbb{R}$

**Optional correlations with a driver $x$:**
For lags $\ell \in \{0,1,\dots\}$, we compute
$$
\rho_{\ell}^{(\text{factor})}=\mathrm{Corr}\!\big(x_{t-\ell},\,\beta_{\text{factor},t}\big),\;\; \text{factor} \in \{\text{level},\text{slope},\text{curvature}\}.
$$
The report shows a table across lags and the **best lag per factor**.

---

#### 4) How to read the plots/tables

##### A) Basis plot (three curves vs horizon)
- Shows the **shapes** that Level/Slope/Curvature impose across horizons.
- Use it to verbalize effects: *“A positive Slope raises the short end more than the long end.”*

##### B) $R^2$ over time
- High and stable $R^2$ ⇒ NS explains the term structure well.
- Drops in $R^2$ ⇒ structural change, missing seasonal component, or noisy quotes.

##### C) Factor series plot
- Compare dynamics:  
  - Level tracks common moves in **all** maturities.  
  - Slope flips sign when market swings from contango to backwardation.  
  - Curvature highlights **belly** deformations.

##### D) Correlation by lag (with $x$)
- A **row per lag**; columns are Level/Slope/Curvature.  
- Peaks indicate **lead time** and **which curve dimension** $x$ relates to.

---

#### 5) Typical narratives you can extract
- **“FX BRL/USD → Slope (lag ≈ 5 days):** short-dated contracts react more than long-dated ones.”  
- **“Spot cattle price → Level (lag ≈ 3–10 days):** broad parallel shifts of the curve.”  
- **“Feed cost (corn) → Curvature:** belly of the curve responds, ends move less.”

These are concise, horizon-agnostic conclusions that scale across many drivers.

---

#### 6) Practical thresholds and cautions
- Correlations $\lvert \rho \rvert \in [0.1,0.3]$ can already be meaningful in finance if stable and consistent in sign across adjacent lags. Use FDR if screening many $x$.
- Always inspect **sign stability** across nearby lags; isolated spikes may be noise.
- If $\lambda$ differs wildly across subperiods, consider **fixing $\lambda$** to a sensible value (e.g., based on a long sample) for more stable interpretation.
- NS captures *shape*, not seasonality. With strong yearly seasonality in delivery months, consider:
  - Working with *deseasonalized* curves, or  
  - Adding a **seasonal index** separately and interpreting factors on residual curves.

---

#### 7) Common pitfalls
- **Spurious relations** from shared trends: relate $x_{t-\ell}$ to factors on **detrended** data when appropriate.  
- **Over-interpreting Curvature**: ensure it is not just absorbing seasonal month-of-delivery effects.  
- **Sparse or missing maturities**: NS assumes reasonably complete 1–24M quotes; missing segments may bias factors unless handled consistently.

---

#### 8) When to switch/extend the method
- If the NS fit’s $R^2$ is mediocre or shapes look non-NS (e.g., multi-modal seasonality), try **Functional PCA (FPCA)** on curves and then correlate $x$ with FPCA scores (same interpretation pattern: factor loadings vs horizon).
- If you need strict no-arbitrage structure, consider Svensson or dynamic versions, but those are modeling choices rather than exploratory correlation tools.

---
