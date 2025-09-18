### Interpreting L1: Lag–Horizon CCF (prewhitened)

#### 1. What the method measures
For each **lag** $\ell$ (e.g. 0 to 30 business days) and each **horizon** $h$ (1 to 24 months ahead), we compute

$$
\rho_{\ell,h} = \mathrm{Corr}(x_{t-\ell}^{pw},\, y_{t,h}^{pw})
$$

where $x^{pw}$ and $y^{pw}$ are *prewhitened* series (to reduce spurious correlation due to autocorrelation).

**Intuition:** does past movement of $x$ help explain variation in today’s forecast at horizon $h$?

---

#### 2. Interpreting the AR order
- **AR order (x):** how many autoregressive lags were removed from $x$ before correlation.  
  - $p=0$: no autocorrelation was modeled (raw series).  
  - Larger $p$: stronger serial dependence in $x$; the prewhitening filter removes that before comparing with $y$.

---

#### 3. Interpreting the heatmap
- Axes: **lag** (y-axis) vs **forecast horizon** (x-axis).  
- Color = correlation strength (positive/negative).

**Patterns:**
- **Vertical bands:** variable affects all horizons equally (a level effect on the curve).  
- **Horizontal bands:** a specific lag matters most across all horizons (consistent lead time).  
- **Diagonal patterns:** variable influences short horizons at short lags, and long horizons at longer lags.  
- **Local blobs:** variable relates to only certain horizons at specific lags.

---

#### 4. Interpreting the Max $|\rho|$ per horizon table
For each horizon $h$, the table reports:
- **lag\_of\_max\_abs:** the lag where absolute correlation is highest.  
- **rho\_at\_max\_abs:** the value of correlation at that lag.

Example:  
horizon = 6, lag = 8, rho = 0.284

means that 8 days earlier, changes in $x$ are most associated with the 6-month ahead forecast.

If the same lag appears across many horizons, it suggests a consistent lead time.

---

#### 5. Interpreting the Top cells table
- Shows the strongest individual $(\ell,h)$ correlations, sorted by magnitude.  
- Useful to identify:
  - **Short-term drivers:** small horizon, small lag.  
  - **Long-term drivers:** large horizon, consistent lag.  
  - Whether correlations are concentrated or diffuse.

---

#### 6. Practical interpretation examples
- “$x_1$ has peak correlation $\approx 0.29$ at lag 8 across horizons 1–12.”  
  → $x_1$ influences almost the whole curve with an 8-day lead.  
- “$x_2$ shows correlations only at horizons 18–24, lags 15–20.”  
  → $x_2$ is a long-term driver, affecting far-end forecasts with a 3–4 week delay.  
- “$x_3$ has alternating positive/negative bands diagonally.”  
  → $x_3$ induces shape changes (slope/curvature), not just level shifts.
