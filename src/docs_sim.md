## 1) Naming taxonomy for drivers 


Family:
- LIN (linear)
- LAG (lagged linear)
- MON (monotonic nonlinear)
- NMON (non-monotonic)
- INT (interaction)
- REG (regime/threshold)
- VOL (volatility/heteroskedastic)
- SEAS (seasonal)
- COL (collinear)
- NULL (no effect)

Sign/shape: 
- p (positive)
- n (negative)
- sat (saturating)
- quad (quadratic)
- hump (local/mid-curve)

Target (optional): _L, _S, _C (Nelson–Siegel factors), or _front/_mid/_back (horizon region).

Strength: 
- lo
- mid
- hi

Lag: _Lk meaning effect realized at lag k

Examples

- x_LINp_lo_L → weak +linear effect on Level.
- x_LAGp_hi_S_L3 → strong +linear effect on S, realized at lag 3.
- x_MONsat_mid_S → monotonic saturating driver for S.
- x_NMONquad_C → quadratic (U-shape) effect on Curvature.
- x_HUM_midonly → local mid-maturity bump (direct on Y, not via L/S/C).
- x_INT_lin_sat_front → interaction impacting short horizons.
- x_REG_thr_L → regime/threshold shifting Level.
- x_VOL_back → raises back-end noise variance.
- x_SEAS_amp → scales seasonal amplitude.
- x_COL_lin → near duplicate of a linear driver.
- x_NULL1 → no effect (control).