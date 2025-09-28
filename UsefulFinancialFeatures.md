# Structural Breaks as Features

## Why structural breaks matter

* Regime shifts (e.g., mean-reversion → momentum) are where most market participants misprice risk; timely detection yields high risk/reward opportunities. 

---

## Test taxonomy (what to compute)

* **CUSUM family**

  * *Brown–Durbin–Evans (recursive residuals):* detect parameter drift in predictive regressions; sensitive to arbitrary start points.
  * *Chu–Stinchcombe–White (levels):* simpler, works on levels with null of no change; use rolling/backward-shifted windows and take the sup to avoid arbitrary reference levels. 
* **Explosiveness tests (bubble/burst detection)**

  * *Chow-type Dickey–Fuller (DFC/SDFC):* switch from random walk to explosive AR(1); unknown break date handled by supremum over admissible dates. One-break assumption is a limitation. 
  * *Supremum ADF (SADF / GSADF):* backwards-expanding windows at every end-point; no fixed number of regimes; spikes flag bubble-like phases. Prefer **log prices** over raw prices to avoid scale-induced heteroscedasticity. 
* **Sub/Super-martingale trend tests (model-agnostic)**

  * Fit trends to (y_t) or (\log y_t): polynomial, exponential, or power; test (|\beta|/\hat\sigma_\beta) over rolling windows. Penalize long windows with exponent (\varphi\in[0,1]) to balance long- vs short-run bubbles. 

---

## SADF in practice (what it costs & how to harden)

* **Algorithm:** For each time (t), run ADF on all backward starts (t_0) of length (\ge \tau), take the **sup** ( \max_{t_0} \text{ADF}_{t_0,t} ). Complexity ≈ **O(T²)** regressions; long histories can require **HPC/parallelization**. 
* **Robust variants:**

  * **QADF:** take a high quantile (e.g., 95th) of the ADF distribution over (t_0) instead of the max; optionally include an inter-quantile spread ( \dot Q ) to capture dispersion.
  * **CADF:** tail-conditional mean/variance above a high quantile to reduce outlier sensitivity. 

---

## Regime labeling logic (simple, fast post-processing)

* Using the zero-lag ADF form on **log prices**:

  * **Steady:** (\beta<0) (finite mean; half-life ( -\log 2 / \log(1+\beta))).
  * **Unit-root:** (\beta=0).
  * **Explosive:** (\beta>0) (bubble or crash depending on sign of level deviation). 

---

## Feature engineering (drop-in signals for your LLM pipeline)

* **Point signals:** `SADF_t`, `QADF_t(q)`, `CADF_t(q)`, `SMT_t` (with (\varphi) grid).
* **Derived signals:** time-over-threshold, area-over-threshold, slope/acceleration of `SADF`, time-since-last-break, count of breaks in lookback, regime state (steady/unit-root/explosive).
* **Cross-sectional:** co-break breadth (% tickers with `SADF` > crit), lead/lag between assets, cluster-level break indicators. 

---

## Thresholding & hyperparameters (what to tune)

* **Windows:** minimum sample length (\tau); lag order (L); trend spec (`'nc'`, `'ct'`, `'ctt'`).
* **Critical values:** e.g., CSW time-varying cutoff (c_\alpha[n,t]=\sqrt{b_\alpha}+\log(t-n)) (empirical (b_{0.05}\approx 4.6)); for ADF/SADF, prefer bootstrapped/MCS thresholds on your sampling scheme.
* **SMT weight:** (\varphi) (e.g., 0.5 vs 1.0) to target holding period. 

---

## Data & preprocessing (don’t skip)

* Use **log prices**; build continuous series correctly (e.g., futures “ETF trick”), and favor event-based bars (e.g., dollar bars) for stationarity. 

---

## Implementation checklist (minimum viable)

1. Compute **log-price** series; choose bar type.
2. For each asset and day: run **SADF** with (\tau,L,) trend spec; store `SADF_t`.
3. Compute **QADF/CADF** at (q=0.95) (+ spreads).
4. Compute **SMT** features on ({\text{poly, exp, power}}) with (\varphi\in{0.5,1.0}).
5. Label regimes via sign of (\beta) and mark **break events** when signals cross bootstrapped thresholds.
6. Log **complexity metrics** (wall-time, #regressions) and parallelize across assets/time. 

---

## Pitfalls to avoid

* Single-break assumptions (Chow/DFC) miss bubble–burst cycles.
* Using **raw prices** induces scale-driven heteroscedasticity.
* Edge-window bias (exclude early/late (\tau) region).
* SADF **outlier sensitivity** (prefer QADF/CADF for robustness). 

---

**Bottom line:** Detect regime shifts with SADF-style **explosiveness** metrics on **log prices**, harden with **QADF/CADF**, and complement with **SMT** trend tests. Expose both **event flags** and **continuous intensities** as features; tune (\tau, L, \varphi) and thresholds via bootstrap. 

# Entropy Features

## Why entropy (what it buys you)

* Entropy quantifies **information content / redundancy** in financial sequences; low entropy ⇒ more predictable (and often bubble-prone) structure, high entropy ⇒ closer to efficient, pattern-poor markets. Use it to gate strategies (e.g., momentum when info is sparse vs mean-reversion when info is dense). 

---

## Core definitions you’ll actually use

* **Shannon entropy** (H[X]=-\sum p(x)\log_2 p(x)); **redundancy** (R=1-\tfrac{H}{\log_2|A|}).
* **Mutual information** (MI(X,Y)=H(X)+H(Y)-H(X,Y)) (captures nonlinear association; reduces to (-\tfrac12\log(1-\rho^2)) for Gaussian).
* **Gaussian benchmark:** for IID (N(0,\sigma^2)), (H=\tfrac12\log(2\pi e\sigma^2)) → sanity-check your estimators and connect entropy to volatility. 

---

## Estimators (pick two)

* **Plug-in (ML) estimator:** discretize returns; estimate word probabilities of length (w); ( \hat H_{n,w}= -\frac1w\sum \hat p_w \log_2 \hat p_w). Simple, fast, biased if alphabet or (w) are mis-set. 
* **Lempel–Ziv / Kontoyiannis estimator:** measures **compressibility** via longest-match growth; robust to model misspecification; supports sliding or expanding windows; complexity ≈ linearithmic per point but many windows ⇒ heavy. Use modified form ( \tilde H) to avoid Doeblin condition; balance bias/variance with (k\approx(\log_2 n)^2). 

---

## Encoding schemes (critical for good signals)

* **Binary sign** (↑/↓): best with **event-based bars** (volume/trade/dollar) to stabilize (|r|).
* **Quantile encoding** (q-ary): uniform code usage in-sample; tends to **inflate** entropy.
* **Sigma/bucket encoding** (fixed width (\sigma)): non-uniform codes; lower average entropy but **spikes** when rare buckets hit.
* (Tip) Prefer **fractionally-differenced** inputs for stationarity before encoding. 

---

## Features to emit (drop-in for your LLM pipeline)

* **Point measures:** `H_plugin(w)`, `H_LZ(window)`, `Redundancy R`, **MI** with key drivers (e.g., volume, OI).
* **Windowed dynamics:** rolling `H` level/∆/z-score; **time/area over threshold**; **time since last entropy spike**.
* **Cross-sectional breadth:** % symbols with `H` above (or below) cutoffs; sector-/cluster-level entropy.
* **Portfolio concentration:** entropy of **PC risk shares** (\theta_i) (via eigen-risk); (H_\text{conc}=1-\frac1N e^{-\sum \theta_i\log \theta_i}).
* **Microstructure:** entropy of **order-flow imbalance** (quantized (v_B)) to proxy adverse selection risk alongside VPIN. 

---

## When to act (decision rules)

* **Mean-reversion on low (H):** redundancy ⇒ patterns; tighten stops (crowded).
* **Momentum on mid/high (H):** fewer exploitable patterns; size by entropy-implied uncertainty.
* **Hedging/concentration:** raise diversification when PC-entropy drops (risk on few factors). 

---

## Hyperparameters (defaults to start, then tune)

* **Alphabet size:** q = 2/4/8 (quantiles) or (\sigma = 0.5–1.0)×in-sample std of returns.
* **Word length (w):** 3–6; **window (n):** 200–500 (LZ: sliding or expanding).
* **Bias/variance:** for LZ sliding, set matches (k\approx(\log_2 n)^2) with (N\approx n+k).
* **Preprocessing:** event-based bars; fractional differencing; remove zeros for binary codes. 

---

## Caveats (don’t ignore)

* Entropy is **asymptotic**: short messages understate (H); consider **reversing** strings to use tail info; ensure **even length** in expanding windows.
* Encoding drives results; tiny alphabets **underestimate** (H).
* Use the **Gaussian benchmark** to validate estimator + encoding choices. 

---

## Minimal implementation plan (production-ready skeleton)

1. Build event-based bars → **encode** returns (binary / quantile / sigma).
2. Compute `H_plugin(w∈{3,4,5})` and `H_LZ(n∈{256,512}, sliding)`.
3. Standardize to rolling z-scores; derive **intensity** (area/time-over-threshold).
4. Cross-sectional **breadth** + **MI** to key drivers.
5. Portfolio **PC-entropy** for concentration risk monitoring.
6. Microstructure: quantize (v_B) per bar; LZ-entropy of OI sequence for **adverse selection** flag.
7. Validate on Gaussian draws (target (H=\tfrac12\log(2\pi e\sigma^2))); unit tests on estimator stability. 

**Bottom line:** Treat entropy as a **regime/complexity gauge**. Compute it reliably (plug-in + LZ), encode carefully, and wire it to (i) **strategy selection/sizing**, (ii) **portfolio concentration control**, and (iii) **microstructure risk flags**. 

# Microstructural Features

## Why it matters

* **Microstructure = rules + behaviors** of trading (order types, queues, cancellations). Rich **FIX/TAQ** data exposes how intentions are revealed/hidden — a goldmine for predictive features. 

## Core signal families to compute

* **Aggressor / order-flow sign (Tick Rule):** label each trade ±1 by price uptick/downtick; transform via runs tests, entropy, fractional diff, or Kalman forecasts of next sign. Use as the backbone for **signed volume / net order flow**. 
* **Effective spread & volatility (1st gen):**

  * **Roll(1984):** spread ≈ √(max{0, −Cov(Δp_t, Δp_{t−1})}); still useful when quotes are unreliable/illiquid.
  * **High–Low vol (Parkinson/Beckers):** robust σ from log(High/Low).
  * **Corwin–Schultz:** daily **spread** from highs/lows (and implied σ); no quote feed needed. 
* **Price impact / illiquidity (2nd gen):**

  * **Kyle’s λ:** regress Δp on **signed volume** (b_t·V_t).
  * **Amihud’s λ:** |Δlog close| per **$ volume** (daily).
  * **Hasbrouck’s λ:** TAQ-aware, Bayesian, with b·√(pV) term.
  * **Practical tip:** **use t-statistics** of these coefficients as features; they embed estimation uncertainty and are often more informative than raw levels. 
* **Information-driven trading (3rd gen):**

  * **PIN / VPIN:** estimate probability that trades are informed. Prefer **volume-synchronized bars**; **VPIN = mean(|buyVol − sellVol|)/barVolume** over a window. (Evidence mixed; works in some contexts, weak in others — don’t expect linear models to capture all signal.) 
* **Behavioral & market-ops features (from FIX/OB):**

  * **Round-size bias:** spikes at sizes 5/10/25/50/100… ⇒ more “mouse/GUI” (human) activity; deviations can flag trend vs chop regimes.
  * **Cancellation / order-type rates:** track **limit vs market** shares and **cancel rates**; spikes often precede illiquidity and can reveal **predatory algos** (stuffers, danglers, squeezers, pack hunters).
  * **TWAP fingerprints:** volume concentrated in the **first seconds of each minute** and at regional opens/closes; persistent early-minute imbalances hint at execution algos you can front-run.
  * **Options ↔ stock dislocations:** put-call-implied stock ranges vs actual quotes; option **trades** can carry info even when option **quotes** don’t.
  * **Signed order-flow autocorr:** positive and persistent (order splitting > herding). Use short-lag ACF of signed volumes. 

## Implementation guardrails

* **Sampling:** favor **activity clocks** (volume/dollar bars) over fixed time bars; 5-min is common but often inferior.
* **Targets:** when labeling (e.g., for MM profitability), keep **horizons short** and microstructure-consistent.
* **Smoothing:** consider **Kalman** for spread/vol curves; **robust stats** for highs/lows.
* **Data hygiene:** reconcile tick-rule signs with exchange aggressor tags (e.g., FIX 5797); inspect disagreement regimes (jumps, thin books, cancel waves). 

## A modern “information” feature you should add

* Train a classifier to predict **market-maker PnL** (profit=1 / loss=0) from your microstructure feature matrix **X**; compute **cross-entropy loss** out-of-sample; map −loss through a **KDE CDF** to get **ϕ_t ∈ (0,1)**. Rising ϕ_t ⇒ MM models failing ⇒ **higher adverse-selection risk**; use ϕ_t as a real-time “information pressure” feature (think flash-crash early warning). 

## What to feed your LLM training system (practical checklist)

* **Inputs:** trades (p, v, timestamp), quotes (best + depth), highs/lows, FIX events (replacements, cancels), options prints.
* **Derived:** tick-rule signs, signed volume, Roll/Corwin–Schultz spreads, Parkinson σ, Kyle/Amihud/Hasbrouck (coef **t-stats**), VPIN, cancel/market/limit shares, round-size frequencies, early-minute volume skews, signed-flow ACF, options-implied stock bands, **ϕ_t**.
* **Meta:** bar scheme (volume/dollar), window sizes, robust estimators, and any filters (e.g., outlier clipping). 
