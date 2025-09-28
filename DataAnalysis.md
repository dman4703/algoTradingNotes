# Financial Data Structures

## 1) Data sources (and the traps you must avoid)

* **Four primary types:** Fundamental, Market, Analytics, Alternative.

  * **Fundamental:** Low-freq, lagged, often **backfilled/reinstated**. Always time-align by **release time**, not reporting period or vendor index date.
  * **Market:** Raw exchange/venue feeds (e.g., FIX). Rich but messy; **order flow footprints** matter.
  * **Analytics:** Pre-processed signals (ratings, sentiment). Useful but **opaque, costly, non-exclusive**.
  * **Alternative:** Primary, early signals (sensors, web, transactions). **Expensive, privacy-sensitive**, and technically hard—thus potentially unique alpha.
* **Golden rule:** Prefer **raw, timestamped, versioned** data; never leak information earlier than it became public.

## 2) “Bars” — how to regularize irregular markets

* **Avoid time bars**: They oversample quiet periods and undersample busy ones; induce serial correlation/heteroscedasticity.
* **Prefer activity-based bars**:

  * **Tick bars:** Sample every N trades (watch for auction outliers).
  * **Volume bars:** Sample every V units (more robust to order fragmentation).
  * **Dollar bars:** Sample every $ notional (more stable across price levels/corporate actions; often the most stationary choice).
* **Information-driven bars (sample when informed flow arrives):**

  * **Tick Imbalance Bars (TIB):** Trigger when **signed tick imbalance** exceeds EWMA-based expectation.
  * **Volume/Dollar Imbalance Bars (VIB/DIB):** Same idea using **bt·vt** (volume or notional).
  * **Tick Runs Bars (TRB):** Trigger on **runs** of same-side ticks beyond expectation.
  * **Volume/Dollar Runs Bars (VRB/DRB):** Runs measured in **volume/notional**.
  * **Implementation note:** Maintain **E0[T]** and side probabilities with **EWMAs**; emit a bar when realized imbalance/run exceeds expected magnitude.

## 3) Multi-product series without pain (“ETF trick”)

* **Goal:** Model any basket/spread/futures roll as a **single, cash-like total-return series Kt** so your ML code always “trades an ETF.”
* **Mechanics:**

  * Compute **holdings hi,t** from desired **weights ωt**, prices, and contract point values.
  * Update **Kt** by reinvesting **PnL and carry**; handle **rebalance costs**, **bid-ask costs**, and **capacity** (basket volume = min constituent capacity).
  * Result: strictly positive, roll- and dividend-aware series suitable for backtests and ML labels.

## 4) Rolling a single future correctly

* Build a **cumulative roll-gap series** (difference between pre-roll close and post-roll open) and **detract it** from raw prices to form rolled prices.
* Use **rolled prices for PnL/labels**, but **raw prices for sizing/capital**.
* If you need non-negative prices for models, compound **returns** derived from rolled price change over **prior raw price**.

## 5) Feature/event sampling (don’t train on everything)

* **Downsampling**: Linspace/uniform reduce size but may drop the most informative examples.
* **Event-based sampling** (recommended): Trigger training examples on **catalytic events** (breaks, microstructure signals, volatility spikes, macro releases).

  * **CUSUM filter:** Maintain cumulative deviation from a reference; **emit an event when |S| ≥ h**, then reset—naturally avoids threshold chattering.
* Outcome: A features matrix concentrated on **actionable regimes**, improving label quality and model efficiency.

## 6) Practical implementation checklist for your pipeline

* **Data hygiene**

  * Store **(value, first_release_ts, revision_ts, version_id)**; never use revised values at initial timestamps.
  * Normalize clocks; document **exchange calendars, auctions**, and **corporate actions**.
* **Sampling API**

  * `bar_sampler(mode={tick,volume,dollar,TIB,VIB,DIB,TRB,VRB,DRB}, params={...})` returning homogeneous bars plus meta (counts, activity, costs).
* **Event engine**

  * `cusum_events(y, h, ref='lag1' or EWMA)` → event indices for feature/label extraction.
* **Futures/baskets**

  * `roll_gaps(series)`, `roll_prices(...)`, `etfize(weights, prices, carry, costs)` → **Kt**, **ct** (rebalance cost), **ĉt** (spread-cross cost), **vt** (capacity).
* **Weights via PCA (optional)**

  * `pca_weights(cov, riskDist=None, riskTarget=σ)` to target risk across PCs (default: load the **lowest-variance PC**).
* **Diagnostics**

  * Stability of **bars/day**, **serial correlation of returns**, and **distributional tests** across bar types.
  * Leakage checks: ensure **feature timestamps < label timestamps** and respect **release times**.
  * Track **EWMA parameters**, thresholds, and **dataset lineage** for reproducibility.

## 7) Do / Don’t

* **Do:** Align by **release times**; use **dollar/imbalance/run bars**; sample **events**; model complex assets via **ETF trick**; include **transaction costs** and **capacity**.
* **Don’t:** Use time bars; use backfilled/revised fundamentals at original dates; ignore auctions, fragmentation, or corporate actions; train on every bar indiscriminately.

# Labeling

## Why labeling matters

* Supervised models need **labels `y`** aligned to feature rows `X[i]` at exact, market-consistent times. Labels must reflect **how a trade would really exit** (profit, loss, or expiry), not just an arbitrary future price snapshot. 

## Don’t use naive fixed-horizon labels

* The common “**fixed time horizon**” label (`sign(return over next h bars)` with a fixed threshold `τ`) fails because:

  * **Time bars are heteroscedastic** (volatility varies by session/liquidity).
  * A **single τ** ignores current volatility → mostly “0” labels or biased labels.
  * It **ignores the price path** (stop-outs en route).
  * Use **volatility-scaled thresholds** and/or **volume/dollar/information-driven bars** instead. 

## Volatility-aware thresholds (dynamic `τ`)

* Estimate rolling daily vol with **EWMA std** on returns (e.g., `getDailyVol(span=100)`) and scale profit-taking / stop-loss levels accordingly. Outcome: thresholds match regime risk. 

## The Triple-Barrier Method (path-dependent, realistic)

* For each event start time `t0`, define **three barriers**:

  1. **Upper (profit-taking)** = `+pt * trgt(t0)`
  2. **Lower (stop-loss)** = `-sl * trgt(t0)`
  3. **Vertical (expiry)** after `h` bars (or time)
* **Label = first barrier hit** within `[t0, t0+h]`:

  * Hit upper → `+1`
  * Hit lower → `-1`
  * Hit vertical → either `sign(return)` or `0` (design choice)
* Supports 8 configurations `[pt, sl, t1]`, with `[1,1,1]` the **standard** (take profit, stop loss, and max holding period). 

## Learning **side** and **size**

* If model must learn **direction and magnitude**:

  * Use **symmetric horizontal barriers** (pt = sl) so side is irrelevant to first-touch timing.
  * Produce labels via first-touch returns → `bin ∈ {−1,0,1}` (optionally set `0` when vertical hits first). 

## Meta-Labeling (stacked, production-friendly)

* When an exogenous/primary model already decides **side** (buy/sell), train a **secondary classifier** to decide **whether to act (size > 0) or pass**:

  * Input events carry `side ∈ {−1,+1}`; barriers need **not be symmetric**.
  * Output labels become **binary**: `bin ∈ {0,1}` (trade or skip); multiply realized returns by `side`.
  * Use predicted **P(trade=1)** for **position sizing**.
  * Benefits: **higher F1** (filter false positives), lower overfit risk, fits **quantamental** workflows (white-box primary + ML filter). 

## Minimal API (data contracts)

* **Inputs**:

  * `close: pd.Series` (prices, properly time-indexed/bfilled for gaps)
  * `tEvents: pd.Index` (event seeds from your sampling engine)
  * `trgt: pd.Series[float]` (volatility-scaled target widths at `t0`)
  * `ptSl: list[float,float]` (multipliers for profit/stop; symmetric if learning side)
  * `t1: pd.Series[Timestamp] | False` (vertical barrier per event; `False` = none)
  * `side: pd.Series[±1] | None` (None when learning side; provided for meta-labels)
* **Core functions** (conceptual):

  * `getDailyVol(close, span)` → rolling vol estimator (**for `trgt`**)
  * `addVerticalBarrier(tEvents, numDays)` → series `t1`
  * `applyPtSlOnT1(close, events, ptSl, molecule)` → first hit times per barrier
  * `getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None)` → dataframe with `t1` (first-touch) and `trgt` (and `side` if meta)
  * `getBins(events, close)` → dataframe with `ret`, `bin` (`{-1,0,1}` or `{0,1}` if meta)
* **Outputs** feed directly into your trainer (`X[event_idx] ↔ y['bin']`). 

## Class imbalance control

* **Drop under-represented labels** recursively until each class ≥ `minPct` (unless only two classes remain). This stabilizes training for algorithms sensitive to imbalance. 

## Performance & scaling

* Triple-barrier is **computationally heavy** (path scanning). Use **multiprocessing** over event “molecules” to parallelize the barrier-touch search. 

## Practical defaults (sane starting point)

* Bars: **dollar/volume/information-driven**, not time bars.
* Targets: `trgt = EWMA(daily returns std)`; symmetric `ptSl = [1,1]` when learning side; **asymmetric** for meta-labeling per risk/reward.
* Vertical barrier: **1 trading day** (or strategy-specific holding limit).
* Meta-labeling: **use when you already have a side model**; optimize **F1** (precision/recall trade-off).
* Label on **first touch**; ensure strict **timestamp ordering** (features < outcomes). 

# Sample Weights

## The core problem

* Financial labels are **not IID** because outcomes **overlap in time** (path-dependent exits, triple-barrier, vertical horizons). Treating them as IID **inflates accuracy** and **oversamples redundancy**. 

## Key constructs you must compute

* **Concurrency per bar**: (c_t=\sum_i \mathbf{1}*{t,i}) where (\mathbf{1}*{t,i}=1) iff event (i) spans ([t-1,t]).
* **Uniqueness at bar**: (u_{t,i}=\mathbf{1}_{t,i}/c_t).
* **Average uniqueness per event**: (\bar u_i=\frac{\sum_t u_{t,i}}{\sum_t \mathbf{1}_{t,i}}).

  * Use (\bar u_i) to **downweight overlapped** observations instead of dropping them.

## How to set sample weights (training-time)

1. **Return attribution (magnitude-aware)**

   * Weight each event by **absolute log-returns uniquely attributable** to it:
     ( \tilde w_i=\Big|\sum_{t=t_{i,0}}^{t_{i,1}} \frac{r_{t-1,t}}{c_t}\Big|,\quad w_i=\tilde w_i\cdot \frac{I}{\sum_j \tilde w_j}).
   * Rationale: big, unique moves carry more learning signal than tiny, crowded ones.
2. **Time decay (recency-aware)**

   * Apply **piecewise-linear decay** to the cumulative uniqueness order: parameter (c\in(-1,1]).

     * (c=1): no decay; (0<c<1): linear decay to (c); (c=0): decays to 0; (c<0): oldest fraction gets **zero** weight.
3. **Final weight** = (return attribution) × (time-decay factor).

> Implementation detail: Computing (\bar u_i) and return-attribution uses each event’s (t_1) (first-touch / expiry). That’s fine: they are **training-set-only** quantities and **do not leak** into test time.

## Resampling policy (bagging / bootstrap)

* **Don’t** naïvely bootstrap IID on overlapped data → you’ll sample near-duplicates; **OOB accuracy becomes unrealistically high**.
* **Better**:

  * Set `max_samples ≈ mean( tW )` (mean uniqueness) for bagging-style ensembles.
  * Use **Sequential Bootstrap**: iteratively draw samples with probabilities **proportional to the resulting average uniqueness**; reduces redundancy vs. standard bootstrap (higher median (\bar u) in Monte Carlo).

## Class weighting (imbalance control)

* For classification with labels ({-1,1}) (and neutral implied by low confidence), set `class_weight='balanced'` (or `'balanced_subsample'` for bagging) to avoid the model ignoring rare-but-important classes/events.

## What **not** to do

* **Don’t** force non-overlap by stretching horizons or dropping partial overlaps → you lose too much information.
* **Don’t** rely on time bars or fixed horizons alone; keep the path-dependence and overlapping-event reality in your label design and weighting.

## Minimal “API” you should expose in your training pipeline

* **Concurrency & uniqueness**

  * `numCoEvents = count_concurrency(close_index, t1)`
  * `tW = average_uniqueness(t1, numCoEvents)`  # per-event (\bar u_i)
* **Resampling**

  * `phi = seq_bootstrap(indicator_matrix(bar_index, t1), s_length)`  # indices for bagging
* **Weights**

  * `w_ret = return_attribution_weights(t1, numCoEvents, close)`
  * `w_decay = time_decay(tW, c)`
  * `w_final = normalize(w_ret * w_decay)`
* **Trainer hooks**

  * Pass `w_final` to `fit(X, y, sample_weight=w_final)`; set `class_weight` appropriately; adjust bagging `max_samples≈mean(tW)`.

## Sanity checks before you train

* **Leakage**: features strictly **precede** labels; training-only use of (t_1), (\bar u), and weights.
* **Redundancy**: monitor mean (\bar u) (higher is better); compare **OOB vs. CV** (CV without shuffling should be lower/realistic).
* **Stability**: ablation on time-decay (c) and resampling method; confirm material impact on out-of-sample AUC/F1/IC, not just on OOB.

# Fractionally Differentiated Features

* **Problem:** ML needs **stationary** features, but naïve differencing (returns, `d=1`) **erases memory**, i.e., the predictive structure in prices. Goal: make features stationary while **preserving maximum memory**. 

* **Core idea (fractional differencing):** Apply ((1-B)^d) with real (d\in\mathbb{R}*+) (backshift (B)), producing a weighted dot-product ( \tilde X_t=\sum*{k\ge0}\omega_k X_{t-k} ) with binomial-series weights.

  * Iterative weights: ( \omega_0=1,\quad \omega_k=-\omega_{k-1}\frac{d-k+1}{k}); (\omega_k\to0) as (k\uparrow\infty).
  * **Interpretation:** `d=0` keeps the original (non-stationary, high memory); `d=1` (returns) is stationary but **memory-less**; **fractional** `0<d<1` trades off both.

* **Implementations (pick for production):**

  1. **Expanding-window fracdiff** with **weight-loss threshold** `τ` (drops early samples with insufficient cumulative weight) — works but induces **negative drift** (weights change over time).
  2. **Fixed-Width Window FFD (recommended):** truncate the kernel where (|\omega_k|<\tau); use a **constant weight vector** for all timestamps → **stationary, driftless** series that still carries memory.

* **How to choose the amount of differencing (`d*`):**

  * Sweep (d \in [0,1]) (extend if needed); for each, compute FFD(d) and run **ADF**; pick the **minimum (d^*)** with **p-value < 5%**.
  * Empirically, many liquid futures reach stationarity with **(d^*<0.6)**; e.g., for E-mini S&P, **(d\approx0.35)** passes ADF while retaining **very high correlation (~0.995)** to the original level series; returns (`d=1`) drop correlation to near zero (~0.03).

* **Pipeline contract (what your trainer expects):**

  * **Inputs:** `series` (index-aligned, ffilled), `d_grid`, `tau` (e.g., `1e-5`).
  * **Procedure:**

    1. (Optional) **Cumsum** a feature first to ensure some integration order exists.
    2. For each feature, compute **FFD(d)** over `d_grid`, run **ADF**, choose **d***.
    3. Produce **FFD(d*)** using **fixed-width** weights; record **(d*, τ, window width, ADF stat/p-value, corr(original, FFD))**.
  * **Outputs:** Stationary feature columns preserving long-memory structure; ready for event-based sampling, triple-barrier labeling, and overlap-aware sample-weighting.

* **Minimal API (pseudocode names):**

  * `getWeights(d, size)` → fractional kernel; `fracDiff(series, d, tau)` (expanding);
  * `getWeights_FFD(d, tau)` + `fracDiff_FFD(series, d, tau)` (fixed-width, **use this**);
  * `min_d_adf(series, d_grid, tau, adf_alpha=0.05)` → selects (d^*) and metrics.

* **Operational guidance:**

  * **Do:** use **FFD (fixed-width)**; **pick minimal d*** that passes ADF; log kernel/params; verify corr vs. original to confirm **memory retention**; expect non-Gaussian residuals (skew/kurtosis).
  * **Don’t:** default to returns (`d=1`); ship expanding-window outputs in prod; rely on cointegration to “save” non-stationary features; assume one `d` fits all features.

* **One-liner for your LLM:** *“Transform each non-stationary feature with **FFD at the minimal (d^*)** that passes ADF, using **fixed-width kernels**; this yields **stationary, memory-preserving** inputs for labeling and training.”* 