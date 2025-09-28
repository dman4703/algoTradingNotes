# Bet Sizing

* **Why it matters:** High forecast accuracy ≠ profits. **Position size drives P&L.** Two correct signals can earn or lose depending on how you ramp size. Reserve capacity so you can **add** if the signal strengthens, not just cut if it weakens. 

## Strategy-agnostic sizing recipes

* **Concurrency CDF (two-Gaussian mix):** Count concurrent longs/shorts over time (`c_t = c_{t,l} − c_{t,s}`), fit a **mixture of 2 Gaussians** to {c_t}, map to size
  ( m_t=\begin{cases}\dfrac{F[c_t]-F[0]}{1-F[0]},& c_t\ge0[4pt]\dfrac{F[c_t]-F[0]}{F[0]},& c_t<0\end{cases} )
  Intuition: stronger/rarer concurrency ⇒ larger size. 
* **Budgeting by maxima:** Track maxima of concurrent longs/shorts, size as
  ( m_t=\dfrac{c_{t,l}}{\max c_{i,l}}-\dfrac{c_{t,s}}{\max c_{i,s}} ). Ensures you don’t “hit max” before later signals arrive. 
* **Meta-labeling probability sizing:** Train a separate classifier to predict **misclassification risk**; convert predicted class probabilities → continuous size (decouples primary model from sizing). 

## From predicted probabilities → size

* **Binary:** Test (H_0:p(x{=}1)=\tfrac12). With (p=p(x{=}1)), z-score
  ( z=\dfrac{p-\tfrac12}{\sqrt{p(1-p)}} ), then **size** ( m=2\Phi(z)-1 \in[-1,1] ). Direction = predicted class. 
* **Multiclass (OvR):** Let ( \tilde p=\max_i p_i ), test (H_0:\tilde p=1/|X|), compute ( z=\dfrac{\tilde p-1/|X|}{\sqrt{\tilde p(1-\tilde p)}} ), set
  ( m = x,(2\Phi(z)-1) ) where (x\in{-1,0,1,\dots}) encodes side/label. 

## Overlapping bets → average, then discretize

* **Average while active:** Each bet has a holding interval `[t0, t1]` (triple-barrier). At any time `t`, **average all active** bet sizes to reduce churn. 
* **Discretize to cut jitter:** ( m^*=\operatorname{round}(m/d),d ) with step `d∈(0,1]`; cap to `[−1,1]`. Choose `d` to balance turnover vs reactivity. 

## Dynamic target size & execution (with limit price guardrails)

* **Sigmoid ramp from price-forecast divergence:** With forecast (f_i), price (p_t), max position (Q):

  * Bet-size function: ( m(\omega,x)=\dfrac{x}{\sqrt{\omega+x^2}},; x=f_i-p_t )
  * **Target shares:** ( \hat q_{i,t}=\mathrm{int}(m(\omega,f_i-p_t),Q) )
  * **Calibrate curvature:** given `(x, m*)`, ( \omega=x^2,(m^{*-2}-1) )
  * **Loss-avoiding limit price** for order ( \hat q_{i,t}-q_t ) via inverse
    ( L(f,\omega,m)=f-m\sqrt{\dfrac{\omega}{1-m^2}} ) (monotone ⇒ avoid realizing losses as (p_t\to f)). 
* **Alternative ramp:** Power law ( \tilde m(\omega,x)=\operatorname{sgn}(x),|x|^\omega ) on normalized (x\in[-1,1]). Easier curvature control; hits ±1 at ±1. 

## Engineering notes (plug-and-play)

* **Inputs:** predicted class probs, forecast side/price, `t1` (barrier-based horizons), concurrency counts, knobs `{Q, d, ω}`.
* **Pipeline order:** probs → per-bet size → **average active** → **discretize** → **map to target position** → compute **limit price** for safe execution.
* **Risk controls:** clip to `[−1,1]`; enforce gross/net limits, turnover caps, minimum trade notional; cross-validate **sizing policy**, not only labels.
* **Where it breaks:** going full-size too early; no capacity left for later stronger signals; skipping averaging/discretization (overtrading). 

**Source:** Chapter “Bet Sizing” (concurrency-based sizing, probability-to-size mapping, averaging/discretization, dynamic target & limit price with sigmoid/power alternatives). 

# Dangers of Backtesting

* **Backtest ≠ experiment.** It’s a *historical simulation*, not causal evidence and not a guarantee of future Sharpe/P&L. Use it as a **sanity check** (sizing, turnover, cost resilience, scenario behavior), not as proof. 

## Don’ts (the fastest ways to fool yourself)

* **Survivorship bias:** Using today’s universe for the past.
* **Look-ahead / leakage:** Any info not truly available at the decision timestamp (release lags, backfills, vendor corrections).
* **Storytelling after the fact:** Ex-post narratives justifying random patterns.
* **Data mining/snooping:** Tuning on the test set; training until the backtest “wins.”
* **Transaction-cost naivety:** Costs, impact, and slippage are state- and participation-dependent.
* **Outlier-driven edges:** Strategies hinging on a handful of extreme events.
* **Shorting realism:** Locate/borrow constraints, borrow fees, inventory limits.
  Plus: non-standard performance math, hidden risk, correlation≠causation, cherry-picked periods, ignoring stop-outs/margin/funding/practicalities. 

## Second Law of Backtesting (mindset)

* **“Do not research under the influence of a backtest.”** Specify labels, features, weighting, CV, ensembling, and bet sizing **ex-ante**. If the backtest fails, **restart from scratch**—don’t iteratively “fix” the model using the backtest. The more tests you run, the higher the chance of a **false discovery**. 

## What a *good* process looks like

* **Model scope:** Build edges at *asset-class / universe* level, not single names; single-name edges that don’t generalize are suspect.
* **Bagging:** Use to lower variance and curb overfit; if bagging kills performance, your “edge” likely rides a few idiosyncratic points.
* **Scenario simulation > single history:** Profitability should persist across many “what-if” paths, not just the one path that happened.
* **Audit every trial:** Log *all* runs; later **deflate Sharpe** / adjust selection for the number of tries. 

## Strategy selection without leakage: CSCV & PBO (use this)

* **Combinatorially Symmetric Cross-Validation (CSCV):**

  1. Collect P&L series for **N** model configs into matrix **M** (T×N), time-aligned.
  2. Split rows into **S** disjoint, sequential blocks; enumerate all half-/half train–test combinations.
  3. Per combo: pick the **IS best** config; rank its **OOS** performance within peers; convert to a **logit**.
  4. Aggregate logits to estimate the **Probability of Backtest Overfitting (PBO)** = Prob(logit < 0). Lower is better.
     This quantifies performance decay IS→OOS and the chance you selected a false winner. 

## Minimum engineering guardrails for your training system

* **Timestamps first-class:** Enforce event-time joins, release delays, vendor backfill flags; forbid future columns in feature windows.
* **True OOS validation:** Use time-aware folds; add randomization at the *block* level to avoid a single, targetable OOS path.
* **Pre-commit specs:** Freeze label definition, feature set, CV, and sizing *before* any backtest.
* **Costs & constraints:** Model TC/impact vs participation; cap turnover; encode borrow limits/fees for shorts.
* **Universe construction:** Point-in-time membership; delistings included.
* **Trial ledger:** Persist seeds/configs/metrics; compute deflated SR & PBO on the final pick. 

> **Purpose of a backtest:** *Discard* bad models and check logistics (sizing, costs, capacity). Don’t “improve” models with it. 

# Backtesting Through Cross-Validation

* **Purpose shift:** Don’t just “replay history.” Use backtesting to **infer future performance across multiple out-of-sample scenarios**, including stress regimes that didn’t occur sequentially in the past. 

## Walk-Forward (WF): what it is / why it breaks

* **Definition:** Trailing (past-only) training → next decision; simulates *one* historical path; no embargo needed if timestamps are clean.
* **Pros:** Clear historical interpretation; leakage-resistant if purging is correct.
* **Cons:** **Single scenario** (easy to overfit); results depend on **sequence order**; early decisions trained on **less data**; still prone to subtle leakage (label horizons overlapping). 

## Cross-Validation (CV): scenario testing, not history

* **Definition:** Partition time into blocks; train on all but test block(s); every observation is tested **exactly once**; all decisions trained on **equal-sized** datasets.
* **Pros:** Multiple scenario views; longest possible OOS use; uniform information per decision.
* **Cons:** Still **one path** overall; requires **purging + embargo** to prevent temporal leakage; no “paper-trading” interpretation. 

## CPCV — Combinatorial *Purged* Cross-Validation (use this)

* **Core idea:** Generate **many backtest paths** by enumerating **all** train/test splits of **N** ordered groups with **k** groups held out per split, while **purging** overlaps and **embargoing** near-boundary samples.
* **Paths:**

  * Number of paths: ( \varphi[N,k]=\frac{k}{N}\binom{N}{N-k} ) (e.g., with (k=2): ( \varphi=N-1 )).
  * **Training fraction per split:** ( \theta = 1-\frac{k}{N} ).
  * Practical recipe: pick target paths ( \varphi ) → set ( N=\varphi+1 ), (k=2) (good trade-off: many paths, large ( \theta )).
* **Algorithm (sketch):**

  1. Partition T timestamps into **N contiguous groups** (no shuffling).
  2. Enumerate splits with **k** test groups each.
  3. **Purge/embargo**: drop any train labels whose horizons overlap test horizons; embargo near edges.
  4. Fit on each train split; forecast on its test split.
  5. **Assemble (\varphi) independent OOS paths** and compute a **distribution** of metrics (Sharpe, drawdown, turnover, hit-rate). 

## Why CPCV reduces false discoveries

* WF/CV produce **one Sharpe** per strategy → high variance → pick-the-max bias.
* CPCV yields **many path Sharpes** ({y_{i,j}}_{j=1}^{\varphi}), with lower variance of the **mean**:
  ( \mathrm{Var}[\bar y_i] = \frac{1}{\varphi}\sigma_i^2\big(1+(\varphi-1)\bar\rho_i\big) ), where (\bar\rho_i) is average cross-path correlation. Lower (\bar\rho_i) ⇒ tighter estimates ⇒ **fewer false “edges.”** 

## Engineering guardrails (bake these into your training system)

* **Time-aware data API:** first-class timestamps, label horizons (t_1), release lags; forbid future info in feature windows.
* **Group partitioner:** deterministic, contiguous **N** groups; support (k=2) by default; configurable embargo and PurgedKFold-style overlap checks.
* **Metrics as distributions:** always report **mean ± std** across CPCV paths; compare strategies on **path-level** distributions, not single numbers.
* **Stress targeting:** choose test groups to **cover regimes** (crash, rally, chop); CPCV ensures uniform usage without warm-up bias.
* **Trial ledger:** log every config/run; when selecting “the winner,” disclose (\varphi,N,k), path metrics, and apply deflated/adjusted Sharpe. 

> **Bottom line:** Replace single-path WF/CV with **CPCV** to get **many OOS paths**, tighter uncertainty, and materially lower probability of backtest overfitting. 

# Backtesting on Synthetic Data

* **Goal.** Avoid backtest overfitting when tuning trading rules. Fit a simple **stochastic process** to history, **simulate many synthetic paths**, and **pick exit thresholds** (profit-take & stop-loss) that maximize out-of-sample Sharpe—**without** sweeping parameters on a single historical path. 

---

## Core concepts

* **Trading rule**: (R={\pi,\bar\pi}) with stop-loss (\pi<0) and profit-take (\bar\pi>0). Exit when the MtM P&L hits either threshold. Optimize (R) for **Sharpe** across opportunities.
* **Overfitting risk**: Choosing (R) by brute-force backtests often locks onto noise; serial dependence worsens it. Prefer **process-driven synthetic testing** to lower PBO (Probability of Backtest Overfitting).
* **Process choice**: Price (or P&L) follows **discrete Ornstein–Uhlenbeck (O-U)**:
  [
  P_t=(1-\phi),E[P_T]+\phi P_{t-1}+\sigma\varepsilon_t,\quad \varepsilon_t\sim\mathcal N(0,1),; |\phi|<1
  ]
  Half-life: (\tau=-\log 2/\log\phi) (or (\phi=2^{-1/\tau})).

---

## Practical algorithm (use this to generate training data & labels)

1. **Estimate process**: Linearize O-U and use OLS over concatenated opportunities to estimate (\hat\phi) and (\hat\sigma).
2. **Grid of rules**: Build a mesh of ((\pi,\bar\pi)) (e.g., both from (0.5\sigma) to (10\sigma)).
3. **Simulate**: For each opportunity, simulate many paths (e.g., 100k) from the fitted process; enforce a **max holding period** (vertical barrier).
4. **Evaluate**: For each ((\pi,\bar\pi)), apply exits on all paths, collect terminal P&L, compute **Sharpe**.
5. **Select** (three modes):

   * **OTR**: Choose ((\pi,\bar\pi)) with highest Sharpe.
   * **Given (\bar\pi)**: derive optimal (\pi).
   * **Given (\pi)**: derive optimal (\bar\pi).

---

## Empirical patterns you should encode as heuristics/features

* **Zero long-run equilibrium** ((E[P_T]=0), market-maker regime):

  * Best: **small profit-take, large stop-loss** (harvest small mean-reversions, tolerate drawdowns).
  * Worst: **tight stop-loss, large profit-take**.
  * As (\tau\uparrow) ((\phi\to1)): structure fades; Sharpe → 0; **no robust OTR**—beware overfit.
* **Positive equilibrium** ((E[P_T]>0), trend/position-taker):

  * Optimal region: **moderate (\bar\pi)** with **broad (\pi)** (rectangular heat-map).
  * As (\tau\uparrow): optimal (\bar\pi) shifts lower, region spreads, Sharpe declines.
* **Negative equilibrium** ((E[P_T]<0)): mirror of positive case (worst where positive case is best).

---

## What to teach the LLM (policy head or rule recommender)

* **Inputs**: ((\hat\phi,\hat\sigma,;E[P_T]\text{ sign/magnitude},;\text{maxHP})) + opportunity context.
* **Outputs**: recommended ((\pi^*,\bar\pi^*)) (and expected Sharpe).
* **Training data**: auto-generate via Monte Carlo on the fitted process; label with best ((\pi,\bar\pi)) per grid; include **confidence** (Sharpe CI or stability across seeds).
* **Guardrails**:

  * If (\phi\approx1) (near random walk), **down-weight** any sharp OTR; suggest **wider bands** or **no fixed thresholds**.
  * Report **heat-map stability** across seeds/periods; penalize rules that flip with small perturbations (overfit signal).
  * Account for **transaction costs** post-selection; Sharpe is scale-invariant in position size, costs aren’t.

---

## Implementation notes

* Use **per-unit P&L** (Sharpe unaffected by position scaling).
* **Parallelize** grid × paths; precompute per product; **recalibrate** when (\hat\phi,\hat\sigma,E[P_T]) drift.
* Vertical barrier = **max holding period** (triple-barrier link).
* A working Python sketch is provided in the chapter (mesh + simulation + Sharpe).

---

## Bottom line

* **Don’t tune exits on a single historical path.** Fit a simple mean-reversion process, **simulate broadly**, and pick exits that are **process-optimal**. Empirically, an O-U-driven market admits a **unique optimal pair** ((\pi,\bar\pi)) that maximizes Sharpe; finding it numerically is fast and far safer than historical parameter sweeps. 

# Backtest Statistics

## Why this matters

* Regardless of backtest type (historical, scenario/CV, or synthetic), **results must be reported with a consistent, decision-useful metric set** so investors can compare strategies and detect hidden risks (asymmetry, capacity, costs). 

## Metric buckets to compute & log

**General characteristics**

* **Time range** (cover multiple regimes), **avg AUM**, **capacity** (AUM at target risk-adjusted return), **leverage** (avg dollar position ÷ AUM), **max dollar position size** (avoid reliance on extremes), **ratio of longs** (~0.5 for market-neutral), **bet frequency** (bets/year; count flips/flattenings, not trades), **avg holding period**, **annualized turnover**, **correlation to underlying** (flag unintended beta). 

**Performance (unadjusted)**

* **PnL** (total & by side), **annualized return** (TWRR with cash-flow handling), **hit ratio**, **avg gain** (hits), **avg loss** (misses). 

**Runs & drawdowns**

* **Concentration (HHI)** of **positive** and **negative** bet returns (prefer low), and **time concentration** of bets (low).
* **Drawdown (DD)** series and **Time under Water (TuW)** series; report **95th-percentile DD** and **95th-percentile TuW**. 

**Implementation shortfall (execution realism)**

* **Broker fees/turnover**, **slippage/turnover**, **$ performance per turnover**, **return on execution costs** (should be a large multiple to survive worse execution). 

**Efficiency (risk-adjusted)**

* **Sharpe ratio** (assumes IID; scale by √annualization factor).
* **Probabilistic Sharpe (PSR)**: adjusts SR for **track length**, **skewness**, **kurtosis**. Use as:
  ( \text{PSR} = \Phi!\Big(\frac{\hat{SR}-SR^*}{\sqrt{\frac{1-\hat{\gamma}_3 \hat{SR}+\frac{\hat{\gamma}_4-1}{4}\hat{SR}^2}{T-1}}}\Big) ) → target **PSR > 0.95**.
* **Deflated Sharpe (DSR)**: PSR with **multiple-testing correction**; benchmark (SR^*) increases with **#trials (N)** and **variance of SRs across trials**; target **DSR > 0.95**.
* **Third law of backtesting**: **Always disclose all trials (N) and their variance**, or the result’s false-discovery probability can’t be assessed. 

**Classification (for meta-labeling overlays)**

* **Accuracy, Precision, Recall, F1**, **negative log-loss**. Handle **degenerate cases** (all-positive or all-negative labels/predictions) where F1 is undefined—log warnings and fallbacks. 

**Attribution**

* Decompose PnL (and info ratios) by **risk classes** (e.g., duration, credit, sector, currency). Build **portfolio-weighted** vs **universe-weighted** indices per category to estimate contributions; non-orthogonality implies sums won’t exactly match total PnL. 

## Minimal disclosure set (ship these with every backtest)

* Time range; avg AUM; capacity estimate; leverage; max dollar position size; ratio of longs.
* Bets/year, avg holding period, annualized turnover, correlation to underlying.
* PnL (total/by side), TWRR, hit ratio, avg gain/loss.
* 95-pct **DD** & **TuW**; **HHI** (pos, neg, time).
* Costs: broker & slippage per turnover; **$-per-turnover**; **return-on-execution-costs**.
* **Annualized SR**, **Info ratio**, **PSR**, **DSR** with **N** trials & **Var(SR)** disclosed.
* Overlay classifier metrics; attribution summary by risk class. 

## Implementation hints (for your pipeline)

* Derive **bets** from **position flips/flattenings**; compute **holding period** from target-position time-weights; build **DD/TuW** from running HWM; compute **HHI** for returns/time buckets.
* Validate capacity: **high Sharpe + high turnover + high leverage + short holding period** ⇒ usually **low capacity**. Prefer stable metrics across regimes. 

# Understanding Strategy Risk

## What “strategy risk” is

* **Strategy risk ≠ portfolio (vol) risk.** It’s the chance the **strategy fails its target Sharpe** even if holdings look “safe.” Quantify it from **betting precision *p***, **bet frequency *n***, and **payout asymmetry (π⁺, π⁻)**—not just return variance. 

## Core relationships your pipeline should encode

* **Symmetric payouts (π⁺ = −π⁻ = π):**
  [
  \theta(p,n)=\frac{2p-1}{2\sqrt{p(1-p)}}\sqrt{n}
  ]
  Implication: even tiny edge (**p > 0.5**) can reach high Sharpe if **n** is large (basis of HFT).
* **Precision–frequency tradeoff (symmetric):** to hit target (\theta^*), required **p** increases as **n** falls (and vice-versa).
* **Asymmetric payouts (π⁺≠π⁻):**
  [
  \theta(p,n,\pi^-,\pi^+)=
  \frac{(\pi^+-\pi^-)p+\pi^-}{(\pi^+-\pi^-)\sqrt{p(1-p)}}\sqrt{n}
  ]

  * **Implied precision to reach (\theta^*):** solve quadratic (ap^2+bp+c=0) with
    (a=(n+\theta^{*2})(\pi^+-\pi^-)^2,;; b=[2n\pi^--\theta^{*2}(\pi^+-\pi^-)](\pi^+-\pi^-),;; c=n\pi_-^2).
  * **Implied frequency to reach (\theta^*)** (given (p)):
    [
    n=\frac{\big(\theta^*(\pi^+-\pi^-)\big)^2,p(1-p)}{\big((\pi^+-\pi^-)p+\pi^-\big)^2}
    ]
    (Check for extraneous roots by plugging back into (\theta(\cdot)).)

## Probability the strategy fails (deploy/no-go metric)

1. **Estimate payouts & frequency:**
   (\pi^-=\mathbb{E}[\text{losses}],; \pi^+=\mathbb{E}[\text{gains}],; n=) bets/year.
   (Optionally fit a 2-Gaussian mixture to returns to get (\pi^-,\pi^+).)
2. **Bootstrap/KDE the precision distribution (f[p]):** resample **(nk)** bet outcomes for investment horizon **k** (e.g., 2 years) and compute (p_i).
3. **Compute threshold precision (p_{\theta^*}):** minimal **p** that attains target (\theta^*) using the implied-*p* formula above.
4. **Strategy risk:** ( \Pr[\text{fail}] = \Pr[p < p_{\theta^*}] = \int_{-\infty}^{p_{\theta^*}} f[p],dp).
   **Rule of thumb:** discard if (\Pr[\text{fail}]>0.05).
   (Closely related to PSR/DSR, but focuses on levers **under PM control**: (\pi^-,\pi^+,n).)

## Practical takeaways for model design

* **Sensitivity matters:** When (\pi^-) is large in magnitude (bad tails) or **n** is small, the **required (p)** rises sharply—strategies become brittle.
* **Small (p) drifts can kill edge:** e.g., (p:0.70\to0.67) can push (\theta) to ≈0 with typical asymmetries/frequencies.
* **Edge sources differ:**

  * **HFT:** tiny (p-0.5), huge **n**.
  * **Swing/position-taking:** moderate **n**, must improve (p) and/or payout asymmetry (raise (\pi^+/\lvert\pi^-\rvert)).
* **Stop/target design is economics:** For fixed (\pi^+,\pi^-), you can trade **frequency** against **precision** to hit (\theta^*); heat-map this surface during tuning.

## What to implement (pipeline checklist)

* **Estimate** ((\pi^-,\pi^+,n,p)) per strategy/version/regime; store by date.
* **Compute**: (\theta(\cdot)), implied-*p* for (\theta^*), implied-*n* for (\theta^*), and **(\Pr[\text{fail}])** via bootstrap/KDE (normal approx only as a fallback).
* **Gatekeeping:** ship **(\Pr[\text{fail}])** with CI; **block deploy** if over 5% unless risk committee approves.
* **Diagnostics:** sensitivity of (\theta) to each parameter; heat-maps over ((p,n)) and ((\pi^-,\pi^+)); flag extraneous roots when inverting formulas.
* **Governance:** compare with PSR/DSR; reconcile divergences (PSR high but (\Pr[\text{fail}]) high ⇒ payout asymmetry/precision fragile).

> Bottom line: **Model the strategy as a binomial bet machine.** Tie performance to **precision, frequency, and payout asymmetry**, then **quantify failure probability**. Optimize levers you control ((\pi^-,\pi^+,n)) and only ship edges that remain robust to small (p) shocks. 

# Machine Learning Asset Allocation

* **Why HRP exists**

  * Classical mean–variance optimizers (e.g., Markowitz/CLA) are **unstable**, **concentrated**, and often **underperform out-of-sample** due to covariance inversion on ill-conditioned matrices (“Markowitz’s curse”).
  * Naïve risk parity (IVP) ignores correlation structure, leaving portfolios exposed to **systemic shocks**. 

---

## Core Idea (HRP)

* Replace full-matrix inversion with a **hierarchical (tree) structure** built from correlations, then **allocate top-down** to clusters and **within** clusters by inverse variance.
* Works even when the covariance matrix is **singular**; avoids numerical fragility inherent to quadratic programming. 

---

## Algorithm (3 Stages)

1. **Tree Clustering**

   * Compute correlations ρ and **distance** (d_{ij}=\sqrt{\tfrac{1}{2}(1-\rho_{ij})}).
   * Build a hierarchical clustering (e.g., single-linkage) to get a **linkage matrix** (the tree). 
2. **Quasi-Diagonalization**

   * Reorder rows/cols of the covariance so similar assets form **blocks** along the diagonal (no basis change). 
3. **Recursive Bisection (Allocation)**

   * Repeatedly **split** the ordered list into adjacent halves.
   * For each half, compute cluster variance via **inverse-variance weights**; **split weight** inversely to cluster variances.
   * Guarantees **non-negative weights summing to 1**; complexity ≈ **O(n)–O(log n)**. 

---

## What to Implement (minimal, robust)

* **Inputs:** asset return matrix (X) → covariance (V), correlation (C).
* **Distance metric:** (d=\sqrt{\tfrac{1}{2}(1-C)}).
* **Clustering:** linkage on distance (e.g., SciPy `linkage(dist, 'single')`).
* **Ordering:** convert linkage → leaf order (quasi-diag).
* **Cluster variance:** IVP within each subset; variance via quadratic form.
* **Top-down split:** allocate between left/right subsets by inverse cluster variances; recurse until singletons. 

---

## Empirical Findings (design targets)

* **Out-of-sample variance:** HRP < IVP < CLA in Monte Carlo tests (large margin; CLA worst despite in-sample min-var objective).
* **Concentration:** CLA heavily concentrates (zeros/narrow top weights); HRP produces **intuitive, diversified** allocations across clusters.
* **Rebalancing:** CLA reacts erratically → **higher turnover**; HRP **stable** paths → lower implied costs. 

---

## Practical Notes for an LLM-Driven System

* **Stability over exactitude:** prefer HRP for default allocation; treat CLA outputs as **diagnostics** (not production) unless covariance is well-behaved.
* **Metrics to monitor:** condition number of (C), turnover, out-of-sample variance, cluster composition stability, concentration (top-k weight share).
* **Constraints/extensions:** easy to add caps/floors, BL views, Ledoit–Wolf shrinkage, expected-return tilts, or use cluster splits instead of strict bisection.
* **Generality:** same hierarchy trick can stabilize **ensemble weighting**, **manager allocation**, and **bagging/boosting** of ML signals. 

---

## Key Formulas (use verbatim)

* **Correlation distance:** (d_{ij}=\sqrt{\tfrac{1}{2}(1-\rho_{ij})})
* **IVP weights in subset (S):** (w_k \propto 1/\text{Var}(k)), normalized on (k\in S)
* **Split factor:** (\alpha = 1-\dfrac{\tilde V_L}{\tilde V_L+\tilde V_R}) (allocate (\alpha) to left subset, (1-\alpha) to right) 

---

## When to Prefer Each

* **HRP (default):** many correlated assets; limited history; need robustness & interpretability.
* **IVP:** quick baseline; when correlations are unreliable but variances are stable.
* **CLA/min-var:** small, well-conditioned universes with strong confidence in (V^{-1}); accept concentration risk. 

---

## Testing Blueprint

* **Train/Validate loop:** rolling lookback (e.g., 1y) → rebalance monthly; compare HRP vs IVP vs CLA.
* **KPIs:** out-of-sample variance, Sharpe, max drawdown, turnover, weight entropy, top-k concentration.
* **Stress:** inject idiosyncratic & common shocks; vary N, conditioning of (C), and rebalance cost model. 

---

**Bottom line:** Use **hierarchy over inversion**. Build the tree from correlations, quasi-diagonalize, and allocate top-down by inverse cluster variance. You’ll get **more stable, less concentrated**, and **lower-variance** portfolios out-of-sample. 

