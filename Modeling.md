# Ensemble Methods

## What ensembles fix

* Prediction error decomposes into **bias² + variance + irreducible noise**. Ensembles are tools to reduce bias and/or variance; noise is unavoidable. 

## Bagging (Bootstrap Aggregation)

* **How it works:** sample with replacement → train N independent base learners → **average** predictions (or majority vote). Parallelizable. 
* **Variance reduction math:** with average single-model variance (\bar\sigma^2) and average inter-model correlation (\bar\rho):
  [
  \mathrm{Var}!\left(\tfrac{1}{N}\sum \hat\phi_i\right)=\bar\sigma^2\Big(\tfrac{1-\bar\rho}{N}+\bar\rho\Big)
  ]
  So bagging helps **only if (\bar\rho<1)** and improves as N↑ and correlations↓. Driving down correlation is as important as adding estimators. 
* **Accuracy uplift condition (classification):** majority vote can beat average base accuracy once N is large **and** base accuracy (p>\frac{1}{k}) (k classes). Bagging won’t rescue uniformly poor learners. 
* **Practical pitfall in finance:** non-IID, overlapping labels → bootstraps look alike → (\bar\rho\approx1) ⇒ little variance reduction; **out-of-bag (OOB) accuracy gets inflated**. Prefer **stratified k-fold without shuffling** (small k) and use **sequential bootstrap** / **max_samples ≈ average label uniqueness**. Ignore OOB in this setting. 
* **Sklearn quirk:** historic OOB bug with label encoding in `BaggingClassifier`; renumber labels sequentially if using older setups. 

## Random Forest (RF)

* **Bagging + feature subsampling per split** to further decorrelate trees and cut variance; gives **feature importance**, but **bias may remain**. OOB accuracy can still be inflated on non-IID data. 
* **When data are redundant:** default bootstraps (same size as train set) yield many near-identical, overfit trees. Mitigations:

  * Lower `max_features` to force tree diversity.
  * **Early stopping**: increase `min_weight_fraction_leaf` until OOB≅CV.
  * Wrap **`DecisionTreeClassifier` or `RandomForestClassifier` in `BaggingClassifier`** with `max_samples = avg_uniqueness`.
  * Swap standard bootstrapping for **sequential bootstrap**.
  * Rotate features (e.g., **PCA**) to axis-align splits; use `class_weight='balanced_subsample'` for imbalance. 

## Boosting (e.g., AdaBoost)

* **Sequential** training; re-weights observations; combines learners with **performance-weighted averaging**. It reduces **variance and bias**, but **overfits more easily**, especially in low signal-to-noise financial data. Often **prefer bagging** first in finance. 

## Bagging for scalability

* For algorithms that **don’t scale** (e.g., large-N SVMs), train many **early-stopped** base models in parallel (e.g., low `max_iter` or looser `tol`) and bag them. The per-model variance increase is outweighed by ensemble variance reduction. 

---

## Implementation checklist (finance-aware)

1. **Labeling & uniqueness:** compute average **label uniqueness** (overlap-aware); set `max_samples ≈ avg_uniqueness`. Use **sequential bootstrap** to reduce (\bar\rho). 
2. **Choose ensemble:** start with **bagging** (trees or your preferred base learner). Consider RF only after enforcing diversity (low `max_features`, early stopping). Use **boosting** cautiously. 
3. **Regularization knobs:** `min_weight_fraction_leaf`, `max_depth`, `max_features`, early-stop params (`max_iter`, `tol`), and class weights; consider **PCA** pre-processing. 
4. **Validation protocol:** **avoid OOB** on financial, overlapping data; use **stratified k-fold (no shuffle, small k)** that respects temporal/overlap structure. Report CV not OOB. 
5. **Diversity over quantity:** once (\bar\rho) is low, increase N; otherwise prioritize decorrelation (feature subsampling, different seeds, sequential bootstrap) over simply adding more estimators. 

## Do / Don’t

* **Do:** decorrelate learners; respect non-IID structure; use overlap-aware sampling; tune for **variance** first in finance. 
* **Don’t:** trust OOB on overlapping labels; expect bagging to fix universally weak base models; assume more trees ≡ better when trees are near-identical. 

---

*Use this as system context for your LLM so it picks ensembles that **lower correlation**, validate properly on **non-IID** data, and favors **variance control** before bias correction in financial applications.* 

# Cross-Validation

## Why standard CV fails

* **Non-IID reality:** Financial features are serially correlated and **labels often span overlapping time windows**, so naive k-fold CV leaks information between train/test. Reported scores become **optimistic**. 
* **Test-set reuse:** Hyperparameter search repeatedly “peeks” at the test folds → **multiple testing / selection bias**. CV itself can **contribute to overfitting**. 
* **Shuffling is harmful:** Shuffling breaks temporal structure and **amplifies leakage**; higher k (more, smaller folds) typically **increases** overlap-driven leakage and inflates performance. 

## What counts as leakage (and what doesn’t)

* Leakage occurs when **(Xᵢ, Yᵢ) ≈ (Xⱼ, Yⱼ)** is split across train/test due to **overlapping label intervals** and serial correlation—**not** merely because features overlap. 

## The fix: Purged k-fold + embargo

* **Purge:** For each test fold, **drop any training sample whose label interval overlaps** the test labels. (Use label start/end times to detect interval intersections.) 
* **Embargo:** After each test fold, **exclude a short post-test window** (e.g., ~1% of the sample) from training to break near-term serial dependence. 
* **Behavioral check:** With proper purging/embargo, performance may improve with k initially (better recalibration) but **plateaus past k***; **unbounded gains with k** signal residual leakage. 

## Implementation essentials

* **Inputs you must carry:**

  * A Series `t1` giving each sample’s **label “through” (end) time**; index aligned to `X`.
  * `pctEmbargo` (fraction of samples to embargo after each test block). 
* **Splitting policy:** Test folds are **contiguous, non-shuffled** blocks. Train = all samples **outside** test, **minus**: (i) any **overlapping** label intervals (purge), (ii) the **post-test embargo** window. 
* **Class to use:** Implement a **`PurgedKFold`** (sklearn-style) that enforces the above logic; ensure `X.index == t1.index`. 
* **Scoring gotchas (sklearn):**

  * Some sklearn scorers lack access to `classes_`; pass labels explicitly for `log_loss`.
  * `cross_val_score` historically **mismatched sample weights** between `fit` and `log_loss`. Prefer a custom `cvScore` that forwards **sample weights consistently**. 

## Protocol for model development

1. **Never shuffle.** Use **PurgedKFold + embargo** for **all** tuning, evaluation, and backtests. 
2. **Guard against selection bias:** Constrain search breadth, use **nested** purged CV when feasible, and reserve a **final untouched test** evaluated **once**. 
3. **Report robustly:** Provide **per-fold scores**, dispersion (e.g., mean ± std), and **k-sensitivity**; show that scores **do not keep rising** as k→T. 

## Quick checklist (drop-in for your pipeline)

* [ ] Build/compute label intervals (e.g., from triple-barrier) → `t1`.
* [ ] Use **contiguous folds**; **purge** overlaps; **embargo** post-test window.
* [ ] **No shuffling**, no naive k-fold/OOB.
* [ ] Use a **custom CV runner** that passes `sample_weight` and `labels` correctly.
* [ ] Limit and log hyperparameter trials; consider **nested** purged CV.
* [ ] Validate that performance **plateaus** with k; if not, re-audit leakage. 

# Feature Importance

## Core principle

* **Don’t backtest your way into “truth.”** Repeating tests until a backtest looks good is selection bias. Use **feature importance** to learn *why* a model works **before** any backtest (“Backtesting is not a research tool. Feature importance is.”). 

## Substitution & masking effects (why importance can lie)

* **Substitution (multicollinearity):** Related features can steal each other’s credit; importance is diluted or misassigned.
* **Masking in trees:** Some features never get considered if others dominate early splits.
* **Mitigation:** Work with **orthogonalized features (PCA)** before importance; lower tree **`max_features`** to 1 at split time to force diversity; treat zero importances as **missing** (not 0) when `max_features=1`. 

## Three complementary importance methods

* **MDI (Mean Decrease Impurity)** — *Fast, in-sample, tree-specific.*

  * Pros: cheap; importances sum to 1; great for quick diagnostics.
  * Cons: **IS-only**, tree-only, biased toward high-cardinality features, suffers substitution and masking; every feature gets some nonzero score even if useless.
  * Practice: use RF/Bagging with decision trees, **`max_features=1`**, replace 0→`NaN` before averaging. 
* **MDA (Permutation/Mean Decrease Accuracy)** — *Model-agnostic, OOS.*

  * Procedure: fit → score OOS (with **purged k-fold + embargo**) → **permute one column** → re-score; drop in score = importance.
  * Pros: works with any model and metric (accuracy, **log-loss**, **F1** for meta-labels).
  * Cons: **Still fooled by substitution** (correlated features look unimportant); can find **all features “unimportant”** if the model truly has no OOS signal.
  * Notes: pass labels explicitly for `log_loss`; ensure proper sample weights; never use naive CV here. 
* **SFI (Single-Feature Importance)** — *Model-agnostic, OOS, no substitution.*

  * Procedure: evaluate each feature **alone** via purged CV.
  * Pros: avoids substitution entirely.
  * Cons: **misses interactions/hierarchies** (a feature useful only with another looks weak solo). 

## Orthogonal features (PCA) — practical workflow

* Standardize X → compute eigenvectors/eigenvalues → **P = ZW**; keep PCs to explain ~**95% variance**; run MDI/MDA/SFI on **P**.
* Sanity check: compute **weighted Kendall’s τ** between (importance ranks) and (PCA eigenvalue ranks). High τ ⇒ PCA (unsupervised) and importance (supervised) agree → **lower overfit risk**. 

## Universe-level strategies

* **Parallelized importance:** compute per-instrument, then aggregate. Pros: fast, scalable; Cons: rank variance from substitution across instruments.
* **Stacked importance:** transform each instrument (rolling standardization) and **stack** into one dataset; one model learns cross-instrument importance. Pros: more general, less overfit to a single name; Cons: heavier compute/memory. Prefer **stacking** when feasible. 

## What to expect on synthetic data

* **MDI & MDA**: typically rank **informative & redundant** features above **noise**; MDA shows higher variance but is OOS; both suffer with correlated features.
* **SFI**: can underrank genuinely useful features that act only via **interactions**. Use as a **complement**, not a replacement. 

## Drop-in implementation checklist (for your pipeline)

* **CV policy everywhere:** **Purged k-fold + embargo**; never shuffle time.
* **MDI:** RF/Bagging of trees, `criterion='entropy'`, `class_weight='balanced'`, **`max_features=1`**, consider `min_weight_fraction_leaf` to stabilize.
* **MDA:** Same CV; permute one column at a time; support `accuracy`/**`neg_log_loss`**/**`F1`**; pass `classes_` correctly to `log_loss`; propagate **sample weights**.
* **SFI:** Same CV; evaluate each feature alone; record mean ± SE across folds.
* **Orthogonalization:** Optional but recommended: PCA to 95% variance; then recompute importance; compute **weighted τ** vs. PCA ranks.
* **Reporting:** Always give **means + SE**, show **consensus across MDI/MDA/SFI**, and highlight **disagreements** (often interactions or multicollinearity). 

## Do / Don’t

* **Do:** combine **MDI + MDA + SFI**; orthogonalize to reduce substitution; respect non-IID structure; audit importance stability over time/instruments.
* **Don’t:** trust naive CV/OOB in finance; assume one method is definitive; drop a feature solely because MDA says it’s “unimportant” in the presence of strong collinearity. 

# Hyper-Parameter Tuning w/ Cross-Validation

## Core idea

* Tune **only** under a **Purged K-Fold + embargo** CV to avoid look-ahead/overlap leakage; pass this splitter into `GridSearchCV`/`RandomizedSearchCV`, then **refit the validated pipeline on all data**. Optionally **bag** the tuned estimator to stabilize variance. 

## Practical recipe (drop-in)

* **CV generator:** `PurgedKFold(n_splits=cv, t1=label_end_times, pctEmbargo=p)`; **no shuffling**; contiguous folds. 
* **Search:**

  * Small/low-dim: `GridSearchCV`.
  * Large/high-dim or budgeted: `RandomizedSearchCV(n_iter=...)`.
  * For positive-only params (e.g., SVM **C**, **gamma**), sample from **log-uniform** over ([a,b]) rather than uniform. 
* **Scoring (finance):**

  * **Meta-labeling (imbalanced 0/1):** use **`f1`**.
  * **General strategies:** prefer **`neg_log_loss`** over accuracy (aligns with bet sizing by confidence and sample-weighted PnL). 
* **Sample weights:** scikit pipelines don’t pass `sample_weight` natively; wrap with a small subclass (e.g., **`MyPipeline.fit(..., sample_weight=...)`** that forwards `...__sample_weight` to the final step). 
* **Bagging the best model:** after search, optionally wrap the tuned pipeline with `BaggingClassifier(n_estimators, max_samples, max_features)` to reduce variance. 

## Implementation gotchas

* Older sklearn had a **`neg_log_loss` scorer quirk**; prefer a custom CV scorer that consistently forwards **labels and sample weights** (as outlined earlier in the book). 
* **Report** mean ± dispersion of CV scores; don’t trust accuracy alone—**probability-aware** loss (log-loss) is more faithful to risk. 

## Minimal checklist

* [ ] Build label intervals → `t1`; choose `cv`, `pctEmbargo`. 
* [ ] Use `GridSearchCV` or `RandomizedSearchCV` **with `PurgedKFold`**. 
* [ ] **Scoring:** `f1` (meta-labeling) or `neg_log_loss` (others). 
* [ ] Pass `sample_weight` via a patched pipeline (`MyPipeline`). 
* [ ] For positive-only params, draw from **log-uniform**; set an explicit **search budget** (`n_iter`). 
* [ ] Optionally **bag** the tuned estimator; refit on all data. 

**Bottom line:** Hyper-parameter tuning in finance = **purged CV + embargo**, **probability-aware scoring**, **weight-aware plumbing**, and (when big/imbalanced) **randomized, log-uniform search**—then **bag** and refit. 
