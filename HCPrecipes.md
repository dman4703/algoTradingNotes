# Multiprocessing & Vectorization

## Why this chapter matters

* CPU-bound ML workloads in Python won’t scale with threads due to the **GIL**; you need **vectorization** and **multiprocessing** (and, if available, clusters) to use the hardware you paid for. 

## Core ideas to bake into your system

* **Vectorize first**: replace explicit Python loops with array ops / compiled iterators (e.g., `itertools.product` → generator of jobs). Scales to arbitrary dimensions without changing code. 
* **Prefer multiprocessing over multithreading** for CPU-bound tasks; processes don’t share memory, so plan for **explicit data passing**. 
* **Multi-level parallelism**: (1) vectorized kernels, (2) multiprocessing across cores, (3) distribute across nodes/HPC when available. 

## Work unit model you should implement

* **Atoms** = indivisible tasks.
* **Molecules** = groups of atoms each handled by one worker (the unit of scheduling). Parallelism happens at the **molecule** level. 

## Partitioning strategies (load balance ≫ elegance)

* **Linear partition** (`linParts`): split a 1-D list of atoms into ~equal slices. Simple & fast. 
* **Two-nested-loops partition** (`nestedParts`): when cost grows across a triangular index space, assign **row blocks** so each molecule has ~equal work; supports **lower** and **upper** triangular cases. Prevents stragglers. 
* **Over-decompose** with `mpBatches`: create > cores molecules to smooth stragglers (heavy molecules first if needed). 

## Engine design pattern (what to build)

* **Generic job runner** (`mpPandasObj` / `mpJobList`):

  * Inputs: `func` (callback), `(argName, atomList)`, `numThreads`, `mpBatches`, partition type (`linMols` vs nested), and `**kwargs`.
  * Output: stitch per-molecule results into a `Series`/`DataFrame` or a user-defined reduction. 
* **Synchronous debug path** (`processJobs_`): run jobs sequentially when hunting Heisenbugs. Flip to parallel only after it’s clean. 
* **Asynchronous execution**: use `multiprocessing.Pool.imap_unordered(...)` + a **progress reporter** to stream results without waiting for slow jobs. 
* **Callback unwrapping** (`expandCall`): accept each job as a dict, pop `'func'`, call `func(**kwargs)`. This decouples the engine from the function signature. 

## Inter-process compatibility

* **Pickling bound methods**: processes must pickle callables; add the standard bound-method (un)pickling shim so class methods can be dispatched. 

## Memory-safe output handling (must-have)

* **On-the-fly reduction** (`processJobsRedux`): instead of keeping all per-molecule outputs in RAM, **reduce** them as they arrive:

  * `redux` (callable, e.g., `DataFrame.add`, `dict.update`, `list.append`)
  * `reduxArgs` (kwargs)
  * `reduxInPlace` (True for in-place ops)
    This avoids OOM and cuts post-processing latency. 

## Canonical pattern for big matrix ops

* Example: computing principal components when data `Z` is too large to load at once.

  * **Shard by columns** into files (`Z_b`), stream one shard per molecule, multiply by the corresponding rows of `Ŵ` (`np.dot(df.values, eVec.loc[df.columns].values)`), and **sum** partial PCs with an on-the-fly reducer.
  * Benefits: bounded RAM, parallel speedup. 

## Config knobs your LLM/system should expose

* `numThreads` (≤ physical cores), `mpBatches` (over-decomposition factor), `partition` (`linear`/`nestedLower`/`nestedUpper`), `redux`/`reduxArgs`/`reduxInPlace`, `debugSequential` flag, and `progress` verbosity. 

## Anti-patterns to catch and reject

* Spawning one process per atom (scheduler thrash).
* Keeping **all** results in memory before reducing (OOM risk).
* Using threads for CPU-bound loops (GIL bottleneck).
* Hard-coding dimensionality (breaks generality and vectorization). 

## Minimal API your engine should implement

```text
run_parallel(func, atoms, arg_name="molecule",
             num_threads=24, mp_batches=1,
             partition="linear|nestedLower|nestedUpper",
             redux=None, redux_args=None, redux_in_place=False,
             debug=False, **func_kwargs) -> ReducedResult
```

* Guarantees: stable load balance, async progress, deterministic reduction semantics, and a debug path. 

---

Use this as scaffolding to teach the LLM when to **vectorize**, how to **partition atoms into molecules**, how to **launch processes asynchronously**, and how to **reduce outputs on the fly**—while staying robust to pickling limits and memory ceilings. 

# Brute Force & Quantum

## The problem (in one line)

Find a **global dynamic** portfolio trajectory across (H) horizons that **maximizes Sharpe Ratio** while paying **non-convex, non-smooth transaction costs**—a setup that breaks classic convex optimizers. 

## Core formulation

* Assets (X={x_i}_{i=1}^N); per-horizon forecasts ((\mu_h, V_h)); transaction costs (\tau_h[\omega]).
* Trajectory (\omega \in \mathbb{R}^{N\times H}) with **full-investment** each horizon: (\sum_{i=1}^N |\omega_{i,h}|=1).
* Costs (example): (\tau_1=\sum_i c_{i,1}\sqrt{\lVert \omega_{i,1}-\omega_i^*\rVert}), and for (h\ge2): (\tau_h=\sum_i c_{i,h}\sqrt{\lVert \omega_{i,h}-\omega_{i,h-1}\rVert}).
* Objective:
  [
  \text{SR}(\omega)=\frac{\sum_{h=1}^{H}\mu_h' \omega_h - \tau_h[\omega]}
  {\sqrt{\sum_{h=1}^{H}\omega_h' V_h \omega_h}}
  ]
  Not convex due to time-varying ((\mu_h,V_h)), non-smooth (\tau), and SR’s ratio form.

## Discretize → integer optimization (enumerable search space)

* Choose **K “units of capital”**; generate all **pigeonhole partitions** of (K) into (N) slots (order matters).
* Convert each partition to absolute weights (|\omega_i|=p_i/K); apply all (2^N) sign patterns ⇒ **Ω = all feasible weight vectors** for a horizon.
* Trajectories are **Cartesian products** of Ω repeated (H) times ⇒ **Φ = all feasible trajectories**.
* Evaluate each trajectory: compute (\tau[\omega]) sequentially across (h), then SR; keep the argmax.

## Why quantum helps

* Digital machines enumerate Φ **sequentially** (NP-hard scale).
* **Qubits** can represent **superpositions** of candidate solutions; **quantum annealers**/algorithms can “search” combinatorial spaces more effectively, enabling **quantum brute force** for the same discretized problem. (Demonstrated on optimal trading trajectory via a quantum annealer.)

## Implementation blueprint (what your system should do)

1. **Generate partitions** (p_{K,N}) (pigeonhole/“stars and bars”) and **permute** when order matters.
2. **Build Ω**: map partitions → absolute weights → apply all sign vectors.
3. **Build Φ**: Cartesian product of Ω across (H).
4. **Evaluate** each trajectory:

   * **Transaction costs** (\tau_h) via running difference of (\omega_{:,h}).
   * **SR** via accumulated mean and variance terms.
5. **Select** the best trajectory; return (\omega^*), its SR, and diagnostics (costs by horizon, turnover).

## Practical guardrails (for scale + correctness)

* **Keep K small** initially (e.g., 3–10 per asset) to bound (|Φ|); increase only if runtime allows.
* **Vectorize** cost/variance ops; **multiprocess** over trajectories; cache repeated (\omega'V_h\omega) components by reusing subresults.
* **Prune** early with admissible bounds (upper bound on SR for partial prefix of horizons); short-circuit losers.
* **Exploit symmetry** (e.g., identical assets, zero weights) to deduplicate.
* **Record seeds and configs** (K, N, H, signs policy) for reproducibility and for LLM training metadata.

## Static vs dynamic

* **Static**: optimize each horizon independently (local optima), then stitch; can be far from global optimum when costs/path-dependence matter.
* **Dynamic**: global search over Φ captures **path-dependent** costs and regime shifts; dominates static when turnover is penalized and forecasts vary across time.

## What to expose to an algorithm-training LLM

* **Inputs**: ((\mu_h, V_h, c_{:,h}), K, N, H, \omega^*, \text{SR}^*, \tau_h, \text{runtime}).
* **Actions**: choose (K), pruning heuristic, parallelism level, search policy (full enumeration vs beam/anneal).
* **Rewards/labels**: achieved (\text{SR}), regret vs best known, cost breakdown, #trajectories explored.
* **Curriculum**: start with tiny ((N,H,K)); scale up; compare **static vs dynamic** SR to teach why path-dependence matters.

## When to switch from brute force

* If (|Φ|) explodes (large (N,H,K)): move to **heuristics** (beam search, genetic, coordinate search) or **quantum annealing** with the same discretization. Keep brute force as **oracle** for small cases (unit tests and LLM evaluation).

# High-Performance Computational Intelligence & Forecasting

## Mission & Context

* **Goal:** Bring **HPC tools** (not just cloud) to **streaming, time-critical analytics** (finance, grid, traffic, science). Triggered by the 2010 Flash Crash where ~**20 TB** delayed regulators; HPC routinely handles **100s of TB in minutes**. 

## Why HPC over (only) Cloud for Streaming

* **Latency first, not just throughput.** Streaming windows are small; you parallelize **within a timestep** (fine-grained), not across millions of objects. HPC software/hardware fits this pattern. 
* **Virtualization overhead kills scale.** Scientific apps ran **2–10× slower** on commercial cloud; a case hit **53×** slowdown (virtualized networking stalls scaling). 
* **Cost:** Studies show dedicated HPC often **cheaper (≈2–7×)** for comparable workloads, especially with heavy data ingress/egress. 

## Core HPC Software to Leverage

* **MPI** for low-latency, explicit inter-process messaging (standard on every supercomputer). 
* **HDF5** for array-native I/O: compression, chunking, indexing → fewer bytes over the wire, faster scans. **21× faster** than ASCII in tests. 
* **In-situ / in-transit** analytics with **ADIOS/ICEE**: analyze during I/O, discard irrelevancies before disk/network bottlenecks; used for real-time distributed workflows. 

## Design Patterns Your LLM System Should Internalize

* **Work decomposition:** Split compute **inside each tick/window** across cores; pipeline I/O→compute→reduce with back-pressure. 
* **Data model:** Store as **multi-dimensional arrays** (HDF5 groups/datasets). Turn on **chunking + compression + indices** for selective reads. 
* **In-situ reduction:** Filter/aggregate **before** write; stream compact state to disk/network. 
* **Calibration loops at scale:** Fast recomputation enables broad **parameter sweeps** and **false-positive minimization**. 

## Measured Wins (what to aim for)

* **Flash-Crash indicators at scale:**

  * HDF5 vs ASCII for SP500/10y: **~3.5 h → 604 s** on 1 core; with **512 cores: 2.58 s** (plus **3.7×** extra via HDF5 indexing). 
* **VPIN engine:** **720×** faster (≈**1.5 s** per contract for **67 months** of futures vs ~18 min legacy). Allows robust hyper-parameter search. 
* **VPIN false-positive drop:** Average **20% → 7%** via a tuned set (median-price bars; 200 buckets/day; 30 bars/bucket; 1-day support; 0.1-day duration; bulk-volume t-dist ν=0.1; CDF threshold 0.99). 

## Use-Case Signals (templates for features/labels)

* **Market microstructure early warnings:** **VPIN + fragmentation HHI** computed intraday at scale for pre-crash conditions. **Label events** by threshold crossings + subsequent volatility. 
* **Non-uniform FFT (NUFFT) for HFT/TWAP fingerprints:** Strong **once-per-minute** spectral line; leap-year daily frequency check validates method; track growth of high-frequency components over years. 
* **Fusion plasma “blob” tracking:** Distributed feature extraction with MPI + shared memory; **ms-level per timestep**; pattern applies to any spatiotemporal segmentation task. 
* **AMI electricity baselines:** **LTAP** (piecewise-linear vs temperature) beats black-box GTB for long-horizon baselining; avoids error accumulation from lagged-feature recursion. Good example of **white-box + constraints**. 
* **Astronomy triage:** Automated classification pipeline found **SN 2011fe ~11 h** post-explosion → shows value of **fast classifier + confidence** for scarce follow-ups. 

## Practical Guardrails

* **Treat ASCII as a last resort.** Standardize on HDF5 with explicit schema; precompute **indices** for hot queries. 
* **Minimize virtualization layers** on latency-critical paths (network, storage). If cloud is unavoidable, use bare-metal or SR-IOV-backed instances. 
* **Measure end-to-end:** Time each stage (ingest, parse, window, compute, reduce, emit). Set SLOs aligned to market/device cadence (e.g., **≤1 s** per 1-min bar). 
* **Curriculum for the LLM:** Start with small windows/assets; scale **N×T** while keeping latency budgets; include **parameter-search traces** and **ablation runs** as supervision. 

## What to Log/Expose for Training

* **Inputs:** windowed arrays (prices/volumes/quotes or sensor fields), metadata (clock, exchange, instrument), config (bar size, buckets, kernels).
* **State:** intermediate aggregates (e.g., VPIN buckets, HDF5 dataset stats, spectral peaks), calibration grids, false-positive curves.
* **Outputs:** alerts with confidence, latency stats, resource usage, and rollback decisions. 

## Bottom Line

If you need **near-real-time** decisions on **streaming, dependency-rich** data, the winning combo is: **MPI + HDF5 + in-situ pipelines** on HPC-class infrastructure, with fast **re-calibration** loops. It’s faster, often cheaper, and—critically—**on time**. 