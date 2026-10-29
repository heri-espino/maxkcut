# Experiment023 — Full G-set comparison of missing algorithm families and graph topology

**Status: IMPLEMENTED, NOT YET VERIFIED COMPLETE.** A one-time, branch-specific GitHub Actions workflow launches the **three-family pilot** and, only if all pilots pass, schedules **all 71 canonical G-set graphs**. The Windows runner's live status and all graph-method results must be checked before claiming they completed. This work is independent of frozen Experiment021 and retrospective Experiment022.

## Question and motivation

Across the G-set's **30 random, 24 planar-family/almost-planar, and 17 toroidal** instances, which *rounding algorithm family* tends to produce the strongest binary signed Max-Cut when paired with the **same** downstream graph-local refinement? Can original topology and EVA-vector geometry help predict that choice on new graph instances?

Experiment021's 31,950 saved searches compared nine **ways to initialize spherical 2-means**, not EM, MST or multigroup rounding. Experiment022 proved only limited historical coverage for those earlier families (mostly G63). Experiment023 extends across the **canonical 71 graphs** without rerunning expensive EVA/SDP factor construction.

## Fixed source and evaluation

- Original signed, undirected graph: `data/G*.txt`, 71 canonical IDs G1–G67, G70, G72, G77, G81. This is **Max-Cut k=2**, not general k>2.
- Frozen nonconvex EVA factors, in priority order: Experiment021's completed G60/G61/G62, Experiment003's available verified large factors, Experiment002's verified older factors. No new factor is solved; graph/factor identities are recorded in every case.
- Every distinct hard binary candidate is scored using exactly `sum_(i<j) w_ij 1[x_i != x_j]` on the **original signed edges**.
- The common postprocessing sequence is: **20 positive flip sweeps → generalized KL 30 passes → ILS-KL 80 cycles, eight inner KL passes and kicks 8/16/32/64**. The pilot uses smaller KL/ILS budgets purely for code and resource validation; never pool it into final results.
- Original `SeedSequence` downstream streams: flips `[2026,g,restart,1313]`, KL `[2026,g,restart,101]`, ILS `[2026,g,restart,1401]` reused consistently. Initializer randomness is independently seeded by method ID. The historical original K-MeansNM baseline and alternate *Tabu-KL* postprocessor appear as controls.
- Save each restart's binary Round, Flip, KL and Final partitions, exact scores, timing, original factor and graph SHA256, algorithm configuration fingerprint and optimization trace. Re-scoring and SHA checks precede reuse. Running again must be **idempotent**; do not overwrite preexisting completed Experiments002–022.

## Registered methods (17 distinct protocols)

| Family | New full-protocol names | Important comparison caveat |
| --- | --- | --- |
| Baseline control | `control-paired` | Original K-MeansNM restart, but **new** evaluation budget; do not mix with Experiment021's best-of-50 |
| Exact hyperplane threshold | `gw-threshold` | Optimizes the threshold **conditional on the sampled projection**, not general Max-Cut |
| Cut-aware spherical rounding | `cutaware-kmeans` | Multiple directional proposals; overhead measured |
| Direct hyperplane arc optimization | `hyperplane-sweep` | Exact great-semicircle line sweeps per proposed tangent; overhead measured |
| Multi-group spherical K-means | `multigroup-k4`, `multigroup-k8` | Best exact binary merge of temporary 4 or 8 groups, NOT Max-4/8-Cut |
| Soft, non-elite mean-field clustering | `em-hard`, `em-cem`, `em-gibbs` | Soft EM threshold alone, CEM generation, or Gibbs annealing followed by common refinement; unlike earlier Experiment016 elite mode, **no old best solution is inserted** |
| Full-ambient-space Euclidean MST | `mst-l2-longest`, `mst-l2-balanced`, `mst-l2-diameter`, `mst-l2-bestcut`, `mst-l2-multicut` | Same exact MST can be cached per frozen factor. Preprocessing is O(n²p) and **shared/costly**; report cache hits and time. `bestcut` scores all original-graph signed cuts of a tree edge |
| Alternative MST distances | `mst-l1-bestcut`, `mst-linf-bestcut` | Distinct exact L1/L∞ vector-distance trees; not original-graph-weight trees |
| Different local-search *refiner* | `control-tabu` | Same original K-MeansNM seed but Tabu-KL instead of ILS-KL. **Separate postprocessor axis**—not comparable as an alternative initializer alone |

The earlier experimental grid included **additional hyperparameter variants, combinations and elite-source modes**. Their existence is recorded in Experiment022; Experiment023 tests **every major missing family**, not every historical grid cell. In particular, `em-cem-kl` collapses into the `em-cem` initializer under the shared downstream KL policy. The old elite-EM retention mode is intentionally **excluded from the no-elite primary comparison** to prevent trivial source inheritance; any future elite arm must record source and incremental gain separately.

## Two stages and hard cost limits

1. **Pilot:** G1 (random), G14 (almost planar), G11 (toroidal); 2 restarts per randomized algorithm, reduced KL/ILS; maximum ~120 seconds per algorithm with 270-second hard process timeout. If any algorithm fails or hits the hard timeout, the 71-graph matrix **does not begin**. The graph-feature extractor must also pass and save all original-graph measurements.
2. **Full:** 71 GitHub Actions matrix graph jobs, each covering **all 17 registered protocols**, up to 100 restarts *or* **240 seconds cumulative wall-clock method budget**, whichever comes first. A method exceeding 510 seconds is forcibly terminated; it is recorded as `hard_timeout`, **not a completed result**. Each graph job has a 360-minute cap. The suite uses **one** self-hosted Windows runner with `max-parallel=1` jobs and three concurrent non-MST subprocesses per graph; MST jobs are serialized to avoid duplicate tree caches.
3. **Final coverage:** a distinct job publishes a 71 × 17 = **1,207-cell** graph-method coverage table. `not_run`, `failed`, `hard_timeout`, `budget_limited`, and `completed_restarts` are explicit. It **fails** if any of the 1,207 cells lacks a verified completed/budget-limited status. A budget-limited result with `over_budget=true` is **censored for equal-time claims**; the raw score remains archived.

**Interpretation:** These initial results impose a per-method wall-clock ceiling, but not a guaranteed identical number of cut evaluations or a rigorous equal-time *anytime curve*. The exact O(n²p) MST preprocessing and expensive EM/hyperplane internal loops may overrun the budget; an algorithm that cannot return its first candidate within the cap is a documented failure or timeout, not a missing observation silently filtered from comparisons. The subsequent scientific analysis must account for paired graph identity, sign fraction, graph size, training/test separation, generator siblings, multiple comparisons, remaining runtime differences and deterministic vs randomized initializers.

**This is an expensive, long-running computational campaign.** Completion cannot be assumed based on merely committing or launching a GitHub workflow. All checkpoint commits stay on the experiment branch, and `main` and original source-factor archives remain untouched.

## Windows execution

The workflow is `.github/workflows/experiment023-method-topology.yml`, triggered **once** by insertion of that file on the named `experiment023-all-method-topology` branch. Subsequent result commits do not retrigger because the push path filter includes only the workflow file. A manual `workflow_dispatch` is also available for recovery, after checking currently active runner jobs. CI selects the existing Conda `maxkcut` Python; it does not reinstall or upgrade the environment.

From the repository root in the Windows Conda environment:

```powershell
python -m pytest tests/test_experiment023_methods.py -q
python -m maxkcut.experiment023_methods inventory
python -m maxkcut.experiment023_methods batch --graph G11 --phase pilot --budget-seconds 120 --timeout-seconds 270
python -m maxkcut.experiment023_methods batch --graph G11 --phase full --budget-seconds 240 --timeout-seconds 510
python -m maxkcut.experiment023_methods aggregate --phase full
```

The matching PowerShell runner harness is `scripts/experiment023-windows.ps1`, which checkpoints and pushes successful graph outputs after a safe rebase on the exact experiment branch. Its push loop refuses a rebase conflict. The original experimental data and code are the source of truth; GitHub Actions job logs and verified output statuses determine whether a graph was finished.

## Primary endpoints for the final topology analysis (after full runs)

1. Best feasible signed cut **within budget**, its source and refinement stage, and average signed cut over available restarts.
2. Fractional wins within each graph, separating tied scores, failure/censoring and deterministic algorithms. Avoid artificially calling repeated identical MST sources 100 independent candidate solutions.
3. Graph-normalized uplift relative to **newly rerun** original `control-paired` under the *same* experiment budget, with original Experiment021 best-of-50 shown separately as historical context.
4. Runtime including **preprocessing**, with cached MST preprocessing reported separately and clearly cost-allocated; report Pareto trade-offs rather than cherry-picking winners.
5. Stratify 30/24/17 graph-family results, signed/positive edge classes, graph size, vertex-degree CV, transitivity, cycle rank. Avoid claiming the generator's `planar` label implies graph planarity (e.g., G14 is almost-planar).
6. Repeated graph-generator families are correlated: use graph-level paired interval estimation, family-block cross-validation and holdout models. Never use 1,207 graph-method cells as 1,207 independent experimental graphs.

No strong method-selection conclusion or new approximation guarantee is justified until all complete case statuses, budgets and signed cut rescoring are verified.
