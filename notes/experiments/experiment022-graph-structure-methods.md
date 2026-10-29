# Experiment022 — Graph topology and initializer performance, retrospective analysis

**Status (29 October 2026):** reproducible **retrospective** exploration of the already completed Experiment021 Max-Cut (`k=2`) data. No new EVA factor, SDP solve, rounding, or local-search run was performed for this study. Initial results were computed directly from the committed 71 × 9 `per_instance_methods.csv` and a published G-set topology classification. The new Python analysis and synthetic tests were locally checked; full extraction of the original 71 graphs' higher-order structural features is provided as a command, **not claimed as run**.

## Research question

Does graph structure predict which of the **nine fixed Experiment021 initializers** reaches (a) the best observed signed Max-Cut across 50 restarts, or (b) the highest mean final score after the same four-stage Round → Flip → generalized KL → ILS pipeline?

Two distinct targets must not be conflated: a method may win more graphs' **best-of-50** contests, but be inferior in **mean-of-50**. Furthermore, `gw-screen16` spends **16 additional original-graph starting-cut evaluations per restart** and `pca-antipodal` has preprocessing overhead. These experiments compare observed cut quality, **not equal wall-clock compute budgets**.

## Sources, labels, and audit

- Primary numerical data: `results/gset/experiment021-complete-71-gset/statistics/per_instance_methods.csv` on the original graph and immutable EVA factor for each of 71 graphs; 50 matched restart IDs per initializer and fixed downstream seed streams.
- Secondary metadata: [OpenIsing's G-set README](https://github.com/KULeuven-MICAS/openising/blob/main/ising/benchmarks/G/README.md), which labels the 71 canonical graphs **random** (30), **planar** (24), and **toroidal** (17).
- **Important metadata correction:** the secondary table categorizes G48/G49/G50's weights as `+1,-1`, but direct inspection of the *canonical original* `data/G48.txt`, `G49.txt`, and `G50.txt` finds all 6,000 weights equal to **+1** on each graph. The research code records their true **positive-only** edge class, distinct from topology. G48 and G49's unweighted structure is also noted in Rodríguez-Fernández et al. (2020), page 8.
- **Visualization caution:** a circular-looking network drawing is not by itself evidence of a toroidal graph, or evidence that the G-set supplied spatial vertex coordinates. Graph drawings depend on layout; here the classification is based on the benchmark's graph-generation metadata, and structural features come from the undirected original graph.
- Literature reference cut values and method outputs are distinct; do **not** copy a published best-known cut into the empirical score columns.

## Exact top-cut comparisons (ties counted for every tied initializer)

| Family | Graphs | Most frequent best-cut initializer | Number of graphs it ties/reaches global best | Fractional wins (ties split equally) |
| --- | ---: | --- | ---: | ---: |
| random | 30 | `sample-antipodal` | **10** | **5.712** |
| planar | 24 | `gw-screen16` | **8** | **5.700** |
| toroidal | 17 | `gw-screen16` | **11** | **5.250** |

For `gw-screen16` specifically: **5/30 random**, **8/24 planar**, **11/17 toroidal** have a best-cut tie or sole win. However **G48/G49/G50** have *all nine initializers tied* in best-of-50, so their inclusion inflates the 11/17: among the **14 toroidal graphs with at least two different method maxima**, screen-16 wins or ties on **8/14**. Among signed-weight non-toroidal graphs, only **2/22** show a screen-16 best-cut tie or win. A two-sided Fisher exact calculation for these selected subsets (8/14 vs 2/22) is **p≈0.00545**, *exploratory only*: families share generators, many tests and subgroups were inspected, restarts are correlated, and additional cut evaluations are not budget-matched. This is **not a controlled causal effect** or population-level significance claim.

## Mean final-cut differences versus original paired K-MeansNM

Mean differences = unweighted mean of each graph's method mean over 50 restarts **minus the same graph's paired-method mean**. Raw cuts from different graphs are never pooled as independent restarts.

| Family | GW screen-16 | Sample antipodal | Mean perturbation | Original paired |
| --- | ---: | ---: | ---: | ---: |
| random (30) | **+0.115** | −0.374 | −0.615 | 0 |
| planar (24) | **−0.399** | −1.196 | −0.671 | 0 |
| toroidal (17) | **+0.605** | −0.518 | **+0.452** | 0 |

These are **original signed cut units**, generally tiny relative to the achieved cuts and possibly sensitive to graph composition. A high win frequency does not imply a positive mean effect on a typical run.

## Does a structural method selector generalize? First out-of-graph test

We tested a **nested train/test selection** across the 71 existing graphs: for each held-out graph, select an initializer using *only the other 70 graphs*. The training target is mean graph-normalized **best-of-50 improvement over paired** (`(best_method − best_paired)/abs(best_paired)`). Two selectors: `global`, which pools the 70 training graphs, versus `family`, which pools only the held-out graph's family in training. The held-out cut is never used to choose its method. Results remain retrospective and **not compute-budget-matched**.

| Candidate methods | Selector | Average held-out cut gain vs paired | Mean relative gain vs paired | Mean regret vs oracle best-of-nine |
| --- | --- | ---: | ---: | ---: |
| all nine | global | **+0.732** | +0.0143% | 3.986 |
| all nine | family | **+0.113** | +0.0010% | 4.606 |
| excluding GW screen-16 | global | −0.690 | −0.0227% | 4.761 |
| excluding GW screen-16 | family | −0.690 | −0.0217% | 4.761 |

**Conclusion:** the available categorical topology label has a strong descriptive association with best-cut wins, **but the family-specific selection policy did not improve held-out results** relative to a globally trained policy in this simple LOOCV test. The one-family policy also has fewer training graphs and is affected by signed-weight class and graph size. Neither policy should be deployed as an automatic selector based on these data alone.

## Reconstructed structural examples from the original graphs

The following features were computed directly from canonical, one-based `data/G*.txt` edge lists (not a visualization):

| Graph | Labeled family | Vertices | Edges | Mean degree | Degree CV | Exact triangles | Transitivity | Negative edge fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 | random | 800 | 19,176 | 47.940 | 0.1318 | 18,093 | 0.0593 | 0 |
| G14 | planar benchmark family | 800 | 4,694 | 11.735 | 0.9830 | 5,302 | 0.1535 | 0 |
| G11 | toroidal | 800 | 1,600 | 4 | 0 | 0 | 0 | 0.4894 |
| G48 | toroidal | 3,000 | 6,000 | 4 | 0 | 0 | 0 | 0 |

**Planarity label caveat:** G14 is from the benchmark's `planar`/*almost-planar* **generator family**, not a mathematically planar simple graph: `m=4,694 > 3n−6=2,394` for `n=800`. The G-set paper describes these graphs as *almost planar*. Avoid inferring a strict genus/topological embedding from the generator-family name.

## Visual evidence

![Best-cut win rates per graph family](../../results/gset/experiment022-graph-structure-methods/topology_best_cut_heatmap.svg)

![Retrospective held-out policy comparison](../../results/gset/experiment022-graph-structure-methods/topology_policy_loo.svg)

The figures are reproducible from the saved Experiment022 CSV results using:

```powershell
python scripts/plot_experiment022.py
```

This requires Matplotlib for plotting only. Graph-family percentages include **all best-cut ties**, so interpret them together with the fractional-win metric and the noninformative G48–G50 graphs.

## Graph-level explanatory variables to evaluate next

`src/maxkcut/graph_structure_methods.py` reads the canonical original `data/G*.txt` and can save:

- graph size `n`, edges `m`, edge density / average degree;
- degree coefficient of variation;
- fraction of negative original edge weights;
- number of connected components and largest component fraction;
- exact undirected triangle count and global clustering coefficient (`3 × triangles / wedges`);
- cyclomatic number `m−n+c` and normalized cycle rank.

These quantities describe the **graph itself**; they do not depend on visualization coordinates or embedder seed.

**A second, distinct geometry is the EVA vector cloud**: each vertex has a unit vector `y_i` in the saved factor `Y∈R^{n×p}`. A 2D PCA/t-SNE/UMAP scatter can create circular-looking patterns that are not actual spatial input coordinates; it must not substitute for intrinsic high-dimensional measurements. Relevant proposed features include anisotropy `||mean_i y_i||`, normalized Gram-spectrum effective rank, edge versus nonedge cosine distributions, spherical cluster separability, and antipodal pairing concentration. These are **not yet measured** for all 71 graphs. Preserve each factor's SHA256 when that extension is implemented.

For sparse toroidal/grid-like graphs, **four-cycles and signed-cycle frustration** may be more diagnostic than triangles alone: e.g. G11 and G48 are triangle-free with constant degree four. Count or sample short even cycles and record the sign product around them in a future prespecified extension. A separate future extension could study unit-vector EVA factor geometry (effective rank, concentration, edge cosine distribution). That extension must preserve original factor SHA and distinguish low-rank nonconvex EVA from full convex SDP solutions.

### Reproduce without optimization

From repo root in the existing Conda environment:

```powershell
python -m pytest tests/test_graph_structure_methods.py -q
python -m maxkcut.graph_structure_methods --skip-features
python -m maxkcut.graph_structure_methods
```

The first analysis command reproduces family comparisons and LOOCV directly from the previous Experiment021 CSV, without reading all original graph edges; the second adds canonical graph-structure features. Outputs are under `results/gset/experiment022-graph-structure-methods/`. The four retrospective statistics files and two vector graphics are already committed to this branch. The full 71-graph \`graph_features.csv\` will be generated only when the feature-extraction command is executed in a complete repository checkout; no expensive workflows are necessary. Synthetic tests do not require EVA, GPUs or saved factors.

## Research limits and next hypotheses

**Hypothesis:** toroidal sparse signed graphs might favor searching several initial hyperplanes because geometry leads to different near-best partitions; this is not established by present evidence. Test it using graph-level normalized outcomes with covariates for `n`, signed fraction, degree variability, exact clustering and cycle rank; use leave-one-generator-family/block-out validation to handle correlation between G-set graph siblings. Count tied cuts fairly, evaluate method costs, and pre-register candidate models/primary endpoints before looking at final scores.

**Crucial next experiment (independent):** compare `gw-screen16` and structured alternatives with the original method under a **matched total starting-cut evaluation and wall-clock budget** on genuinely new graph instances, to separate genuine rounding benefits from additional attempts. This should be recorded under a new sequential experiment ID, without overwriting Experiment021/022.