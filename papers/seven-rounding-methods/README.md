# Seven leading Max-Cut rounding protocols — modular LaTeX manuscript

**Status:** working research manuscript, 10 November 2026. The editable paper is in **main.tex**, with seven method-specific source files in **sections/**, mathematical appendices, tables, reproducible source TikZ figures in **figures/**, and a BibTeX bibliography. This is a new dedicated branch based on the existing Experiment027 research branch; do not merge without authorization.

## The precise top-seven selection

The ranking is the **71-graph equally weighted mean relative gain of the best-of-16 final best-ever cut over refined GW**, for 18 historical Experiment025 and three new Experiment027 protocols. These are **post-Flip/KL/ILS** protocol rankings, not raw-only hyperplane rankings or approximation guarantees.

| Rank | Method | Gain (%) | Primary evidence |
|---:|---|---:|---|
| 1 | Hyperplane Sweep | 0.317618 | Provisional Experiment025 aggregate |
| 2 | Angular Gain Greedy | 0.316704 | Independently audited Experiment027 |
| 3 | Angular Margin Greedy | 0.302682 | Independently audited Experiment027 |
| 4 | Angular Pair Greedy | 0.281426 | Independently audited Experiment027 |
| 5 | Paired K-Means Control | 0.271877 | Provisional Experiment025 aggregate |
| 6 | EM–Gibbs | 0.270688 | Provisional Experiment025 aggregate |
| 7 | Cut-aware K-Means | 0.266807 | Provisional Experiment025 aggregate |

## Source layout

The top-level **main.tex** includes, in order: **01_problem_and_protocol.tex**, **hyperplane_sweep.tex**, **angular_gain.tex**, **angular_margin.tex**, **angular_pair.tex**, **control_paired.tex**, **em_gibbs.tex**, **cutaware_kmeans.tex**, **09_empirical_results.tex**, **10_discussion.tex**, **appendix_derivations.tex**, and **appendix_reproducibility.tex**.

Each of the seven method chapters explains the name, input, signed graph objective, geometry or stochastic mechanism, equations, source-grounded pseudocode, reported numerical findings and limitations. Figures are editable source TeX, not raster-only output: **rounding_pipeline.tex**, **ranking.tex**, **arc_events.tex**, **raw_gap.tex**. **data/top_seven.csv** is a compact derived table, not a replacement for primary input CSV files.

## Reproducibility

From repository root:

~~~bash
python papers/seven-rounding-methods/verify_results.py
cd papers/seven-rounding-methods
latexmk -pdf -file-line-error -halt-on-error -interaction=nonstopmode main.tex
~~~

The verifier uses only Python standard library, requires all 1,278 completed Experiment025 graph-method rows, verifies Experiment027's 3,408 new + 2,272 historical comparator archives and all 6,816 paired delta rows, and matches every selected table value to its original source report. The dedicated **seven-rounding-methods-manuscript.yml** hosted GitHub Actions workflow produces an inspectable PDF and LaTeX log; artifacts have finite retention, while editable source stays in Git. **No PowerPoint. No EVA/SDP/optimizer or numerical reruns.**

Original E025 ranking:
results/gset/experiment025-gw16-bestcut-quality/analysis/experiment025_method_ranking.csv.
Original E027 audit:
published/experiment027/releases/experiment027-data-71g-2b313a624780291eb4dd/analysis/.

**Provenance caveat:** E025's full 18-method raw NPZ/JSONL history is not completely independently verified off the university VM, despite its completed aggregate. E027 independently audited the three new methods plus historical GW and Hyperplane Sweep, not all 18 E025 methods. Matching 16 starts does not equalize internal candidate evaluation or CPU time. Raw-only scores and post-refinement scores address different questions.

## Scientific boundaries

Do not call the largest descriptive mean statistically proven best. Do not call a fixed-arc or fixed-normal exact search a globally exact Max-Cut algorithm. Do not infer classical GW approximation guarantees for the signed graphs and approximate EVA factors. EM–Gibbs updates binary labels directly and is not pure hyperplane rounding. The proposed future rotation-and-translation study with **no KL/ILS** remains unexecuted. Preserve old paper/deck versions and experiment data.

See **AGENTS.md** and **HANDOFF.md** for agent guidance and next steps.
