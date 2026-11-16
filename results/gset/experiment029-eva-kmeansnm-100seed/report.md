# Experiment029 — Antipodal K-MeansNM with 100 independent EVA seeds

All hard cuts are rescored on original signed G-set edges. Each seed provides
one independent EVA initialization and one K-MeansNM clustering run per rank.
The first five seeds reuse Experiment028 factor/partition archives without recomputation.
Internal EVA convergence is not a certificate of SDP global optimality.

Coverage: 1000/1000 score records; 500/500 paired comparisons.

| Graph | Mean p=41 | Mean p=800 | Delta | 95% paired bootstrap CI | Wins/ties/losses | Best p=41 | Best p=800 |
|---|---:|---:|---:|---:|---:|---:|---:|
| G1 | 11443.20 | 11442.01 | -1.19 | [-4.48, +2.17] | 49/3/48 | 11471 | 11469 |
| G6 | 1967.98 | 1968.46 | +0.48 | [-1.73, +2.94] | 48/7/45 | 2001 | 1992 |
| G11 | 521.76 | 521.86 | +0.10 | [-1.80, +1.98] | 43/10/47 | 534 | 536 |
| G14 | 2970.94 | 2970.82 | -0.12 | [-2.68, +2.40] | 49/4/47 | 2985 | 2984 |
| G18 | 908.35 | 905.92 | -2.43 | [-5.17, +0.35] | 40/6/54 | 927 | 926 |

Equal-graph relative mean change: **-0.0477%**.

## EVA objective and runtime

| Graph | Objective p=41 | Objective p=800 | EVA time p=41 (s) | EVA time p=800 (s) | Ratio |
|---|---:|---:|---:|---:|---:|
| G1 | 12083.1828 | 12083.1826 | 0.117 | 1.570 | 13.5× |
| G6 | 2656.1512 | 2656.1512 | 0.130 | 1.808 | 13.9× |
| G11 | 629.0829 | 629.0804 | 0.238 | 2.644 | 11.1× |
| G14 | 3191.5612 | 3191.5612 | 0.096 | 1.112 | 11.6× |
| G18 | 1166.0024 | 1166.0023 | 0.088 | 1.090 | 12.4× |

**Uncertainty:** Confidence intervals resample 100 paired seed identifiers **within each graph**, conditioning on the five selected graphs. They do not quantify between-graph variation or the significance of a population-level rank advantage.
**Cost caveat:** Runtime excludes compilation overhead and artifact/IO time; branch and environment comparisons require identical hardware.
**Factor width is not effective numerical rank.** No SDP dual certificate is inferred.
