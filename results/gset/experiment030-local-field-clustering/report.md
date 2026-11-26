# Experiment030 — Local-field spherical clustering on frozen EVA embeddings

**Verified scope:** Five G-set graphs, 100 EVA seeds, two factor widths, six K-MeansNM variants; no new EVA or SDP solves.
Every baseline hard partition reproduces Experiment029 exactly. All scores use original signed graph weights.

| Graph | Rank | Method | Mean cut | Mean Δ vs Y | 95% paired CI | W/T/L | Best |
|---|---|---|---:|---:|---|---:|---:|
| G1 | default | baseline_y | 11443.20 | +0.00 | [+0.00, +0.00] | 0/100/0 | 11471 |
| G1 | default | unit_minus_h | 11443.10 | -0.10 | [-0.30, +0.00] | 0/99/1 | 11471 |
| G1 | default | raw_minus_h | 11442.77 | -0.43 | [-1.96, +0.94] | 52/6/42 | 11471 |
| G1 | default | field_strength_y | 11442.75 | -0.45 | [-2.06, +0.91] | 51/7/42 | 11471 |
| G1 | default | coherence_y | 11441.10 | -2.10 | [-4.09, -0.27] | 34/15/51 | 11463 |
| G1 | default | abs_degree_y | 11443.11 | -0.09 | [-2.56, +1.67] | 55/14/31 | 11465 |
| G1 | full | baseline_y | 11442.01 | +0.00 | [+0.00, +0.00] | 0/100/0 | 11469 |
| G1 | full | unit_minus_h | 11442.03 | +0.02 | [-0.04, +0.12] | 1/97/2 | 11469 |
| G1 | full | raw_minus_h | 11440.60 | -1.41 | [-3.56, +0.36] | 47/10/43 | 11474 |
| G1 | full | field_strength_y | 11440.52 | -1.49 | [-3.64, +0.30] | 47/10/43 | 11474 |
| G1 | full | coherence_y | 11439.64 | -2.37 | [-4.21, -0.85] | 30/23/47 | 11472 |
| G1 | full | abs_degree_y | 11443.72 | +1.71 | [+0.78, +2.67] | 55/12/33 | 11470 |
| G6 | default | baseline_y | 1967.98 | +0.00 | [+0.00, +0.00] | 0/100/0 | 2001 |
| G6 | default | unit_minus_h | 1967.93 | -0.05 | [-0.15, +0.00] | 0/99/1 | 2001 |
| G6 | default | raw_minus_h | 1967.06 | -0.92 | [-2.08, +0.19] | 43/7/50 | 1994 |
| G6 | default | field_strength_y | 1967.10 | -0.88 | [-2.06, +0.17] | 43/8/49 | 1994 |
| G6 | default | coherence_y | 1965.88 | -2.10 | [-3.22, -1.02] | 33/11/56 | 1990 |
| G6 | default | abs_degree_y | 1966.77 | -1.21 | [-2.38, -0.10] | 46/8/46 | 1994 |
| G6 | full | baseline_y | 1968.46 | +0.00 | [+0.00, +0.00] | 0/100/0 | 1992 |
| G6 | full | unit_minus_h | 1968.46 | +0.00 | [-0.21, +0.19] | 2/96/2 | 1992 |
| G6 | full | raw_minus_h | 1967.11 | -1.35 | [-3.39, +0.33] | 43/6/51 | 1994 |
| G6 | full | field_strength_y | 1967.12 | -1.34 | [-3.37, +0.39] | 43/6/51 | 1994 |
| G6 | full | coherence_y | 1968.42 | -0.04 | [-1.26, +1.17] | 40/8/52 | 1990 |
| G6 | full | abs_degree_y | 1968.50 | +0.04 | [-1.09, +1.14] | 49/9/42 | 1992 |
| G11 | default | baseline_y | 521.76 | +0.00 | [+0.00, +0.00] | 0/100/0 | 534 |
| G11 | default | unit_minus_h | 521.74 | -0.02 | [-0.12, +0.08] | 1/96/3 | 534 |
| G11 | default | raw_minus_h | 522.66 | +0.90 | [+0.18, +1.68] | 34/41/25 | 534 |
| G11 | default | field_strength_y | 522.62 | +0.86 | [+0.12, +1.68] | 33/42/25 | 534 |
| G11 | default | coherence_y | 522.62 | +0.86 | [+0.16, +1.62] | 33/42/25 | 534 |
| G11 | default | abs_degree_y | 521.76 | +0.00 | [+0.00, +0.00] | 0/100/0 | 534 |
| G11 | full | baseline_y | 521.86 | +0.00 | [+0.00, +0.00] | 0/100/0 | 536 |
| G11 | full | unit_minus_h | 521.78 | -0.08 | [-0.20, +0.02] | 1/95/4 | 536 |
| G11 | full | raw_minus_h | 522.76 | +0.90 | [+0.04, +1.76] | 42/37/21 | 536 |
| G11 | full | field_strength_y | 522.74 | +0.88 | [+0.00, +1.72] | 41/38/21 | 536 |
| G11 | full | coherence_y | 522.74 | +0.88 | [+0.02, +1.76] | 41/38/21 | 536 |
| G11 | full | abs_degree_y | 521.86 | +0.00 | [+0.00, +0.00] | 0/100/0 | 536 |
| G14 | default | baseline_y | 2970.94 | +0.00 | [+0.00, +0.00] | 0/100/0 | 2985 |
| G14 | default | unit_minus_h | 2970.91 | -0.03 | [-0.14, +0.07] | 1/97/2 | 2985 |
| G14 | default | raw_minus_h | 2968.38 | -2.56 | [-4.10, -0.96] | 28/4/68 | 2988 |
| G14 | default | field_strength_y | 2968.39 | -2.55 | [-4.08, -0.97] | 27/5/68 | 2990 |
| G14 | default | coherence_y | 2969.94 | -1.00 | [-1.97, -0.08] | 40/11/49 | 2985 |
| G14 | default | abs_degree_y | 2967.80 | -3.14 | [-5.06, -1.10] | 26/4/70 | 2995 |
| G14 | full | baseline_y | 2970.82 | +0.00 | [+0.00, +0.00] | 0/100/0 | 2984 |
| G14 | full | unit_minus_h | 2970.77 | -0.05 | [-0.13, +0.01] | 1/96/3 | 2984 |
| G14 | full | raw_minus_h | 2967.90 | -2.92 | [-4.43, -1.32] | 26/8/66 | 2988 |
| G14 | full | field_strength_y | 2967.92 | -2.90 | [-4.38, -1.31] | 25/9/66 | 2988 |
| G14 | full | coherence_y | 2970.59 | -0.23 | [-1.34, +0.85] | 42/19/39 | 2983 |
| G14 | full | abs_degree_y | 2968.38 | -2.44 | [-4.08, -0.75] | 30/6/64 | 2986 |
| G18 | default | baseline_y | 908.35 | +0.00 | [+0.00, +0.00] | 0/100/0 | 927 |
| G18 | default | unit_minus_h | 908.33 | -0.02 | [-0.05, +0.00] | 0/98/2 | 927 |
| G18 | default | raw_minus_h | 910.76 | +2.41 | [+0.79, +4.03] | 62/7/31 | 933 |
| G18 | default | field_strength_y | 910.74 | +2.39 | [+0.73, +4.08] | 62/7/31 | 933 |
| G18 | default | coherence_y | 908.13 | -0.22 | [-1.61, +1.23] | 40/6/54 | 925 |
| G18 | default | abs_degree_y | 905.79 | -2.56 | [-4.52, -0.62] | 43/4/53 | 937 |
| G18 | full | baseline_y | 905.92 | +0.00 | [+0.00, +0.00] | 0/100/0 | 926 |
| G18 | full | unit_minus_h | 905.88 | -0.04 | [-0.12, +0.00] | 0/99/1 | 926 |
| G18 | full | raw_minus_h | 907.56 | +1.64 | [+0.09, +3.25] | 63/5/32 | 933 |
| G18 | full | field_strength_y | 907.63 | +1.71 | [+0.16, +3.35] | 63/6/31 | 933 |
| G18 | full | coherence_y | 904.95 | -0.97 | [-2.03, +0.07] | 37/5/58 | 931 |
| G18 | full | abs_degree_y | 903.53 | -2.39 | [-4.48, -0.31] | 44/3/53 | 934 |

**Confidence intervals**: percentile paired bootstrap within each fixed graph and rank; these do not infer population-level performance or correct for multiple comparisons.
**Interpretation**: -H at exact EVA fixed points is D Y, so raw-field clustering is (approximately) field-strength-weighted spherical clustering, not a fundamentally different directional embedding.
Raw-field and field-weighted-Y outcomes should be interpreted alongside their observed angular deviations; unnormalized rows affect centroid weighting.
**Bias control**: absolute-weighted-degree-Y isolates a degree/weight-magnitude effect. Coherence weights are norm(H)/weighted-degree, bounded by one.
**Caveat**: The six methods were studied on the same five graphs; any winner chosen after seeing these results is exploratory.
