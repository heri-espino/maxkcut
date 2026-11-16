# Experiment028 — Full-rank EVA paired pilot

This is a preliminary, five-graph experiment, **not** a certified SDP solve.
Both rank settings use independent seeded EVA trajectories; matched seed identifiers
do not imply identical initial embeddings across different dimensions.

Coverage: 150/150 graph × seed × rank × method rows.

| Graph | Method | Default mean | Full mean | Paired mean delta | Full better / tied / worse |
|---|---|---:|---:|---:|---:|
| G1 | FJ_EVA | 11257.000 | 11263.200 | +6.200 | 3/0/2 |
| G1 | K-Means2N_EVA | 11438.600 | 11422.800 | -15.800 | 0/1/4 |
| G1 | K-MeansNM_EVA | 11451.400 | 11446.000 | -5.400 | 2/0/3 |
| G6 | FJ_EVA | 1847.200 | 1798.800 | -48.400 | 1/0/4 |
| G6 | K-Means2N_EVA | 1954.400 | 1956.800 | +2.400 | 3/1/1 |
| G6 | K-MeansNM_EVA | 1961.200 | 1972.800 | +11.600 | 4/1/0 |
| G11 | FJ_EVA | 501.200 | 502.800 | +1.600 | 2/0/3 |
| G11 | K-Means2N_EVA | 522.800 | 514.400 | -8.400 | 1/1/3 |
| G11 | K-MeansNM_EVA | 523.600 | 524.400 | +0.800 | 3/1/1 |
| G14 | FJ_EVA | 2930.600 | 2917.600 | -13.000 | 1/0/4 |
| G14 | K-Means2N_EVA | 2962.000 | 2971.600 | +9.600 | 4/0/1 |
| G14 | K-MeansNM_EVA | 2971.800 | 2972.000 | +0.200 | 4/0/1 |
| G18 | FJ_EVA | 855.000 | 859.600 | +4.600 | 3/0/2 |
| G18 | K-Means2N_EVA | 902.600 | 904.400 | +1.800 | 2/0/3 |
| G18 | K-MeansNM_EVA | 913.000 | 901.800 | -11.200 | 1/0/4 |

## Best-of-five feasible cuts (not a paired estimator)

| Graph | Method | Best default | Best full | Best-full minus best-default |
|---|---|---:|---:|---:|
| G1 | FJ_EVA | 11339 | 11298 | -41 |
| G1 | K-Means2N_EVA | 11442 | 11437 | -5 |
| G1 | K-MeansNM_EVA | 11471 | 11464 | -7 |
| G6 | FJ_EVA | 1886 | 1874 | -12 |
| G6 | K-Means2N_EVA | 1958 | 1963 | +5 |
| G6 | K-MeansNM_EVA | 1966 | 1983 | +17 |
| G11 | FJ_EVA | 512 | 510 | -2 |
| G11 | K-Means2N_EVA | 530 | 524 | -6 |
| G11 | K-MeansNM_EVA | 532 | 536 | +4 |
| G14 | FJ_EVA | 2948 | 2927 | -21 |
| G14 | K-Means2N_EVA | 2974 | 2979 | +5 |
| G14 | K-MeansNM_EVA | 2978 | 2981 | +3 |
| G18 | FJ_EVA | 883 | 874 | -9 |
| G18 | K-Means2N_EVA | 921 | 921 | +0 |
| G18 | K-MeansNM_EVA | 922 | 908 | -14 |

## EVA objective and computational cost

| Graph | Rank | Mean continuous EVA objective | Converged / 5 | Mean EVA time (s) |
|---|---|---:|---:|---:|
| G1 | default | 12083.182473 | 5/5 | 0.12 |
| G1 | full | 12083.182556 | 5/5 | 1.56 |
| G6 | default | 2656.151301 | 5/5 | 0.13 |
| G6 | full | 2656.151217 | 5/5 | 1.78 |
| G11 | default | 629.083699 | 5/5 | 0.25 |
| G11 | full | 629.081683 | 5/5 | 2.67 |
| G14 | default | 3191.561081 | 5/5 | 0.10 |
| G14 | full | 3191.561178 | 5/5 | 1.08 |
| G18 | default | 1166.002279 | 5/5 | 0.08 |
| G18 | full | 1166.002263 | 5/5 | 1.07 |

Every objective, sampled partition and completed EVA sweep is archived by run.
The comparisons are exploratory and not a universal rank-improvement claim.
The optimizer is nonconvex and convergence does not certify SDP optimality.