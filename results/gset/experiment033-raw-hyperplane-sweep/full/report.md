# Experiment033: raw hyperplane rounding

71 frozen EVA factors; 16 paired starts per graph; 100 random-tangent sweeps or 100 fresh GW draws; no Flip, KL, ILS, K-means or EVA solves.

- Mean gain relative to graph-specific initial GW: repeated GW 2.4110%, sweep 4.2389%.
- Equal-graph mean final sweep-minus-GW difference: +1.8279 percentage points.
- Descriptive graph-bootstrap 95% interval: [1.5079, 2.1602] percentage points.
- Graph wins/ties/losses: 69/2/0.
- Independently rescored candidate records: 229472.
- Numerical fallback count: 0.

The curves average seed-level incumbent gains normalized by each graph's positive mean initial GW cut, then give equal weight to each graph. The time plot uses mean cumulative measured time, not a matched-time experiment. Timing excludes setup, warmup, RNG and external verification. One sweep explores many event intervals; equal draws do not imply equal work. No certified optimum or improved approximation ratio is claimed.

See convergence.pdf, time_quality.pdf, graph_gains.pdf, all_71_curves.pdf, per_graph.csv and iterations.csv. The exact original signed partitions, normals, timestamps and source hashes are retained in graphs/.
