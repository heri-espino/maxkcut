# G-set literature references for Experiment026

The versioned gset-literature-2026-11.csv table records 71 G-set targets with graph size, cut score, selected source URL, source classification, and (only where available) a published upper bound. A reported best-known feasible cut is NOT an upper bound or automatically a proven optimum.

The canonical 71-row base is the OpenIsing G-set benchmark compilation:
https://github.com/KULeuven-MICAS/openising/blob/5a22c619cbdaef06637c1a4a43c0d2c96fc3fa05/ising/benchmarks/G/README.md

All 71 graph vertex/edge counts were cross-checked against:
https://github.com/Quicopt/Benchmarks/blob/4ef288bf9f7cc697ed867df2da403eade45a9c14/docs/gset.md

When they exceed the base, Quicopt's documented feasible solver scores or attributed best-known cuts are included. Additional higher records were checked against Benlic and Hao's Breakout Local Search tables, published Toshiba comparisons, Ma and Hao reports, Khan and Shukla's G63 value of 27047 (2025), and Glass–Feder's 2026 toroidal results. The chosen source for each final row is preserved in source_id and source_url. This is a best-of-sources-collected snapshot rather than a guarantee of comprehensive, current best known across the entire literature.

The published_upper_bound column is populated only for the 17 documented toroidal cases in Glass and Feder (2026), arXiv:2606.28478, Table 1. conditional_proven_optimum means the *published feasible cut* equals the published upper bound, conditional on the canonical graph data being identical.

IMPORTANT: Historical Experiment013–025 references intentionally remain unchanged to preserve experiment provenance and the optimizer-source hash. Their earlier internal reference_cut is NOT a world-record comparator. The dedicated Experiment026 script reads this new versioned CSV for forward-looking comparisons.

The eight former internal improvements are not eight literature records: G1 and G44 match existing published feasible cuts; G9, G28, G36, G40, G42 and G53 trail stronger documented values.

Candidate records require independent original signed-cut partition rescore, original graph SHA/identity checks, an updated literature check, and independent reproduction before public claims.
