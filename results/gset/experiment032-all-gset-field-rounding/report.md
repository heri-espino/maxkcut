# Experiment032 — All 71 G-set graphs: GW and K-MeansNM on Y versus -WY

**Completed coverage:** 71 original EVA factors, 50 matched normals/factor, 4 rounding arms, 14,200 independently verified original signed cuts. No EVA, SDP, Flip, KL or ILS.

All paired differences are **raw -H minus Y** at fixed source factor and shared initial Gaussian normal. The first K-MeansNM assignment is the corresponding GW partition (up to label complement); K-MeansNM retains the best signed cut along its own Lloyd path.

## All-graph mean percentage effects (graphs equally weighted)

| Algorithm | All 71 | Positive-weight | Signed-weight |
|---|---:|---:|---:|
| GW | +0.00020% | +0.00016% | +0.00024% |
| K-MeansNM | +0.02396% | +0.00153% | +0.04576% |

## Per-graph original signed cut comparisons

| Graph | Family | Sign | GW mean Δ | KM mean Δ | KM best Δ | KM 95% paired CI |
|---|---|---|---:|---:|---:|---|
| G1 | random | positive | +0.08 | +0.06 | -1 | [-1.04, +1.26] |
| G2 | random | positive | -0.14 | +2.76 | +6 | [-0.82, +6.60] |
| G3 | random | positive | +0.04 | +1.20 | -2 | [-0.62, +3.00] |
| G4 | random | positive | +0.00 | -4.78 | -17 | [-7.12, -2.36] |
| G5 | random | positive | -0.04 | -2.46 | -2 | [-4.78, -0.56] |
| G6 | random | signed | +0.02 | -2.26 | -9 | [-4.20, -0.42] |
| G7 | random | signed | -0.10 | +1.98 | +10 | [-1.42, +5.68] |
| G8 | random | signed | +0.04 | -6.40 | +8 | [-10.68, -2.34] |
| G9 | random | signed | +0.00 | -2.58 | -7 | [-5.52, -0.44] |
| G10 | random | signed | -0.04 | +3.04 | +6 | [-0.28, +6.54] |
| G11 | toroidal | signed | -0.04 | +2.16 | +0 | [+0.84, +3.64] |
| G12 | toroidal | signed | -0.04 | -0.48 | -2 | [-0.88, -0.12] |
| G13 | toroidal | signed | +0.00 | +0.36 | +0 | [-0.32, +1.08] |
| G14 | planar | positive | +0.02 | -3.38 | +0 | [-5.12, -1.58] |
| G15 | planar | positive | +0.00 | +0.64 | -3 | [-0.00, +1.22] |
| G16 | planar | positive | -0.08 | +1.94 | +6 | [-0.36, +4.08] |
| G17 | planar | positive | +0.00 | +0.06 | +0 | [-1.94, +2.44] |
| G18 | planar | signed | +0.06 | +2.94 | +3 | [+0.76, +5.24] |
| G19 | planar | signed | -0.02 | +3.62 | -5 | [+1.96, +5.18] |
| G20 | planar | signed | -0.04 | -1.58 | +8 | [-2.72, -0.32] |
| G21 | planar | signed | +0.08 | +1.64 | -1 | [-1.42, +4.86] |
| G22 | random | positive | +0.08 | +2.70 | +7 | [+0.78, +4.96] |
| G23 | random | positive | +0.02 | -0.46 | +4 | [-4.28, +2.34] |
| G24 | random | positive | +0.14 | +3.96 | +7 | [+1.92, +6.12] |
| G25 | random | positive | -0.20 | +1.24 | -2 | [-1.30, +4.02] |
| G26 | random | positive | -0.08 | -3.24 | -1 | [-5.86, -0.26] |
| G27 | random | signed | -0.14 | -6.94 | -12 | [-9.90, -4.04] |
| G28 | random | signed | -0.04 | +0.38 | +2 | [-2.08, +2.94] |
| G29 | random | signed | -0.02 | -3.52 | -6 | [-5.64, -1.26] |
| G30 | random | signed | -0.04 | -3.16 | -1 | [-6.12, -0.48] |
| G31 | random | signed | +0.20 | +0.72 | +33 | [-4.54, +5.40] |
| G32 | toroidal | signed | +0.08 | +0.96 | +2 | [+0.24, +1.68] |
| G33 | toroidal | signed | +0.00 | +0.76 | +4 | [-0.44, +2.08] |
| G34 | toroidal | signed | +0.08 | +0.36 | +0 | [-0.52, +1.24] |
| G35 | planar | positive | +0.14 | -2.52 | +18 | [-7.38, +2.04] |
| G36 | planar | positive | +0.00 | +2.00 | +11 | [+0.60, +3.62] |
| G37 | planar | positive | +0.02 | +8.46 | +7 | [+6.04, +10.96] |
| G38 | planar | positive | +0.06 | +1.24 | +1 | [-1.48, +3.94] |
| G39 | planar | signed | +0.02 | +4.96 | -4 | [-2.54, +12.34] |
| G40 | planar | signed | +0.00 | -4.50 | -7 | [-12.46, +2.68] |
| G41 | planar | signed | +0.06 | -2.42 | -7 | [-5.58, +0.80] |
| G42 | planar | signed | +0.02 | +8.00 | +8 | [+5.02, +10.92] |
| G43 | random | positive | +0.00 | -0.86 | +0 | [-3.20, +1.24] |
| G44 | random | positive | +0.04 | +3.26 | -7 | [-0.88, +7.02] |
| G45 | random | positive | +0.00 | -3.26 | -1 | [-5.26, -1.34] |
| G46 | random | positive | +0.10 | -3.64 | -10 | [-6.06, -1.30] |
| G47 | random | positive | +0.14 | -0.04 | +2 | [-1.84, +1.84] |
| G48 | toroidal | positive | +0.00 | +0.00 | +0 | [+0.00, +0.00] |
| G49 | toroidal | positive | +0.04 | +0.00 | +0 | [+0.00, +0.00] |
| G50 | toroidal | positive | +0.08 | +0.00 | +0 | [+0.00, +0.00] |
| G51 | planar | positive | -0.02 | +1.32 | +0 | [-0.34, +2.94] |
| G52 | planar | positive | +0.08 | +1.42 | +4 | [-0.02, +2.82] |
| G53 | planar | positive | +0.00 | -0.84 | -8 | [-3.10, +1.20] |
| G54 | planar | positive | +0.02 | -1.52 | -1 | [-3.30, +0.24] |
| G55 | random | positive | -0.10 | +2.04 | +4 | [-0.96, +5.02] |
| G56 | random | signed | +0.06 | +9.64 | +1 | [+6.10, +13.08] |
| G57 | toroidal | signed | +0.04 | +0.24 | +2 | [-1.16, +1.64] |
| G58 | planar | positive | -0.18 | -2.38 | +15 | [-8.50, +3.10] |
| G59 | planar | signed | -0.06 | +8.76 | -8 | [+4.16, +13.52] |
| G60 | random | positive | +0.06 | -3.90 | -7 | [-8.26, +0.68] |
| G61 | random | signed | +0.10 | -4.36 | +3 | [-9.02, +0.16] |
| G62 | toroidal | signed | -0.16 | +4.68 | -2 | [+2.52, +6.80] |
| G63 | planar | positive | -0.16 | +9.18 | +6 | [+2.84, +14.00] |
| G64 | planar | signed | +0.18 | +13.16 | +24 | [+7.78, +18.62] |
| G65 | toroidal | signed | +0.08 | +1.76 | -2 | [+0.24, +3.24] |
| G66 | toroidal | signed | -0.04 | +2.44 | +2 | [+0.52, +4.44] |
| G67 | toroidal | signed | -0.20 | +2.44 | +24 | [+0.08, +4.96] |
| G70 | random | positive | -0.10 | -0.32 | +9 | [-1.32, +0.76] |
| G72 | toroidal | signed | -0.04 | +1.44 | +4 | [-0.04, +2.92] |
| G77 | toroidal | signed | +0.24 | +1.72 | -28 | [-1.96, +5.00] |
| G81 | toroidal | signed | +0.16 | +1.32 | -8 | [-0.68, +3.16] |

**Interpretation limitations:** One historical EVA factor is frozen for each graph; 50 starts quantify rounding stochasticity, not independent EVA initialization uncertainty. Graph families are correlated; graphwise CIs condition on each fixed factor and are not multiplicity-adjusted. Compare equal-graph-relative percentages, not pooled signed cut weights. No positive-weight-only reoptimization or GW approximation guarantee for signed weights.

**Archive:** `all_runs.csv`, `per_graph.csv`, `summary.json`, and all 71 `graphs/G*/{cases.csv,partitions.npz,traces.jsonl,manifest.json}` with immutable factor SHA identities and signed-cut independent rescoring.
