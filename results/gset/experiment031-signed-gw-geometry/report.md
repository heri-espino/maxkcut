# Experiment031 — Signed-edge geometry and GW projection diagnostics

**Frozen factors only:** 1,000 independently archived EVA embeddings, 16 matched Gaussian hyperplanes per embedding, 16,000 paired GW comparisons; no EVA or SDP optimization.

| Graph | p | Positive-edge mean angle | Negative-edge mean angle | Analytic expected cut | Empirical GW cut | -H vs Y mean Δ | Field-label mismatch | Negative-edge field ablation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 | 41 | 105.79° | n/a | 11270.25 | 11269.95 | +0.018 | 0.0055% | 0.00% |
| G1 | 800 | 105.79° | n/a | 11270.24 | 11267.96 | +0.004 | 0.0053% | 0.00% |
| G6 | 41 | 106.39° | 73.64° | 1821.40 | 1820.85 | +0.024 | 0.0039% | 14.90% |
| G6 | 800 | 106.39° | 73.64° | 1821.39 | 1820.96 | -0.013 | 0.0045% | 14.93% |
| G11 | 41 | 144.64° | 35.69° | 501.24 | 501.46 | +0.006 | 0.0091% | 11.21% |
| G11 | 800 | 144.64° | 35.69° | 501.22 | 501.04 | -0.006 | 0.0094% | 11.24% |
| G14 | 41 | 112.35° | n/a | 2929.89 | 2930.48 | -0.015 | 0.0052% | 0.00% |
| G14 | 800 | 112.35° | n/a | 2929.89 | 2930.04 | +0.013 | 0.0051% | 0.00% |
| G18 | 41 | 120.27° | 57.76° | 846.70 | 846.74 | -0.006 | 0.0037% | 15.90% |
| G18 | 800 | 120.27° | 57.76° | 846.70 | 846.87 | -0.004 | 0.0048% | 15.88% |

The exact angular GW expectation is Σ_(i,j) w_ij arccos(y_i·y_j)/π, including negative weights with their original signs.
The negative-edge field ablation removes W_- from the projected field **at frozen Y**; it does not solve or reoptimize a positive-only graph.
EVA fixed-point alignment predicts that -H and Y produce the same GW signs except near hyperplane boundaries at finite convergence tolerance.
The 16 hyperplanes share each underlying factor; uncertainty must be resampled over the 100 EVA seeds, not over 1,600 independent observations.
All conclusions describe these five graphs. Positive-only G1/G14 are controls. No approximation guarantee or global SDP certificate follows.

**Coverage:** 1000 source factors, 16000 matched normals, 48000 independently re-scorable signed hard cuts, with full bit-packed labels and SHA-256 manifests.
