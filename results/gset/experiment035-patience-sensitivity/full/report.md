# Experiment035: paired patience sensitivity and convergence diagnostics

[Download the numerical appendix (PDF)](Experiment035_Numerical_Appendix.pdf) for all figures and their explanations.

All E033 and E034 prefixes independently verified and reused. Both arms receive identical additional Gaussian draws; 100 consecutive draws without improvement by either arm stop a trajectory. Smaller patience policies are exact retrospective prefixes. A cap of 1,000 total draws protects resources. This is a heuristic plateau, not a convergence proof. Equal draws are not equal time or equal cut evaluations.

| Metric | Result |
|---|---:|
| complete | True |
| graphs | 71 |
| paired_trajectories | 1136 |
| new_eva_solves | 0 |
| reused_prefix_records | 229472 |
| reused_e34_records | 13636 |
| new_candidate_records | 422348 |
| patience | 100 |
| max_total_draws | 1000 |
| min_extra_draws | 100 |
| median_extra_draws | 170.0 |
| mean_extra_draws | 191.8943661971831 |
| max_extra_draws | 597 |
| cap_hits | 0 |
| graph_wins | 69 |
| graph_ties | 2 |
| graph_losses | 0 |
| mean_paired_gain_pct | 1.7995930780536666 |
| graph_bootstrap_95_pct | [1.492219526684813, 2.118790159538598] |
| numerical_fallbacks | 0 |
| mean_gw_extra_gain_pct | 0.285564657991193 |
| mean_sweep_extra_gain_pct | 0.2572969548662484 |
| mean_gw_total_gain_pct | 2.6965581422513583 |
| mean_sweep_total_gain_pct | 4.496151220305026 |
| mean_gw_extra_seconds | 0.04664504335673434 |
| mean_sweep_extra_seconds | 0.3036163807254281 |
