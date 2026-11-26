# Experiment034: paired continuation with adaptive stopping

All original E033 prefixes independently verified and reused. Both arms receive identical additional Gaussian draws; five consecutive draws without improvement by either arm stop a trajectory. A cap of 1,000 total draws protects resources. This is a heuristic plateau, not a convergence proof. Equal draws are not equal time or equal cut evaluations.

| Metric | Result |
|---|---:|
| complete | True |
| graphs | 71 |
| paired_trajectories | 1136 |
| new_eva_solves | 0 |
| reused_prefix_records | 229472 |
| new_candidate_records | 13636 |
| patience | 5 |
| max_total_draws | 1000 |
| min_extra_draws | 5 |
| median_extra_draws | 5.0 |
| mean_extra_draws | 6.001760563380282 |
| max_extra_draws | 36 |
| cap_hits | 0 |
| graph_wins | 69 |
| graph_ties | 2 |
| graph_losses | 0 |
| mean_paired_gain_pct | 1.8345071733005816 |
| graph_bootstrap_95_pct | [1.520998149183759, 2.161885111627723] |
| numerical_fallbacks | 0 |
| mean_gw_extra_gain_pct | 0.019834393339913818 |
| mean_sweep_extra_gain_pct | 0.026480785461883292 |
| mean_gw_total_gain_pct | 2.4308278776000805 |
| mean_sweep_total_gain_pct | 4.26533505090066 |
| mean_gw_extra_seconds | 0.00163089128704676 |
| mean_sweep_extra_seconds | 0.011095405458702221 |
