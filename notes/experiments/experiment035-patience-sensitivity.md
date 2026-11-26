# Experiment035 — Patience sensitivity and convergence diagnostics

Prospective; no E035 claims until independent audit. Recover E034 archives from run 45815246196, artifacts 15115321660 and 15115776165. E034 failed publication due to Windows CRLF JSON manifests, not compute. It audited 13,636 new cuts, 1,136 paired trajectories; mean total gains 2.4308278776% GW and 4.2653350509% sweep.

## Design
Preserve E033/E034 binaries and source hashes; no new EVA or repeated archived rounding. Continue exact RNG state, best cuts, strict-best sweep normal and existing failure streak. Joint patience 100, 1,000-total-draw cap, retrospectively extract first hits for 5,10,20,50,100. Verify E034 prefix equality for all fields. 71 graphs, 16 matched starts, 8 graph workers, one BLAS/Numba thread each. Equal draws not equal time or cut evaluations.

## Outputs
E035 root results/gset/experiment035-patience-sensitivity/. Archive full post-100 trajectory including verbatim E034 prefix (marked reused), normals, packed labels, signed cuts, incumbent, timing, predicted arc max, validity, intervals and hashes. Independently rescore every cut. Per-seed/graph patience tables, paired graph bootstrap intervals, marginal gains, strict improvement indicators, failure streaks, conditional improvement hazard with at-risk counts, cumulative time, projective normal displacement arccos(|dot|), complement-invariant Hamming turnover from prior incumbent, minority-side fraction, sweep angular step and prediction gap. Active-only diagnostics are subject to survivor selection. Stopped convergence curves held constant. Patience policies are coupled, not independent replications. No optimality claim; report cap censoring.

## Reproduce
After restoring E034 archives, run from isolated root, PYTHONPATH=src:scripts (Windows src;scripts), one BLAS/Numba thread:
```
python scripts/experiment035_patience_sensitivity.py --action test
python scripts/experiment035_patience_sensitivity.py --phase pilot --workers 1
python scripts/experiment035_patience_sensitivity.py --phase full --workers 8
python scripts/experiment035_patience_sensitivity.py --action report
```
Graph checkpoints atomic; artifact upload even on ordinary compute failure. Use workflow_dispatch resume_run_id to restore prior E035 artifact; never repeat validated graph checkpoints. Hosted recovery commits E034 on E035 branch before new compute; normalize generated text only. Hosted report audits then archives E035. Do not modify active VM worktree or merge without authorization. PR #91 remains unchanged.
