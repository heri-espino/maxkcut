# Experiment034 — Adaptive continuation of Experiment033

Status: prospective; no new numerical claims until all 71 graphs pass the independent audit.

## Protocol
- Parent: E033 archived commit c593720c5253d739dc0ad642052ee825c2b1b3d0 (PR #90). Reuse its 71 immutable E021 EVA factors, 16 paired starts per graph, all 100 Gaussian proposals and exact packed partitions. Never solve EVA or repeat the old rounding search. Verify the old archive and advance its RNG past the first 100 draws; recover the first strict-best sweep normal.
- For each graph/start, continue BOTH methods with the same Gaussian proposals: independent GW versus random-tangent exact semicircle sweep. Maintain the original signed objective and existing kernel. Stop at the first 5 consecutive **additional** draws without improvement by either arm (absolute tolerance 1e-8), resetting on improvement by either. Both have the same number of draws. Safety cap: 1,000 total draws; report cap hits separately. No Flip, KL, ILS or clustering. Five misses are not a proof of stationarity, especially for independent GW. No equal-time claim.
- Pilot G1/G11/G14, first two archived starts; full 71 x 16. Default eight graph processes, BLAS/Numba one thread. Isolated Windows Actions checkout, PCAG103A-13 identity and competing-process guard. Hosted test and report. Never touch the separate university research worktree.
- Save every new normal, packed partition, original score, incumbent, timing, interval count, angle and validity in ragged graph-level NPZ; source and input hashes in manifest. Existing verified graph checkpoints reused. Independently rescore all new candidates from original signed text edges. Record stop reasons, extra gain vs iteration 100, total gain from initial GW (same graph normalization as E033), graph bootstrap, runtime and stopping distribution. Curves hold stopped trajectories constant; no extrapolation.
- Publish exclusively to the new branch; E033 remains unchanged. Do not merge without authorization.

## Reproduce
From isolated checkout of this branch, existing maxkcut conda environment, PYTHONPATH=src;scripts (Windows) or src:scripts (Linux); set BLAS/Numba threads to 1:
```
python scripts/experiment034_adaptive_hyperplane_sweep.py --action test
python scripts/experiment034_adaptive_hyperplane_sweep.py --phase pilot --workers 1
python scripts/experiment034_adaptive_hyperplane_sweep.py --phase full --workers 8
python scripts/experiment034_adaptive_hyperplane_sweep.py --action report
```
Results: results/gset/experiment034-adaptive-hyperplane-sweep/full/; summary.json, per_graph.csv, per_seed.csv, iterations.csv, report.md, convergence.pdf, stopping_distribution.pdf, all_71_curves.pdf and graph archives. LF CSV generation avoids E033 publication error. If numerical workflow fails, recover uploaded checkpoints instead of recomputing completed searches. Check complete audit and report before interpretation.
