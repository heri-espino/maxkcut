# Experiment033 — Raw random-tangent Hyperplane Sweep vs random GW

Status: protocol and implementation staged. A completed, independently verified workflow and summary.json are required before reporting numerical results.

## Question
Does the simple, random-only exact semicircle sweep improve rounding quality faster than independent random GW sampling? No guided tangents, K-means, Flip, KL, ILS, or new EVA/SDP solves.

## Protocol
- 71 canonical G-set graphs: G1–G67, G70, G72, G77, G81. Use Experiment021 inventory's original graph and exact archived factor hashes; no factor modifications.
- 16 starts per graph, matching E024/E025 initial Gaussian normals. One new Gaussian stream per graph/start using SeedSequence([2026,33,graph_number,start]).
- At step 0 both methods use the same GW normal. At steps 1–100, each receives the same Gaussian draw; GW normalizes it directly, sweep orthogonalizes it against its incumbent and optimizes the semicircle with the frozen production kernel. Record both candidates and both best-so-far scores. This comparison matches draws, NOT time or number of candidates. No claim that sweep dominates multi-start GW.
- 1,136 paired trajectories, 113,600 additional semicircle sweeps, 113,600 additional GW draws, 229,472 candidate records including duplicate common step zero. Keep actual timings and examined event interval counts; never interpret a sweep as one cut evaluation.
- Use graph-normalized improvement from the common initial cut: divide the within-seed gain by the mean initial GW cut of that graph, average across seeds and then graphs. Report per-graph results, all iterations, paired graph bootstrap (descriptive; fixed benchmark not a random population), graph wins/ties/losses and cost curves. No certified optima.
- Archive all normal vectors, packed partitions, candidate/record cuts, angle diagnostics, elapsed times, interval counts and numerical fallbacks in graphwise NPZ with SHA256 manifests. Independently reconstruct scores directly from signed text edges. Graph-level atomic checkpoints; no changes to older results or source optimizer.
- Run on PCAG103A-13 in the isolated Actions checkout, not the user's direct VM worktree. Pilot G1/G11/G14 with 2 starts × 10 steps, then full 71 × 16 × 100 with 8 graph workers, BLAS/Numba one thread each. Existing maxkcut Conda only. Hosted validation and publication. Do not merge without authorization.

## Reproduce
From root, set PYTHONPATH=src:scripts (Windows: src;scripts) and limit BLAS threads to one:
```
python scripts/experiment033_raw_hyperplane_sweep.py --action test
python scripts/experiment033_raw_hyperplane_sweep.py --action validate
python scripts/experiment033_raw_hyperplane_sweep.py --phase pilot --workers 1
python scripts/experiment033_raw_hyperplane_sweep.py --phase full --workers 8
python scripts/experiment033_report.py
```
The results directory is results/gset/experiment033-raw-hyperplane-sweep/full/. The report generator independently verifies all 229,472 signed scores before writing summary.json, report.md, per_graph.csv, iterations.csv, convergence.pdf, time_quality.pdf, graph_gains.pdf, all_71_curves.pdf and appendix.tex. The standalone appendix compiles with pdflatex and can be appended to the illustrated educational report with pdfunite. Preserve the original report rather than replacing it with an experiment-only manuscript. Do not treat an unfinished run as complete. If computation fails, recover graph archives from Actions artifacts and reuse them only when the manifest and all scores verify.

## Next agent
Read workflow logs, validate all 71 graph archives and summary coverage, check convergence/time figures and LaTeX output, and append the actual completed numerical appendix to the existing illustrated report. Never fill the appendix with synthetic benchmark figures. The existing papers/hyperplane-sweep/main.tex is a separate manuscript with guided E025 results; do not overwrite it.

## First workflow failure and correction
Run 45742813340 passed the synthetic signed-trajectory test but failed hosted validation at G1 (`EVA input graph SHA256 mismatch`), so Windows computation never started. The original Windows archive hashes distinguish CRLF from LF. The driver now follows the previously successful E032 cross-platform validation approach: accept only original, LF or CRLF-equivalent graph bytes and separately verify the exact factor bytes, case metadata, dimensions, sweep count and prepared graph/optimizer identity. Source-code hashes normalize line endings for the hosted audit. No data, factor, optimizer, seed schedule or experimental budget changed.
