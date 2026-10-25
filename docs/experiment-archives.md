# Experiment archives: preserve data and avoid repeat computation

Every G-set benchmark invocation now saves compact, auditable artifacts by default. No extra SDP solve is involved.

## Data saved by the library

| Artifact | Data retained | Why |
| --- | --- | --- |
| SDP cache / G11-k2-[hash].npz | Original numerical Gram matrix and solver diagnostics | Reuse the optimization solution for new rounding seeds |
| Results JSONL | Every restart: score, iterations, timing, status, seeds and provenance IDs | Statistical analysis without new solves |
| Archive / manifest.json | Input SHA256, graph properties, solver settings, software versions, Git revision, status and method summaries | Reproduce the experiment |
| Archive / graph.npz | Normalized vertex indices and original signed edge weights | Recompute cut values independently |
| Archive / FJ.npz | Every GW/FJ partition and its cut score | Examine and independently verify all cuts |
| Archive / K-Means2N.npz | Every K-Means2N partition and score | Compare methods |
| Archive / K-MeansNM.npz | Every K-MeansNM partition and score | Compare methods |

NumPy archives are compressed NPZ files, and readers use allow_pickle=False. For k=2, partitions are packed at one bit per vertex (plus metadata); for k>2, compact unsigned integer arrays are used.

The SDP matrix is **not** duplicated per rounding restart. Instead, its original numerical output is in the fingerprinted cache. When --no-sdp-cache is requested, that solver output is not persisted for later rounding, although the graph, partitions and manifest are saved unless archiving was disabled.

Each JSONL result contains artifact_run_id, partition_archive, and partition_index, pointing to the exact archived partition.

## Windows PowerShell: first experiment

~~~powershell
git pull origin main
python -m pytest -q
maxkcut --graphs G11 --k 2 --solver SCS --eps 1e-3 --max-iters 3000 --restarts 20 --seed 2026 --output reports/gset/g11-archive-first.jsonl
~~~

This can reuse the matching raw SDP cache. The command prints an archive path that ends in G11-k2 and includes a unique run identifier.

An archive begins in status started, moves to sdp_ready after successful factorization, and reaches complete only when every method's partitions have been saved. A crash may leave an **incomplete** archive; verify status before analyzing it.

To independently rescore every saved partition without another SDP solve, use the path printed by the run:

~~~powershell
maxkcut --verify-archive "reports/gset/artifacts/YOUR-RUN-ID/G11-k2"
~~~

This verifies saved cut weights against saved signed edges; it does **not** certify global optimality, the SDP primal objective, or the identity of the local graph relative to the canonical literature dataset.

Inspect the manifest:

~~~powershell
Get-Content "reports/gset/artifacts/YOUR-RUN-ID/G11-k2/manifest.json"
~~~

A new archive ID is generated per graph/k case, so prior bundles are not overwritten. **JSONL output appends**: use a fresh results filename for each invocation.

## Repetition and scaling policy

- Default to **20 restarts**, not 100 for every G-set instance. Keep a fixed, predeclared trial count per comparison; increase only for selected follow-up experiments or a dedicated statistical study.
- Solving and factorizing the SDP is normally much more costly than a single rounding restart, but do not assume 100 restarts are free on large graphs.
- Analyze all per-restart scores, not only the best. For GW and K-MeansNM, matching random starts provide a meaningful within-graph paired comparison. K-Means2N uses independent initialization.
- Repeated restarts on one graph are **not independent graph instances**: for external validity, compare methods across different graphs and report compute budgets.
- Retain the original solver convergence status, minimum Gram eigenvalue and PSD repair diagnostics. Numerical primal scores are not certified SDP upper bounds.
- For future low-rank solvers, store low-rank factors rather than a huge dense Gram matrix. Nonconvex low-rank results are not interchangeable with exact SDP certificates.

To disable optional archive creation use --no-artifacts. To redirect the archive root use, for example, --artifacts-dir C:\\research\\gset-archives. By default archives reside beside the JSONL output, under artifacts/.

## Published optimum labels

For the canonical G11 graph, **564 has been proved optimal**: a feasible cut of 564 matches the independent toroidal planar-relaxation upper bound of 564. See [G-set proven optima and other references](gset-reference-values.md), based on Glass and Feder (2026), Table 1: https://arxiv.org/html/2606.28478

Our G11 run obtained a cut of 546, which is 96.81% of that optimum **conditional on matching the local .mat graph to the canonical input**. The archive preserves graph SHA256, signed edges and structural diagnostics to enable that identity check.

## Optional CUDA/CPU low-rank factor cache

The `maxkcut-gpu` runner uses a **distinct** cache in
`reports/gset/lowrank_cache/`. Unlike the original SDP cache (raw dense
X), it stores the best **rank-r factors** in compact float32 NPZ files
with `allow_pickle=False`, an input graph SHA256 and all optimizer settings,
CUDA runtime/Torch versions, device, GPU name, optimizer seed and history.
The per-run experiment archive also stores `lowrank_factors.npz` once,
plus the existing signed graph snapshot and **every** rounded cut partition.

This solver is a nonconvex approximation and its vector objective must not
be treated as an SDP certificate. A different `--rounding-seed` can reuse
**the same optimized factor** without GPU re-training if graph and optimizer
settings remain unchanged. See [CUDA setup and protocol](cuda-lowrank.md).
