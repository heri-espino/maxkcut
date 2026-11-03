# Experiment025 — Full local computational coverage milestone (2026-11-03)

**Evidence type:** User-supplied output from the university Windows 11 VM, **not** remote GitHub numerical artifacts or an independent partition audit. The terminal reported successful completion and the user then ran *a separate summary and preview* that confirmed 100% recorded coverage. This is a **completed computational sweep**, **not** a finished research publication or verified backup.

## Observed completion

Host: university Windows 11 computer associated with dedicated self-hosted runner PCAG103A-13. Exact observed working directory: C:\Users\175199\maxkcut. Conda: maxkcut. Execution mode: **direct VM PowerShell, local-only output** (not GitHub Actions, and not Git/LFS upload). Scientific optimizer fingerprint from successful VM preflight: 99c0b15ea1309f1a0931e3baba6735f79a1eb6622567d8f27e66be13a898f8ad.

Last full scheduler output showed G81 concluding all 18 method protocols, ending:

    DONE {"graph": "G81", "status": "complete", "methods": 18}
    {
      "phase": "full",
      "graphs": 71,
      "methods": 18,
      "starts_per_method": 16,
      "planned_cells": 1278,
      "completed_cells": 1278,
      "planned_independent_full_refinements": 20448,
      "all_complete": true
    }
    LOCAL CHECKPOINT: experiment025: updated full parallel coverage report (no Git add, commit or push)
    Experiment025 full completed locally. Raw checkpoints remain ONLY on this VM until backed up; none were uploaded.

The user independently re-invoked the aggregate after the scheduler exited:

    $env:PYTHONPATH = (Resolve-Path .\src).Path
    python -m maxkcut.experiment025_bestcut summary --phase full
    python .\scripts\experiment025-summary-preview.py --phase full

The summary again reported 1278 completed of 1278 planned graph-method cells and all_complete=true. The preview reported:

- **71 / 71** graphs have all 18 methods complete.
- **18 / 18** methods each have **71 / 71** fully completed graph-method cells.
- **1,136 seed checkpoints per method** = 71 × 16.
- **20,448 / 20,448 recorded finished seed checkpoints** across the study.
- No unfinished graph-method cells are admitted into the 16-start quality comparison.
- The score selected is the **highest verified feasible original signed cut ever visited** over initializer, Flip, KL and ILS progress, not necessarily the final walk state.

**Precision of this claim:** These counts are derived from the Experiment025 per-method status records, which were already checked for 16/16 coverage and validity by the execution scripts. They are not an independently executed graph-partition rescore or external proof that every raw NPZ archive survived intact. All raw checkpoints remain on the VM; no off-VM backup or GitHub upload has been verified.

## Ten representative graph winners shown in the compact preview

Only these ten graph-level results were displayed by the user. Do not extrapolate their method rankings to the other 61 graphs.

| Graph | Refined GW-RR | Best method among 18 | Best observed cut | Improvement over refined GW-RR |
| --- | ---: | --- | ---: | ---: |
| G1 | 11615 | em-hard | 11624 | +9 |
| G2 | 11599 | hyperplane-sweep | 11615 | +16 |
| G3 | 11587 | em-hard | 11622 | +35 |
| G4 | 11626 | control-paired | 11641 | +15 |
| G5 | 11602 | cutaware-kmeans | 11624 | +22 |
| G6 | 2173 | control-tabu | 2175 | +2 |
| G7 | 1987 | multigroup-k4 | 1998 | +11 |
| G8 | 1990 | em-gibbs | 1997 | +7 |
| G9 | 2037 | multigroup-k4 | 2051 | +14 |
| G10 | 1981 | control-paired | 1999 | +18 |

Because **GW-RR is included** among the 18 methods being maximized, the per-graph selected winner must be at least as good as GW-RR by construction. Positive per-graph **oracle-selected** gains alone do not establish statistical superiority of any *single fixed method*, nor do they control selection bias from 18-way multiple comparisons.

## Key local paths

Raw-data tree on the VM (not remotely published):

    C:\Users\175199\maxkcut\results\gset\experiment025-gw16-bestcut-quality\

Summary products within its analysis folder:

    analysis\full_coverage.json
    analysis\full_best_final_by_graph.csv
    analysis\full_preview.md

Actual source of the summary: src/maxkcut/experiment025_bestcut.py (summarize action) generates the coverage and 1,278-row graph-by-method CSV **from per-method statuses**. The compact preview script aggregates those artifacts, displays method counts, and selects the first ten fully covered graph winners. It does not independently rescore all saved winning partitions or compute paired seed-level significance.

The user's PowerShell 5.1 Get-Content displayed UTF-8 punctuation as mojibake (â€” and â€“). This is a console decoding issue consistent with using default encoding. Display explicitly with:

    Get-Content ".\results\gset\experiment025-gw16-bestcut-quality\analysis\full_preview.md" -Encoding UTF8

Do not modify experiment output or rerun numerical searches solely to fix terminal text rendering.

## Immediate preservation and analysis plan (no new optimizations)

1. **Protect raw data first.** No git clean/reset, branch switches, bulk add, destructive archive movement, or Git/LFS raw pushes from the active result worktree. Do not assume a single university VM is durable storage. Arrange an *authorized, separate, persistent off-VM destination*. When all writers have stopped, copy the complete raw tree, independently verify file counts and SHA256 manifests, record backup timestamp and approved retrieval mechanism. Never publish private institutional paths or credentials.
2. **Freeze provenance.** Record timestamp, VM/Conda environment, optimizer SHA, journal format, methods/seeds/graphs, complete status CSV and JSON, source and factor fingerprints, full logs and replay instructions. Compare the final local script against remote main / PR #70 before any future merge, since the VM may retain working local-only patches.
3. **Independent graph objective audit.** Reopen saved seed partitions (without re-optimization), verify SHA256/identity/factor and original signed graph cut score. Audit *all* 20,448 if feasible, at minimum all 1,278 graph-method winners and a representative sample of nonwinners; record exact audited coverage and failures. Do not call the study independently verified until this is actually performed.
4. **Paired comparative statistics.** Compare each method with refined GW-RR on identical graph and GW seed identities. Across 71 graphs, report per-graph 16-start best, mean, dispersion and relative gains; across method/graph matched seeds, report differences and uncertainty, family and signed-weight strata, rank/win rates with tie handling, fixed-method versus oracle winner, multiplicity considerations, and sensitivity to outliers. Methods are *not equal-cost*, even with common seed directions. Avoid calling an EVA factor a certified SDP optimum, or published best-known feasible cuts proven optima.
5. **Compact public artifact.** After independent backup/audit, publish only full_coverage.json, full_best_final_by_graph.csv, full_preview.md, method-level analyses and figures in a *new clean Git worktree/branch off main*, with a PR. Do not push the local results branch, since past unpublished commits could carry large raw LFS data.
6. **Paper/report.** The full comparison and methodological limitations should be written from verified numerical results, with reproducibility instructions, a stable manifest for the off-VM raw archive, and dated provenance; no additional EVA solve is needed.

## Other relevant records

- [Canonical university VM and Actions decision runbook](../../docs/university-vm-experiment025-operations.md).
- [G44/G56/G54/G67 Windows I/O checkpoint incidents](experiment025-windows-local-recovery-2026-11-03.md).
- [Experiment025 scientific protocol](experiment025-gw16-bestcut-quality.md).
- [Last successful Actions pilot; not the full run](experiment025-successful-pilot-parallel-vm.md).
- [Optional local-only implementation and guarded scheduler PR #70](https://github.com/heri-espino/maxkcut/pull/70).

**Next-agent status sentence:** "Experiment025 full local computational coverage was confirmed twice by the user's Windows VM (71×18×16 = 20,448); no more full numerical refinements are needed. Independent rescore, verified off-VM backup, paired inference and compact GitHub publication remain pending. GitHub remote may still be missing all locally saved raw results."
