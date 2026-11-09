> **NEW PROSPECTIVE RESEARCH (9 November 2026):** The user proposes to isolate the geometry of GW hyperplane rounding by optimizing **only rotations/translations**, with **no Flip/KL/ILS/Tabu**. This is not a retrospective claim about E027's refined scores. See the separate [Experiment028 theory and experimental-protocol proposal](experiment028-hyperplane-geometry-proposal.md), which explicitly acknowledges existing threshold and arc sweeps and literature prior art. No new numerical results yet.

# Experiment027 — complete published 71-graph results and first research interpretation

**Date confirmed:** 2026-11-08. **Status:** Full dataset remotely indexed and independently scored by the VM publication pipeline. The original raw numerical archives remain on the Windows VM, and the complete **new-method** raw iteration histories were uploaded as 71 content-addressed GitHub Release ZIP assets with server-SHA256 verification according to the user's successful VM output. **Do not repeat EVA/SDP or any Experiment025/027 numerical method.** This is a publication milestone, not a global Max-Cut optimality or statistical superiority certificate.

## Durable, exact locations for the next agent

- **Published raw release:** https://github.com/heri-espino/maxkcut/releases/tag/experiment027-data-71g-2b313a624780291eb4dd
- **Stable pointer:** `published/experiment027/latest_release.json`
- **Complete content-addressed index:** `published/experiment027/releases/experiment027-data-71g-2b313a624780291eb4dd/index.json`
- **Per-graph ZIP SHA and every file's SHA:** `published/experiment027/releases/experiment027-data-71g-2b313a624780291eb4dd/manifests/Gxx-<sourcehash>.json`
- **Five directly readable, small analysis products:** `published/experiment027/releases/experiment027-data-71g-2b313a624780291eb4dd/analysis/`
  - `audited_comparison.json` (counts, leaderboard, intervals)
  - `audited_method_leaderboard.csv`
  - `audited_graph_comparisons.csv`
  - `audited_paired_seed_deltas.csv`
  - `audited_seed_scores.csv`
- **Source code:** `scripts/experiment027_angular.py` (FROZEN, content-hashed), `scripts/experiment027_from_archives.py`, `scripts/experiment027_analyze.py`, `scripts/experiment027_publish_release.py`; read `notes/experiments/experiment027-angular-greedy.md` and `docs/experiment027-release-no-lfs.md`.
- **Historical Experiment025 data:** original VM checkout `C:\Users\175199\maxkcut\results\gset\experiment025-gw16-bestcut-quality` (never reset, clean, overwrite or rerun); full 18-method raw histories are **not** included in the new-method Release. Corresponding GW and Hyperplane Sweep comparator scores were independently rescored during Experiment027's audit.
- **VM's local Experiment027 worktree:** `C:\Users\175199\maxkcut-exp027`, local branch `exp027-local`. This is **not** the same worktree as the historical Experiment025 checkout. Never rename, switch, reset or delete it just for publication.

**Remote verification performed in the GitHub connector:** The stable pointer exists on the remote `experiment027-angular-greedy` branch, references the exact Release tag, and the complete index contains 71 SHA-256-tagged assets. The independent audit report contains `status=independently_verified_archives`, 71 graphs, 5,680 rechecked seed archives (3,408 new + 2,272 historical comparator seeds), 213 new graph-method cells and 16 matched GW starting normals per method. The indexed 71 ZIPs total **148,929,738 bytes** compressed and inventory **109,159** source members. **The assistant has not itself downloaded/re-hashed all 71 ZIPs from a second machine**; the VM uploader reported checking the GitHub server-computed asset digests. Preserve the VM raw folders until an independent off-VM read-back.

## Primary outcome — best-of-16 after common Flip/KL/ILS refinement

For graph `g`, let `B_m(g)` be the largest verified original signed cut score obtained across the 16 paired seeds for method `m`; `GW` denotes Experiment025 `gw-rr` (refined, not raw initial hyperplane); `HP` denotes Experiment025 `hyperplane-sweep`. The equal-graph, mean relative gain reported in `audited_method_leaderboard.csv` is

```text
D_m = (1/71) * sum_g 100 * (B_m(g) - B_GW(g))/abs(B_GW(g)).
```

Its 95% bootstrap interval resamples **graphs**, not 1,136 correlated seeds, with the archived audit's fixed 2,000 repetitions. Note the intervals assess mean gain over GW, **not** superiority over historical Hyperplane Sweep.

| New method | Equal-graph mean improvement vs refined GW | 95% graph-bootstrap interval | wins / ties / losses against GW | wins / ties / losses against HP | Mean best-of-16 cut difference vs HP |
|---|---:|---:|---:|---:|---:|
| Angular Gain Greedy | **+0.31670%** | [+0.23336%, +0.40329%] | 61 / 8 / 2 | 30 / 10 / 31 | −0.3099 |
| Angular Margin Greedy | +0.30268% | [+0.21323%, +0.40024%] | 56 / 9 / 6 | 29 / 14 / 28 | −1.0000 |
| Angular Pair Greedy | +0.28143% | [+0.19896%, +0.36784%] | 56 / 7 / 8 | 28 / 9 / 34 | −1.0845 |

**Conclusion supported by saved data:** All three new methods outperform the refined GW baseline on most of the 71 graphs, but **none establishes consistent improved best-of-16 quality over the existing Hyperplane Sweep comparator**. Angular Gain Greedy has the strongest mean GW gain among these three, but **30 graph wins versus 31 graph losses** against HP.

The archived `audited_graph_comparisons.csv` supports a **descriptive, post-hoc portfolio**: selecting the best of the three new angular methods per graph (after running all three) beats HP on **42 graphs**, ties on **13**, loses on **16**; average best-of-three minus HP is **+2.76056 cut units**. The largest improvement against HP is G23, **13322 vs 13303 (+19)**. This is **not a fair same-cost single-method comparison**: it chooses from three new algorithms, with potentially threefold (or otherwise unequal) search cost. It is not proof of better expected performance on new graphs or of a best-known Max-Cut record.

A direct graph-bootstrap comparison versus HP (post-publication descriptive computation, 20,000 graph-cluster bootstrap replicates with a fixed arithmetic PRNG, not the published auditor's GW intervals) gave approximate 95% ranges in **cut units**:
- Angular Gain Greedy: mean −0.310; CI approximately [−2.07, +1.42].
- Angular Margin Greedy: mean −1.000; CI approximately [−2.45, +0.37].
- Angular Pair Greedy: mean −1.085; CI approximately [−2.61, +0.42].

All span zero. These exploratory intervals need a reproducible Python implementation and a consistent paired multiple-comparison plan before being used in a paper; they are not inferential proof of equality.

## Per-seed matched comparisons, raw rounding and compute cost

From `audited_paired_seed_deltas.csv`, the **same original GW normal index** was used for each method (16 per graph; 1,136 per method). Mean paired seed-level final-best differences versus historical HP (graph/seed equally weighted):

| New method | Mean paired final cut minus HP | Mean paired raw cut minus HP |
|---|---:|---:|
| Angular Margin Greedy | −1.4630 | −16.8072 |
| Angular Gain Greedy | −0.9710 | −17.0308 |
| Angular Pair Greedy | −2.1602 | −16.1452 |

These are different from best-of-16 differences because the maximum is nonlinear. The result that raw angular rounding underperformed HP by around 16–17 cut units per matched seed deserves mechanistic analysis; refinement substantially reduced the gap. Do **not** conflate raw and refined quality or pretend the 1,136 seed pairs are statistically independent across graph clusters.

Recorded average per-seed wall seconds (descriptive, from archived `seed_seconds`; **not** an equal-budget benchmark with controlled hardware scheduling): GW-RR **0.937 s**, historical HP **2.487 s**, Angular Margin **1.057 s**, Angular Gain **0.996 s**, Angular Pair **0.983 s**. The new methods saved total attempted great-circle sweep counts **62,888 / 65,488 / 65,320** for margin/gain/pair across all 1,136 starts each (roughly 55.4 / 57.6 / 57.5 per seed). The asymmetry in algorithmic work makes causal claims about CPU efficiency premature, although matching or nearly matching HP quality at lower reported cost is worth a dedicated follow-up.

## Next research and archival steps, without redundant numerical work

1. **Preserve** the 71 ZIPs, 109,159 member checksums, all five CSV/JSON analyses and the original VM checkpoints. Do not delete Experiment025 or change `src/maxkcut/*.py` or `scripts/experiment027_angular.py`. Leave the original full Experiment025 18-method raw archive for a separate durable backup/Release.
2. **Independent off-VM retrieval check:** download at least representative Release ZIP assets, recheck SHA256 against `index.json` and their per-member manifest and reconstruct partition scores where feasible. If expanded to every ZIP, record a checksum log in the repo.
3. **Statistical comparison with multiple comparisons and compute normalization:** preregister graph-level paired differences versus HP and refined GW; stratify by graph family/negative edges; compare at matched CPU budgets and matched semicircle sweep counts. Report effect sizes and confidence intervals, not just per-method win counts or in-sample portfolio selection. A fair three-arm portfolio comparator should receive an equal total budget.
4. **Scientific positioning:** these three angular-guided hyperplane search heuristics are new variations on the saved GW/EVA geometry, not the thesis's clustering-based initializers; do not imply the methods solve general Max-k-Cut or improve published approximation factors. Compare broader historical 18-method Experiment025 results if a full leaderboard is required.
5. **Experiment sequencing:** use the next unused sequential experiment ID after verifying current repo naming, keep research questions, code, parameters, outputs and failed hypotheses together with a handoff. No new EVA or old method recomputations.

**Source-of-truth rule:** This note is explanatory. For any scientific claim, reproduce it from the committed `published/experiment027/releases/experiment027-data-71g-2b313a624780291eb4dd/analysis/*` CSV/JSON and preserve SHA-256 provenance. PR #80 remains open as a draft; do not merge without user authorization.
