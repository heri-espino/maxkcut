# University Windows VM — exact locations and validation of frozen EVA factors

**Status:** Source-verified against the GitHub \`main\` branch on 2026-11-03. This document explains **where Experiment025 reads its previously solved EVA vectors**. It is NOT a request to rerun an SDP or an EVA optimization.

## The answer in one sentence

**The 71 factors are spread across three earlier experiment directories, NOT all stored inside Experiment021.** The authoritative map is \`results/gset/experiment021-complete-71-gset/inventory.json\`. The active \`src/maxkcut/experiment025_bestcut.py\` imports \`select_factor\` from \`src/maxkcut/experiment023_methods.py\`, which checks for both \`cases/Gxx.json\` and \`factors/Gxx.npz\` in its \`FACTOR_DIRS\` search order.

| Graphs (71 distinct) | Authoritative source directory, relative to checkout root | Factor example |
|---|---|---|
| G1–G59 (59) | \`results/gset/experiment002-eva/\` | \`results/gset/experiment002-eva/factors/G1.npz\` |
| G60–G62 (3) | \`results/gset/experiment021-complete-71-gset/eva-numba/\` | \`results/gset/experiment021-complete-71-gset/eva-numba/factors/G60.npz\` |
| G63–G67, G70, G72, G77, G81 (9) | \`results/gset/experiment003-eva-numba/\` | \`results/gset/experiment003-eva-numba/factors/G81.npz\` |

**CRITICAL G60 exception:** \`results/gset/experiment002-eva/factors/G60.npz\` also exists as an older artifact, but the verified Experiment021 inventory **does not select it**. Use the newer \`experiment021-complete-71-gset/eva-numba/factors/G60.npz\` with its corresponding case metadata. \`select_factor\` deliberately checks Experiment021 first. Never use a blind glob and take the first \`G60.npz\`; never silently substitute a factor when SHA validation fails.

Concrete example paths for the **interactive** university VM checkout:

    C:\Users\175199\maxkcut\data\G81.txt
    C:\Users\175199\maxkcut\results\gset\experiment003-eva-numba\cases\G81.json
    C:\Users\175199\maxkcut\results\gset\experiment003-eva-numba\factors\G81.npz

The runner can also have a **separate GitHub Actions checkout**: historical G60 case metadata reports \`C:\Users\175199\actions-runner\actions-runner\_work\maxkcut\maxkcut\`. This was an observed past runner working directory, not a guarantee of the path of future jobs. **These are separate worktrees.** A file visible to Actions may not exist in the user's interactive checkout, and vice versa. Always establish the current root first.

## Read this source code rather than guessing

- [Factor selection logic](https://github.com/heri-espino/maxkcut/blob/main/src/maxkcut/experiment023_methods.py): \`FACTOR_DIRS = (experiment021, experiment003, experiment002)\` and \`select_factor(graph)\`.
- [Actual Experiment025 consumer](https://github.com/heri-espino/maxkcut/blob/main/src/maxkcut/experiment025_bestcut.py): \`run_method\` calls \`select_factor(graph)\` and \`_factor_and_case(Path("data") / (graph + ".txt"), source)\`.
- [Factor loader / case and SHA checks](https://github.com/heri-espino/maxkcut/blob/main/src/maxkcut/eva_rounding_cli.py): \`_factor_and_case\` verifies original graph hash, completed case status, rank, sweeps, optimizer identity, and loads \`factors/Gxx.npz\`.
- [NPZ internal schema](https://github.com/heri-espino/maxkcut/blob/main/src/maxkcut/eva.py): \`load_eva_factor\` reads NumPy \`vectors\` (shape \`n × p\`) and JSON \`metadata\`, with \`allow_pickle=False\` and unit-vector checks.
- [Authoritative 71-entry inventory](https://github.com/heri-espino/maxkcut/blob/main/results/gset/experiment021-complete-71-gset/inventory.json): exact \`factor_source\`, \`factor_sha256\`, \`graph_sha256\`, \`rank\`, \`vertices\`, status for each graph.
- [Experiment021 completion report](https://github.com/heri-espino/maxkcut/blob/main/notes/experiments/experiment021-complete-71-gset.md): 71/71 verified sources, including G60/G61/G62, and how they were produced.

**Terminology:** EVA factors are **nonconvex rank-constrained Gram embeddings**, not guaranteed global optima or certificates of the full convex Max-Cut SDP. If \`Y\` is the \`n × p\` saved factor, its Gram matrix is \`X = Y Y^T\`, but never materialize full \`X\` for a large graph needlessly (G81 has 20,000 vertices). For all 71 Experiment025 rounding experiments, use the saved \`Y\` directly; no SDP matrix solution is needed. \`results/gset/full-sdp/summary.csv\` describes **a different historical full-SDP comparison**, not Experiment025's EVA source factors.

## Exact read-only diagnostic commands for the other AI's PowerShell terminal

**A. Verify which checkout and Python the agent is using**:

\`\`\`powershell
Get-Location
git rev-parse --show-toplevel
git branch --show-current
python -c "import sys; print(sys.executable)"
$env:PYTHONPATH = (Resolve-Path .\src).Path
Test-Path .\data\G81.txt
\`\`\`

If the shell is not in the intended interactive checkout, navigate to the **correct, existing** directory; do not clone over the existing tree. Example: \`Set-Location "$env:USERPROFILE\maxkcut"\`.

**B. Verify the three physical factor roots (does NOT rerun EVA):**

\`\`\`powershell
$paths = @(
  "results\gset\experiment002-eva",
  "results\gset\experiment021-complete-71-gset\eva-numba",
  "results\gset\experiment003-eva-numba"
)
foreach ($d in $paths) {
  $n = @(Get-ChildItem (Join-Path $d "factors") -Filter "G*.npz" -File -ErrorAction SilentlyContinue).Count
  Write-Host "$d -> $n NPZ files"
}
Test-Path .\results\gset\experiment021-complete-71-gset\inventory.json
Test-Path .\results\gset\experiment021-complete-71-gset\eva-numba\cases\G60.json
Test-Path .\results\gset\experiment021-complete-71-gset\eva-numba\factors\G60.npz
Test-Path .\results\gset\experiment003-eva-numba\cases\G81.json
Test-Path .\results\gset\experiment003-eva-numba\factors\G81.npz
\`\`\`

File counts alone are not an integrity test: Experiment002 also has an older \`G60.npz\`; the authoritative mapping is 59+3+9, while a raw directory count for Experiment002 may show 60.

**C. Run the repository's existing 71-graph factor locator:**

\`\`\`powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m maxkcut.experiment023_methods inventory
\`\`\`

Expected: JSON with \`"graph_count": 71\` and each graph's \`"source"\`. This command checks the presence of both a case and a factor, not contents or SHA integrity.

**D. Perform a stronger read-only integrity check (no optimization):**

\`\`\`powershell
@'
import json
from pathlib import Path
from maxkcut.experiment023_methods import GRAPH_IDS, select_factor
from maxkcut.eva_rounding_cli import _factor_and_case

manifest_path = Path("results/gset/experiment021-complete-71-gset/inventory.json")
inventory = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {r["graph"]: r for r in inventory["records"]}
failures = []
for graph in GRAPH_IDS:
    try:
        source = select_factor(graph)
        reference = expected[graph]
        actual_source = str(source).replace("/", "\\")
        if actual_source.lower() != reference["factor_source"].lower():
            raise ValueError(f"wrong factor directory: {actual_source}")
        _, vectors, case, graph_sha, factor_sha = _factor_and_case(
            Path("data") / f"{graph}.txt", source)
        if graph_sha != reference["graph_sha256"]:
            raise ValueError("graph SHA256 mismatch with inventory")
        if factor_sha != reference["factor_sha256"]:
            raise ValueError("factor SHA256 mismatch with inventory")
        if vectors.shape[1] != reference["rank"]:
            raise ValueError("factor rank mismatch with inventory")
        print(f"OK {graph:>3}: {vectors.shape}, {actual_source}")
    except Exception as exc:
        failures.append((graph, str(exc)))
        print(f"FAIL {graph}: {exc}")
print(f"VALIDATED {len(GRAPH_IDS) - len(failures)}/{len(GRAPH_IDS)}")
if failures:
    raise SystemExit("One or more archived EVA factors are missing or incompatible")
'@ | python -
\`\`\`

Expect \`VALIDATED 71/71\`. This uses the project's own loader, validates source hashes, completed case metadata and saved factors. It does **not** solve any EVA or SDP, mutate outputs, generate new random directions or recompute the 20,448 Experiment025 runs.

## What to do if the agent cannot find a factor

1. **Inspect current directory.** Relative paths assume the repository root, NOT \`scripts/\`, \`src/\`, \`results/\` or a different Actions worktree.
2. **Look under all three origins** and inspect the exact \`inventory.json\` source mapping for the missing graph.
3. **Differentiate missing case, missing NPZ and corruption.** \`select_factor\` requires both files; \`_factor_and_case\` further requires matching graph SHA, \`status \`converged\`/\`iteration_limit\`, rank/sweeps and optimizer identity. A file that merely exists does not prove it is correct.
4. **Check whether the agent is reading a different branch/worktree** from the one that successfully completed Experiment025. The interactive VM \`C:\Users\175199\maxkcut\` and historical Actions \`...\_work\maxkcut\maxkcut\` paths are not interchangeable.
5. **Check versioned Git files, not only local filesystem.** These exact factors and case metadata are visible on GitHub \`main\`; the repo's \`.gitattributes\` restricts Git LFS to Experiment025's new NPZ/JSONL files, **not** the historical Experiment002/003/021 EVA factor paths. Accordingly, a generic \`git lfs pull\` is not the presumed fix for a missing historical factor.
6. If the verified factor is genuinely missing in this checkout, **preserve the existing VM data**, compare remote source SHA, and recover the exact missing historical file in an **independent clean worktree** or through a reviewed minimal restoration. Do not replace an already present factor with a different version, force-reset the active experiment checkout, run \`git clean\`, run \`git add .\`, or launch \`eva_cli\` / \`gset-sdp-manual.yml\`.
7. **If factor SHA mismatches, STOP.** Report graph, selected path, expected/actual SHA, case metadata status, branch and git commit before any change. It is not legitimate to silently recompute a fresh factor and treat it as the historical Experiment025 input.

## Current machine, execution and publication state

- Windows 11 university machine, self-hosted Actions runner **PCAG103A-13**; existing Conda \`maxkcut\` interpreter historically \`C:\Users\175199\AppData\Local\miniconda3\envs\maxkcut\python.exe\`; **PowerShell 5.1** (\`powershell.exe\`, not \`pwsh\`).
- Experiment025 full local-only direct-VM run has finished according to two user terminal outputs: **71 graphs ×18 methods ×16 paired seeds = 20,448/20,448 completed; 1,278/1,278 full cells; \`all_complete=true\`**. Do not rerun it or the EVA solve.
- Raw Experiment025 iterations live under \`results/gset/experiment025-gw16-bestcut-quality/full/\`; compact report files under its \`analysis/\`. Local-only results may not be on GitHub. Treat the VM as the immediate working archive; **independent original-cut audit, off-VM backup and compact publication remain pending**.
- Prefer direct Windows VM commands for substantial computation and local inspection. GitHub Actions for small CI or Windows preflight only when no competing direct numerical process is active; never run both against the same result tree. See [complete operations guide](university-vm-experiment025-operations.md), [2026-11-03 compute completion](../notes/experiments/experiment025-full-local-computation-2026-11-03.md), and [source/failure recovery](../notes/experiments/experiment025-windows-local-recovery-2026-11-03.md).
- A separate historical Actions job's checkout may be transient and discard locally produced uncommitted results after the job; the user-driven interactive VM checkout is the intended persistent working copy. **Neither is a substitute for a verified off-VM backup.**

**Required next-agent first action:** Run the read-only \`experiment023_methods inventory\` command and, if necessary, the SHA integrity check above. Do not solve SDP/EVA and do not alter the completed scientific optimizer to “fix” a missing factor location.
