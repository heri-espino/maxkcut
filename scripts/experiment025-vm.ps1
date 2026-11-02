<#
Experiment025 VM driver: no Actions six-hour limit; resume by graph/method/seed.
Run at repository root:
  powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase pilot
  powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase full -AllGraphs
  powershell.exe -NoProfile -File .\scripts\experiment025-vm.ps1 -Phase coverage

Requires: Windows, Git for Windows with Git LFS, available "maxkcut"
Conda environment and GitHub push credentials. Writes ONLY to the dedicated
experiment025-results branch; no force-push, no changes to old experiments.
#>
param(
    [ValidateSet('pilot','full','coverage','seed-report')][string]$Phase = 'pilot',
    [string]$Graph = 'G11',
    [string]$Method = '',
    [switch]$OnlyGraph,
    [ValidateRange(0,16)][int]$Seed = 0,
    [ValidateSet('pilot','full')][string]$ReportPhase = 'full',
    [string]$ExpectedSourceSha256 = '',
    [switch]$AllGraphs,
    [ValidateRange(1,16)][int]$ChunkSize = 16,
    [string]$ResultsBranch = 'experiment025-results',
    [switch]$SkipTests
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
# Seed 1..16 is user-facing, but source GW seed indices are 0..15.
# Seed 0 retains the legacy 16-seed-per-method behavior.
if ($Seed -gt 0 -and (-not $OnlyGraph -or -not $Method -or $AllGraphs -or $Phase -eq 'coverage')) {
    throw 'Seed-specific Actions jobs require -OnlyGraph, -Method, and pilot/full phase'
}
$repo = (& git rev-parse --show-toplevel 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $repo) { throw 'Run from a checked-out Git repository.' }
Set-Location $repo.Trim()
if (-not (Test-Path 'src/maxkcut/experiment025_bestcut.py')) { throw 'Wrong repository: missing Experiment025 source' }
if (-not (Test-Path '.gitattributes')) { throw 'Missing Experiment025 scoped LFS configuration; update from main first' }
$origin = (& git remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or $origin -notmatch '(?i)heri-espino/maxkcut(?:\.git)?$') {
    throw "Origin is not heri-espino/maxkcut: $origin"
}
if ($ResultsBranch -ne 'experiment025-results') { throw 'Only the registered Experiment025 results branch is permitted' }
if ((@(& git status --porcelain)).Count -gt 0) {
    throw 'Git worktree is not clean. Commit or stash unrelated changes before running.'
}
& git lfs version
if ($LASTEXITCODE -ne 0) { throw 'Git LFS is required. Install Git LFS before executing large iteration archives.' }
& git lfs install --local
if ($LASTEXITCODE -ne 0) { throw 'git lfs install failed' }
& git fetch origin main
if ($LASTEXITCODE -ne 0) { throw 'Could not fetch current main' }
$branch = ([string](& git branch --show-current)).Trim()
if ($branch -ne $ResultsBranch) {
    # Refuse switching away from another agent's active worktree.
    $existingLocal = @(& git branch --list $ResultsBranch)
    & git ls-remote --exit-code --heads origin $ResultsBranch *> $null
    if ($LASTEXITCODE -notin @(0,2)) { throw "Could not check remote branch $ResultsBranch (network/authorization failure)" }
    $existsRemote = ($LASTEXITCODE -eq 0)
    if ($existsRemote) {
        & git fetch origin $ResultsBranch
        if ($LASTEXITCODE -ne 0) { throw "Could not fetch $ResultsBranch" }
        if ($existingLocal.Count -gt 0) {
            & git switch $ResultsBranch
            if ($LASTEXITCODE -ne 0) { throw "Could not switch to $ResultsBranch" }
            & git pull --ff-only origin $ResultsBranch
            if ($LASTEXITCODE -ne 0) { throw "Could not fast-forward $ResultsBranch; do not reset results" }
        } else {
            & git switch -c $ResultsBranch "origin/$ResultsBranch"
            if ($LASTEXITCODE -ne 0) { throw "Could not create local result tracking branch" }
        }
    } else {
        if ($existingLocal.Count -gt 0) {
            throw "Local $ResultsBranch exists but no matching remote; inspect manually"
        }
        & git switch -c $ResultsBranch "origin/main"
        if ($LASTEXITCODE -ne 0) { throw "Could not branch from current origin/main" }
        & git push -u origin $ResultsBranch
        if ($LASTEXITCODE -ne 0) { throw "Could not publish initial experiment results branch" }
    }
}
# Important: do NOT merge latest main into a branch with partially completed
# seeds. The source-tree SHA256 pin will reject mixing changed optimizer code.
$branch = ([string](& git branch --show-current)).Trim()
if ($branch -ne $ResultsBranch) { throw "Unexpected active branch $branch" }
if ((@(& git status --porcelain)).Count -gt 0) { throw "Dirty worktree after checkout" }
$lfsProbe=& git check-attr filter -- 'results/gset/experiment025-gw16-bestcut-quality/full/G11/gw-rr/0000.npz'
if ($LASTEXITCODE -ne 0 -or "$lfsProbe" -notmatch 'filter: lfs$') {
    throw 'Git LFS attributes not active for Experiment025 partitions'
}
$jsonlProbe=& git check-attr filter -- 'results/gset/experiment025-gw16-bestcut-quality/full/G11/gw-rr/progress/0000/events.jsonl'
if ($LASTEXITCODE -ne 0 -or "$jsonlProbe" -notmatch 'filter: lfs$') {
    throw 'Git LFS attributes not active for Experiment025 iteration logs'
}

# Use the existing university VM's "maxkcut" Conda Python. Windows
# PowerShell 5.1 is sufficient; PowerShell 7 (pwsh) is NOT required.
# Conda can be installed at a custom university path, not only under USERPROFILE.
$conda = Get-Command conda -ErrorAction SilentlyContinue
$discoveredPrefix = $null
if ($env:MAXKCUT_CONDA_PREFIX) {
    $configured = Join-Path $env:MAXKCUT_CONDA_PREFIX 'python.exe'
    if (-not (Test-Path $configured)) { throw "Configured MAXKCUT_CONDA_PREFIX has no python.exe: $env:MAXKCUT_CONDA_PREFIX" }
    if ((Split-Path $env:MAXKCUT_CONDA_PREFIX -Leaf) -ine 'maxkcut') { throw 'MAXKCUT_CONDA_PREFIX must identify the maxkcut environment' }
    $discoveredPrefix = $env:MAXKCUT_CONDA_PREFIX
}
if ($null -ne $conda) {
    $rawCondaEnvs = (& conda env list --json | Out-String)
    if ($LASTEXITCODE -eq 0 -and $rawCondaEnvs) {
        try {
            $allCondaEnvs = @((ConvertFrom-Json $rawCondaEnvs).envs)
            foreach ($envPath in $allCondaEnvs) {
                if ((Split-Path $envPath -Leaf) -ieq 'maxkcut') {
                    $discoveredPrefix = $envPath
                    break
                }
            }
        } catch {
            Write-Warning "Could not interpret conda env list JSON; falling back to known paths."
        }
    }
    # Attempt normal Conda shell activation, then ALWAYS use the actual
    # maxkcut\python.exe, including when the PowerShell hook is unavailable.
    try {
        $hook = (& conda shell.powershell hook | Out-String)
        if ($LASTEXITCODE -eq 0 -and $hook) {
            Invoke-Expression $hook
            conda activate maxkcut
        }
    } catch {
        Write-Warning "Conda PowerShell hook unavailable; using direct maxkcut interpreter."
    }
}
$python = $null
if ($env:CONDA_PREFIX -and (Split-Path $env:CONDA_PREFIX -Leaf) -ieq 'maxkcut') {
    $candidate=Join-Path $env:CONDA_PREFIX 'python.exe'
    if (Test-Path $candidate) { $python=$candidate }
}
if (-not $python -and $discoveredPrefix) {
    $candidate=Join-Path $discoveredPrefix 'python.exe'
    if (Test-Path $candidate) { $python=$candidate }
}
if (-not $python) {
    foreach ($candidate in @(
        "$env:USERPROFILE\miniconda3\envs\maxkcut\python.exe",
        "$env:USERPROFILE\miniforge3\envs\maxkcut\python.exe",
        "$env:USERPROFILE\anaconda3\envs\maxkcut\python.exe",
        "$env:USERPROFILE\.conda\envs\maxkcut\python.exe",
        "$env:LOCALAPPDATA\Programs\miniconda3\envs\maxkcut\python.exe",
        "C:\ProgramData\anaconda3\envs\maxkcut\python.exe",
        "$env:LOCALAPPDATA\miniconda3\envs\maxkcut\python.exe",
        "C:\ProgramData\miniconda3\envs\maxkcut\python.exe"
    )) {
        if (Test-Path $candidate) { $python=$candidate; break }
    }
}
if (-not $python) { throw 'maxkcut Conda Python not found; activate or configure the existing environment' }
$prefix = Split-Path $python -Parent
$env:CONDA_PREFIX = $prefix
$env:CONDA_DEFAULT_ENV = 'maxkcut'
foreach ($sub in @('Library\mingw-w64\bin','Library\usr\bin','Library\bin','Scripts','')) {
    $dir=Join-Path $prefix $sub
    if (Test-Path $dir) { $env:PATH="$dir;$env:PATH" }
}
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:NUMBA_NUM_THREADS='1'
$env:PYTHONUTF8='1'
Write-Host "Experiment025 source branch: $branch"
Write-Host "Windows PowerShell version: $($PSVersionTable.PSVersion)"
Write-Host "Conda environment: maxkcut ($prefix)"
Write-Host "Conda Python executable: $python"
$env:EXP025_EXPECTED_SOURCE_SHA=$ExpectedSourceSha256
@'
import json
import numpy
import scipy
import numba
import pytest
from maxkcut.experiment025_bestcut import GRAPH_IDS, METHODS, algorithm_source_sha256
from maxkcut.experiment023_references import load_references
assert len(GRAPH_IDS)==71 and len(METHODS)==18
import os
expected=os.environ.get("EXP025_EXPECTED_SOURCE_SHA", "").strip()
if expected and algorithm_source_sha256()!=expected:
    raise RuntimeError("Optimizer source does not match the GitHub Actions dispatch revision")
references,_=load_references()
assert len(references)==71
print("Preflight:",json.dumps(dict(graphs=71,methods=18,
     optimizer_source_sha256=algorithm_source_sha256(),
     numpy=numpy.__version__,scipy=scipy.__version__)))
'@ | & $python -
if ($LASTEXITCODE -ne 0) { throw 'Python preflight, references or missing factor dependencies failed' }

function Publish-Checkpoint {
    param([string]$Message)
    & git add -A -- 'results/gset/experiment025-gw16-bestcut-quality'
    if ($LASTEXITCODE -ne 0) { throw 'Unable to stage result archives' }
    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { return }
    & git -c user.name='Experiment025 VM Runner' -c user.email='experiment025-vm@users.noreply.github.com' commit -m $Message
    if ($LASTEXITCODE -ne 0) { throw 'Unable to commit result checkpoint' }
    if ((@(& git status --porcelain)).Count -gt 0) {
        throw 'Unexpected unstaged modifications; refusing rebase or force-push'
    }
    for ($attempt=1;$attempt -le 5;$attempt++) {
        & git fetch origin $ResultsBranch
        if ($LASTEXITCODE -ne 0) { throw 'Cannot fetch remote result branch' }
        & git rebase "origin/$ResultsBranch"
        if ($LASTEXITCODE -ne 0) {
            throw 'Checkpoint rebase conflicted. Preserve this branch; do not force push.'
        }
        & git push origin "HEAD:$ResultsBranch"
        if ($LASTEXITCODE -eq 0) { return }
        Start-Sleep -Seconds (2*$attempt)
    }
    throw 'Could not push Experiment025 results after retries'
}
function Execute-One {
    param([string]$G,[string]$M,[string]$Part)
    Write-Host "Experiment025: $Part / $G / $M / up to $ChunkSize fresh seeds per invocation"
    $base="results/gset/experiment025-gw16-bestcut-quality/$Part/$G/$M"
    $logdir="results/gset/experiment025-gw16-bestcut-quality/$Part/$G/logs"
    New-Item -ItemType Directory -Force -Path $logdir | Out-Null
    $logfile=Join-Path $logdir "$M-vm.log"
    if ($Seed -gt 0) {
        $seedIndex=$Seed-1
        & $python -m maxkcut.experiment025_bestcut single --graph $G --method $M --phase $Part --seed-index $seedIndex 2>&1 | Tee-Object -FilePath $logfile -Append
        $exit=$LASTEXITCODE
        Publish-Checkpoint "experiment025: $Part $G $M matched GW seed $Seed"
        if ($exit -ne 0) { throw "Experiment025 seed $Seed on $G/$M failed; inspect journal" }
        $record=Get-Content -LiteralPath (Join-Path $base 'status.json') -Raw | ConvertFrom-Json
        if (@($record.completed_seed_indices) -notcontains $seedIndex) {
            throw "Seed $Seed was not verified and checkpointed on $G/$M"
        }
        return
    }
    $tryCount=[int][math]::Ceiling(16.0/$ChunkSize)
    for($chunk=1; $chunk -le $tryCount; $chunk++) {
        # Neither Python nor PowerShell imposes a method-level time budget.
        & $python -m maxkcut.experiment025_bestcut single --graph $G --method $M --phase $Part --max-new-seeds $ChunkSize 2>&1 | Tee-Object -FilePath $logfile -Append
        $exit=$LASTEXITCODE
        # Retain all locally finished iterations AND upload completed seeds.
        Publish-Checkpoint "experiment025: $Part $G $M progress block $chunk"
        if ($exit -ne 0) { throw "Experiment025 $G/$M failed; see $logfile and progress archives" }
    }
    $statusPath=Join-Path $base 'status.json'
    if (-not (Test-Path -LiteralPath $statusPath)) { throw "Missing $statusPath" }
    $status=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($status.status -ne 'complete_16_best_seen' -or $status.starts_completed -ne 16) {
        throw "Incomplete $G $M ($($status.starts_completed)/16 seeds); stop before ranking"
    }
}
if (-not $SkipTests) {
    & $python -m pytest tests/test_experiment025_bestcut.py tests/test_experiment024_paired_gw16.py tests/test_experiment024_integration.py -q
    if ($LASTEXITCODE -ne 0) { throw 'Numerical integration tests failed: no optimization executed' }
}
if ($Phase -eq 'seed-report') {
    & $python -m maxkcut.experiment025_seed_progress --phase $ReportPhase
    if ($LASTEXITCODE -ne 0) { throw 'Seed-round cumulative best-so-far report failed' }
    Publish-Checkpoint "experiment025: record $ReportPhase cumulative seed-quality progress"
} elseif ($Phase -eq 'coverage') {
    & $python -m maxkcut.experiment025_bestcut summary --phase full
    if ($LASTEXITCODE -ne 0) { throw 'Coverage report failed' }
    Publish-Checkpoint 'experiment025: final original-cut coverage and best-ever summaries'
    $report=Get-Content 'results/gset/experiment025-gw16-bestcut-quality/analysis/full_coverage.json' -Raw | ConvertFrom-Json
    if (-not $report.all_complete) { throw "Experiment incomplete: $($report.completed_cells)/$($report.planned_cells) graph-method cells" }
} else {
    $graphs=if ($Phase -eq 'pilot' -and $OnlyGraph) { @($Graph) }
            elseif ($Phase -eq 'pilot') { @('G1','G14','G11') }
            elseif ($AllGraphs) { @((1..67 | ForEach-Object { "G$_" })) + @('G70','G72','G77','G81') }
            else { @($Graph) }
    # AllGraphs phase full explicitly expands the canonical 71 graphs.
    $methods=if ($Method) { @($Method) } else {
        @('gw-rr','control-paired','gw-threshold','cutaware-kmeans',
          'hyperplane-sweep','multigroup-k4','multigroup-k8',
          'em-hard','em-cem','em-gibbs','mst-l2-longest','mst-l2-balanced',
          'mst-l2-diameter','mst-l2-bestcut','mst-l2-multicut',
          'mst-l1-bestcut','mst-linf-bestcut','control-tabu')
    }
    foreach($g in $graphs) {
        foreach($m in $methods) { Execute-One -G $g -M $m -Part $Phase }
    }
    if ($Seed -eq 0) {
        & $python -m maxkcut.experiment025_bestcut summary --phase $Phase
        if ($LASTEXITCODE -ne 0) { throw 'Final summary calculation failed' }
        Publish-Checkpoint "experiment025: updated $Phase status and reference-cut comparisons"
    }
}
Write-Host "Experiment025 operation completed for phase $Phase. All tracked results are in $ResultsBranch."
