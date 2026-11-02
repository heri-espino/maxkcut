param(
    [ValidateSet('pilot','full','coverage')][string]$Phase='pilot',
    [string]$Graph='G11'
)
$ErrorActionPreference='Stop'
$branch='experiment025-gw16-bestcut-quality'
if ($env:GITHUB_REF -and $env:GITHUB_REF -ne "refs/heads/$branch") {
    throw "Ref mismatch: $env:GITHUB_REF"
}
$python=$null
foreach ($candidate in @(
    "$env:USERPROFILE\miniconda3\envs\maxkcut\python.exe",
    "$env:LOCALAPPDATA\miniconda3\envs\maxkcut\python.exe",
    "$env:USERPROFILE\miniforge3\envs\maxkcut\python.exe",
    "$env:USERPROFILE\anaconda3\envs\maxkcut\python.exe",
    "C:\ProgramData\miniconda3\envs\maxkcut\python.exe"
)) {
    if (Test-Path -LiteralPath $candidate) { $python=$candidate; break }
}
if (-not $python) {
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        $envs=& conda env list --json | ConvertFrom-Json
        foreach ($prefix in $envs.envs) {
            if ((Split-Path $prefix -Leaf) -eq 'maxkcut') {
                $candidate=Join-Path $prefix 'python.exe'
                if (Test-Path -LiteralPath $candidate) { $python=$candidate; break }
            }
        }
    }
}
if (-not $python) { throw 'Existing maxkcut Conda interpreter not found' }
$prefix=Split-Path $python -Parent
foreach ($sub in @('Library\mingw-w64\bin','Library\usr\bin','Library\bin','Scripts','')) {
    $dir=Join-Path $prefix $sub
    if (Test-Path -LiteralPath $dir) { $env:PATH="$dir;$env:PATH" }
}
$env:PYTHONPATH=Join-Path (Get-Location).Path 'src'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:NUMBA_NUM_THREADS='1'
$env:PYTHONUTF8='1'
@'
import numpy, scipy, numba, pytest
from maxkcut.experiment025_bestcut import GRAPH_IDS, METHODS
assert len(GRAPH_IDS)==71 and len(METHODS)==18
print("Experiment025 best-so-far Python environment verified")
'@ | & $python -
if ($LASTEXITCODE -ne 0) { throw 'Experiment025 Python preflight failed' }

$methods=@(
    'gw-rr','control-paired','gw-threshold','cutaware-kmeans',
    'hyperplane-sweep','multigroup-k4','multigroup-k8',
    'em-hard','em-cem','em-gibbs',
    'mst-l2-longest','mst-l2-balanced','mst-l2-diameter',
    'mst-l2-bestcut','mst-l2-multicut','mst-l1-bestcut',
    'mst-linf-bestcut','control-tabu'
)
if ($methods.Count -ne 18) { throw 'Expected 18 method protocols' }

function Publish-Checkpoint {
    param([string]$CommitMessage)
    git config user.name 'github-actions[bot]'
    git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
    git add -A -- 'results/gset/experiment025-gw16-bestcut-quality'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot stage output results' }
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { return }
    git commit -m $CommitMessage
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
    $dirty=@(git status --porcelain)
    if ($dirty.Count -gt 0) {
        $dirty | ForEach-Object { Write-Host $_ }
        throw 'Dirty worktree; refusing to rebase or force-push'
    }
    for ($attempt=1; $attempt -le 5; $attempt++) {
        git fetch origin $branch
        if ($LASTEXITCODE -ne 0) { throw 'Fetch failed' }
        git rebase "origin/$branch"
        if ($LASTEXITCODE -ne 0) { throw 'Rebase conflict; no force-push' }
        git push origin "HEAD:$branch"
        if ($LASTEXITCODE -eq 0) { return }
        Start-Sleep -Seconds (2*$attempt)
    }
    throw 'Unable to safely publish checkpoint'
}

function Execute-Graph {
    param([string]$Case,[string]$CasePhase)
    foreach ($method in $methods) {
        Write-Host "::group::Best-seen $Case $method"
        # Four disjoint seed-index chunks, each followed by a Git push.
        # No timeout/elapsed wall-clock condition is used to rank methods.
        for ($chunk=1; $chunk -le 4; $chunk++) {
            & $python -m maxkcut.experiment025_bestcut single --graph $Case --method $method --phase $CasePhase --max-new-seeds 4
            $ec=$LASTEXITCODE
            # Preserve available iteration journals even on algorithm failure.
            Publish-Checkpoint "experiment025: $Case $method quality checkpoint $chunk"
            if ($ec -ne 0) { throw "Algorithm failure: $Case $method chunk $chunk; inspect checkpoint" }
        }
        $status="results/gset/experiment025-gw16-bestcut-quality/$CasePhase/$Case/$method/status.json"
        if (-not (Test-Path -LiteralPath $status)) { throw "Missing status for $Case $method" }
        $record=Get-Content -LiteralPath $status -Raw | ConvertFrom-Json
        if ($record.status -ne 'complete_16_best_seen' -or $record.starts_completed -ne 16) {
            throw "Method incomplete: $Case $method has $($record.starts_completed) refinements"
        }
        Write-Host "::endgroup::"
    }
}
if ($Phase -eq 'pilot') {
    & $python -m pytest tests/test_experiment025_bestcut.py tests/test_experiment024_paired_gw16.py tests/test_experiment024_integration.py -q
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic experiment tests failed' }
    foreach ($g in @('G1','G14','G11')) { Execute-Graph -Case $g -CasePhase pilot }
    & $python -m maxkcut.experiment025_bestcut summary --phase pilot
    if ($LASTEXITCODE -ne 0) { throw 'Pilot coverage analysis failed' }
    Publish-Checkpoint 'experiment025: preserve paired best-seen pilot coverage'
} elseif ($Phase -eq 'full') {
    Execute-Graph -Case $Graph -CasePhase full
} else {
    & $python -m maxkcut.experiment025_bestcut summary --phase full
    if ($LASTEXITCODE -ne 0) { throw 'Coverage computation failed' }
    Publish-Checkpoint 'experiment025: archive best-seen full coverage'
    $p='results/gset/experiment025-gw16-bestcut-quality/analysis/full_coverage.json'
    $state=Get-Content -LiteralPath $p -Raw | ConvertFrom-Json
    if (-not $state.all_complete) {
        throw "Coverage incomplete: $($state.completed_cells) of $($state.planned_cells) graph-methods"
    }
}
