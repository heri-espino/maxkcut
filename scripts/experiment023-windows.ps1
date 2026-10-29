param(
    [ValidateSet('pilot','full','aggregate')][string]$Phase = 'pilot',
    [string]$Graph = 'G11'
)
$ErrorActionPreference = 'Stop'
$branch = 'experiment023-all-method-topology'
if ($env:GITHUB_REF -and $env:GITHUB_REF -ne "refs/heads/$branch") {
    throw "Ref $env:GITHUB_REF is not $branch. Refusing to publish results."
}
$python = $null
$candidates = @(
    "$env:USERPROFILE\miniconda3\envs\maxkcut\python.exe",
    "$env:USERPROFILE\miniforge3\envs\maxkcut\python.exe",
    "$env:USERPROFILE\anaconda3\envs\maxkcut\python.exe",
    "$env:LOCALAPPDATA\miniconda3\envs\maxkcut\python.exe",
    "$env:LOCALAPPDATA\Programs\miniconda3\envs\maxkcut\python.exe",
    "C:\ProgramData\miniconda3\envs\maxkcut\python.exe"
)
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) { $python = $candidate; break }
}
if (-not $python) { throw 'Existing maxkcut Conda environment not found; install/configure manually.' }
$prefix=Split-Path $python -Parent
$env:PATH = "$(Join-Path $prefix 'Library\bin');$(Join-Path $prefix 'Scripts');$prefix;$env:PATH"
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
& $python -c 'import numpy, scipy, numba, pytest; import maxkcut.experiment023_methods; print("Verified maxkcut Conda")'
if ($LASTEXITCODE -ne 0) { throw 'Missing required installed packages.' }

function Publish-Results {
    param([string]$Message)
    git config user.name 'github-actions[bot]'
    git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
    git add -- 'results/gset/experiment023-method-topology'
    $feature = 'results/gset/experiment022-graph-structure-methods/graph_features.csv'
    if (Test-Path -LiteralPath $feature) { git add -- $feature }
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { Write-Host 'No new results to publish'; return }
    git commit -m $Message
    if ($LASTEXITCODE -ne 0) { throw 'Cannot commit results' }
    for ($attempt=1; $attempt -le 5; $attempt++) {
        git fetch origin $branch
        if ($LASTEXITCODE -ne 0) { throw 'Fetch failed' }
        git rebase "origin/$branch"
        if ($LASTEXITCODE -ne 0) { throw 'Rebase conflict; refusing overwrite' }
        git push origin "HEAD:$branch"
        if ($LASTEXITCODE -eq 0) { Write-Host 'Results committed and pushed'; return }
        Start-Sleep -Seconds (3*$attempt)
    }
    throw 'Unable to push results after five safe retries'
}

if ($Phase -eq 'pilot') {
    & $python -m pytest tests/test_experiment023_methods.py tests/test_graph_structure_methods.py tests/test_historical_structure_comparison.py -q
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic test suite failed; full benchmark blocked.' }
    & $python -m maxkcut.experiment023_methods inventory
    if ($LASTEXITCODE -ne 0) { throw 'Original frozen EVA factor inventory incomplete.' }
    & $python -m maxkcut.graph_structure_methods
    if ($LASTEXITCODE -ne 0) { throw 'Structure feature extraction failed.' }
    foreach ($g in @('G1','G14','G11')) {
        Write-Host "::group::PILOT $g"
        & $python -m maxkcut.experiment023_methods batch --graph $g --phase pilot --budget-seconds 120 --timeout-seconds 270
        if ($LASTEXITCODE -ne 0) { throw "Pilot process failed for $g" }
        # A pilot only passes if every method ran without failure or hard timeout.
        @'
import json, pathlib, sys
root=pathlib.Path('results/gset/experiment023-method-topology/pilot')/sys.argv[1]/'statuses'
rows=[json.loads(p.read_text()) for p in root.glob('*.json')]
assert len(rows)==17, (sys.argv[1], len(rows))
assert all(r['status'] in ('completed_restarts','budget_limited') for r in rows), [(r['method'], r['status']) for r in rows]
print('Pilot passed:',sys.argv[1],len(rows),'methods')
'@ | & $python - $g
        if ($LASTEXITCODE -ne 0) { throw "Pilot $g has failed, incomplete or timed-out methods; full sweep blocked." }
        Publish-Results "experiment023: checkpoint pilot $g (all historical method families)"
        Write-Host '::endgroup::'
    }
    & $python -m maxkcut.experiment023_methods aggregate --phase pilot
    if ($LASTEXITCODE -ne 0) { throw 'Pilot coverage aggregate failed' }
    Publish-Results 'experiment023: preserve pilot completeness report and graph features'
} elseif ($Phase -eq 'full') {
    & $python -m maxkcut.experiment023_methods batch --graph $Graph --phase full --budget-seconds 240 --timeout-seconds 510
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed unexpectedly for $Graph" }
    Publish-Results "experiment023: checkpoint full $Graph across 17 methods"
} else {
    & $python -m maxkcut.experiment023_methods aggregate --phase full
    if ($LASTEXITCODE -ne 0) { throw 'Final coverage analysis failed' }
    Publish-Results 'experiment023: report full method-by-graph coverage including incomplete cases'
    @'
import json, pathlib, sys
p=pathlib.Path('results/gset/experiment023-method-topology/analysis/full_coverage.json')
r=json.loads(p.read_text())
print(json.dumps(r,indent=2))
if r['complete_graph_method_rows']!=71*17:
    print('INCOMPLETE: some graph-methods timed out, failed, or have not run',file=sys.stderr)
    sys.exit(2)
'@ | & $python -
    if ($LASTEXITCODE -ne 0) { throw 'Coverage is incomplete; consult saved explicit status table; do not claim 71x17 success' }
}
