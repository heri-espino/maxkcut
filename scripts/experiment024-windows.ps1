param(
    [ValidateSet('pilot','full','coverage')][string]$Phase = 'pilot',
    [string]$Graph = 'G11'
)
$ErrorActionPreference='Stop'
$branch='experiment024-paired-gw16'
if ($env:GITHUB_REF -and $env:GITHUB_REF -ne "refs/heads/$branch") {
    throw "Ref mismatch ($env:GITHUB_REF). Do not publish to another branch."
}
$python=$null
$candidates=@(
    "$env:USERPROFILE\miniconda3\envs\maxkcut\python.exe",
    "$env:LOCALAPPDATA\miniconda3\envs\maxkcut\python.exe",
    "$env:USERPROFILE\miniforge3\envs\maxkcut\python.exe",
    "$env:USERPROFILE\anaconda3\envs\maxkcut\python.exe",
    "C:\ProgramData\miniconda3\envs\maxkcut\python.exe"
)
foreach ($path in $candidates) {
    if (Test-Path -LiteralPath $path) { $python=$path; break }
}
if (-not $python) {
    $conda=Get-Command conda -ErrorAction SilentlyContinue
    if ($conda) {
        $known=& conda env list --json | ConvertFrom-Json
        foreach ($prefix in $known.envs) {
            if ((Split-Path $prefix -Leaf) -eq 'maxkcut') {
                $candidate=Join-Path $prefix 'python.exe'
                if (Test-Path $candidate) { $python=$candidate; break }
            }
        }
    }
}
if (-not $python) { throw 'Missing existing Conda maxkcut environment' }
$prefix=Split-Path $python -Parent
foreach ($name in @('Library\mingw-w64\bin','Library\usr\bin','Library\bin','Scripts','')) {
    $dir=Join-Path $prefix $name
    if (Test-Path $dir) { $env:PATH="$dir;$env:PATH" }
}
$env:PYTHONPATH=Join-Path (Get-Location).Path 'src'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMBA_NUM_THREADS='1'
$env:PYTHONUTF8='1'
Write-Host "Using $python"
@'
import numpy, scipy, numba, pytest
from maxkcut.experiment024_paired_gw16 import GRAPH_IDS, METHODS
assert len(GRAPH_IDS) == 71 and len(METHODS) == 18
print("Experiment024 Python preflight OK", numpy.__version__)
'@ | & $python -
if ($LASTEXITCODE -ne 0) { throw 'Python preflight failed; see traceback' }

function Publish-Results {
    param([string]$Message)
    git config user.name 'github-actions[bot]'
    git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
    git add -A -- 'results/gset/experiment024-paired-gw16'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot stage Experiment024 results' }
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { Write-Host 'No new outputs to publish'; return }
    git commit -m $Message
    if ($LASTEXITCODE -ne 0) { throw 'Cannot commit Experiment024 results' }
    $uncommitted=@(git status --porcelain)
    if ($LASTEXITCODE -ne 0 -or $uncommitted.Count -gt 0) {
        $uncommitted | ForEach-Object { Write-Host $_ }
        throw 'Dirty worktree: preserving outputs instead of risking rebase'
    }
    for ($attempt=1; $attempt -le 5; $attempt++) {
        git fetch origin $branch
        if ($LASTEXITCODE -ne 0) { throw 'Fetch failed' }
        git rebase "origin/$branch"
        if ($LASTEXITCODE -ne 0) { throw 'Rebase failed; no force push allowed' }
        git push origin "HEAD:$branch"
        if ($LASTEXITCODE -eq 0) { Write-Host 'Checkpoint committed and pushed'; return }
        Start-Sleep -Seconds (3*$attempt)
    }
    throw 'Unable to push checkpoint safely'
}
function Assert-CompleteGraph {
    param([string]$CheckGraph,[string]$CheckPhase)
    @'
import json, pathlib, sys
graph,phase=sys.argv[1:]
root=pathlib.Path('results/gset/experiment024-paired-gw16')/phase/graph/'statuses'
entries=[json.loads(x.read_text(encoding='utf8')) for x in root.glob('*.json')]
errors=[(x['method'],x['status'],x.get('starts_completed')) for x in entries
        if x.get('status')!='complete_16' or x.get('starts_completed')!=16]
print('Paired GW-16 graph validation:',graph,phase,len(entries),'of 18',errors)
if len(entries)!=18 or errors:
    sys.exit(2)
'@ | & $python - $CheckGraph $CheckPhase
    if ($LASTEXITCODE -ne 0) { throw "Incomplete 16-seed comparison on $CheckGraph ($CheckPhase)" }
}

if ($Phase -eq 'pilot') {
    & $python -m pytest tests/test_experiment024_paired_gw16.py tests/test_experiment024_integration.py -q
    if ($LASTEXITCODE -ne 0) { throw 'Seed-pairing unit tests failed' }
    foreach ($g in @('G1','G14','G11')) {
        Write-Host "::group::PILOT $g"
        & $python -m maxkcut.experiment024_paired_gw16 batch --graph $g --phase pilot --hard-timeout-seconds 180
        $ec=$LASTEXITCODE
        Publish-Results "experiment024: preserve 16-GW-start pilot $g and diagnostic logs"
        if ($ec -ne 0) { throw "Pilot batch failed for $g; logs committed" }
        Assert-CompleteGraph -CheckGraph $g -CheckPhase pilot
        Write-Host '::endgroup::'
    }
    & $python -m maxkcut.experiment024_analysis --phase pilot
    if ($LASTEXITCODE -ne 0) { throw 'Pilot comparative analysis failed' }
    Publish-Results 'experiment024: publish paired pilot topology report'
} elseif ($Phase -eq 'full') {
    & $python -m maxkcut.experiment024_paired_gw16 batch --graph $Graph --phase full --hard-timeout-seconds 420
    $ec=$LASTEXITCODE
    Publish-Results "experiment024: checkpoint all 16 starts for $Graph"
    if ($ec -ne 0) { throw "Full batch process failed for $Graph; logs saved" }
    Assert-CompleteGraph -CheckGraph $Graph -CheckPhase full
} else {
    & $python -m maxkcut.experiment024_analysis --phase full
    if ($LASTEXITCODE -ne 0) { throw 'Full topology analysis failed' }
    Publish-Results 'experiment024: archive 71-graph 16-seed paired coverage and topology comparison'
    @'
import json, pathlib, sys
p=pathlib.Path('results/gset/experiment024-paired-gw16/analysis/full_coverage.json')
report=json.loads(p.read_text(encoding='utf8'))
print(json.dumps(report,indent=2))
if not report['primary_analysis_valid']:
    print('INCOMPLETE 16-of-16 methods; report is descriptive only',file=sys.stderr)
    sys.exit(2)
'@ | & $python -
    if ($LASTEXITCODE -ne 0) { throw 'Coverage incomplete: see saved per-method statuses' }
}
