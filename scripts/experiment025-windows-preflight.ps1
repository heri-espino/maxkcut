<#
Experiment025 Windows university-machine smoke test.
Does NOT execute Max-Cut optimization or modify the results branch.
Runs under built-in Windows PowerShell 5.1 (powershell.exe), not pwsh.
#>
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

Write-Host "Windows runner: $env:RUNNER_NAME"
Write-Host "PowerShell version: $($PSVersionTable.PSVersion)"
if ($env:RUNNER_NAME -ne 'PCAG103A-13') {
    throw "Wrong runner. Expected PCAG103A-13, got $env:RUNNER_NAME"
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git for Windows is missing from PATH"
}
& git lfs version
if ($LASTEXITCODE -ne 0) { throw 'Git LFS is not available' }

$prefix = $null
if ($env:CONDA_PREFIX -and (Split-Path $env:CONDA_PREFIX -Leaf) -ieq 'maxkcut') {
    if (Test-Path (Join-Path $env:CONDA_PREFIX 'python.exe')) {
        $prefix=$env:CONDA_PREFIX
    }
}
if (-not $prefix -and (Get-Command conda -ErrorAction SilentlyContinue)) {
    $json=(& conda env list --json | Out-String)
    if ($LASTEXITCODE -eq 0 -and $json) {
        $entries=@((ConvertFrom-Json $json).envs)
        foreach($entry in $entries) {
            if ((Split-Path $entry -Leaf) -ieq 'maxkcut' -and
                 (Test-Path (Join-Path $entry 'python.exe'))) {
                $prefix=$entry
                break
            }
        }
    }
}
if (-not $prefix) {
    foreach ($candidate in @(
        "$env:USERPROFILE\miniconda3\envs\maxkcut",
        "$env:USERPROFILE\miniforge3\envs\maxkcut",
        "$env:USERPROFILE\anaconda3\envs\maxkcut",
        "$env:LOCALAPPDATA\miniconda3\envs\maxkcut",
        "C:\ProgramData\miniconda3\envs\maxkcut"
    )) {
        if (Test-Path (Join-Path $candidate 'python.exe')) {
            $prefix=$candidate
            break
        }
    }
}
if (-not $prefix) {
    throw 'Conda environment maxkcut not found for runner account. Verify conda env list and runner-user PATH.'
}
$python=Join-Path $prefix 'python.exe'
Write-Host "Conda maxkcut interpreter: $python"
$env:PYTHONPATH=Join-Path (Get-Location).Path 'src'
$env:CONDA_PREFIX=$prefix
$env:CONDA_DEFAULT_ENV='maxkcut'
foreach($sub in @('Library\mingw-w64\bin','Library\usr\bin','Library\bin','Scripts','')) {
    $entry=Join-Path $prefix $sub
    if (Test-Path $entry) { $env:PATH="$entry;$env:PATH" }
}
$env:NUMBA_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
@'
import sys
import numpy
import scipy
import numba
import pytest
from maxkcut.experiment025_bestcut import GRAPH_IDS, METHODS, algorithm_source_sha256
from maxkcut.experiment023_references import load_references
references, _ = load_references()
assert len(GRAPH_IDS) == 71 and len(METHODS) == 18 and len(references) == 71
print("Conda Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("Numba:", numba.__version__)
print("Optimizer source hash:", algorithm_source_sha256())
'@ | & $python -
if ($LASTEXITCODE -ne 0) { throw "maxkcut Conda Python import/reference preflight failed" }
Write-Host "PASS: Windows PowerShell, dedicated runner, Git LFS, Conda maxkcut and 71 reference graphs."
