# Compiled EVA performance study

An isolated Numba backend preserves sequential updates and saves separate factor checkpoints. Benchmark NumPy and compiled EVA with `python -m maxkcut.eva_benchmark_cli --graphs G1 G11 G55` after installing Numba 0.67 or newer. No user's existing experiment002 factors are changed.

## Observed timings

The official MATLAB paper reports EVA times G1=1.00 seconds,
G11=1.49 seconds, and G55=5.49 seconds. Read-only verification
of experiment002's archived case files found G1=1.532 seconds,
G11=5.281 seconds and G55=10.813 seconds. Despite the speed
difference, the G1 and G55 continuous objectives agreed with
published values to the reported precision. These are different
computers, not hardware-controlled timings.

## Algorithmic equivalence

Our compiled kernel uses the same sequential vertex-update order
as MATLAB and the same closed-form coordinate optimum. It accumulates
the exact local objective gain instead of rescanning the graph after
every sweep and evaluates the largest update angle using a single
acos applied to the smallest per-vertex cosine.
The full objective is independently recomputed at checkpoints.

The user has NumPy 2.5, so install Numba version 0.67 or newer.
JIT compilation is excluded from numerical solver time but
recorded separately. Do not compare wall times without accounting
for first-run JIT compilation.

## Validation

Run these commands from the experiment003 branch after the
existing EVA study has stopped:

```powershell
conda activate maxkcut
python -m pip install "numba>=0.67,<0.69"
$env:PYTHONPATH = Join-Path (Get-Location).Path "src"
python -m pytest tests/test_eva_numba.py tests/test_eva.py -q
python -m maxkcut.eva_benchmark_cli --graphs G1 G11 G55 --output-dir results/gset/experiment003-eva-perf --push --push-branch perf/eva-numba-sequential-20261028
```

Inspect the new timing and objective-parity report under
results/gset/experiment003-eva-perf/summary.csv.
Never overwrite experiment002 factors with the compiled backend.
