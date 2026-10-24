# Windows PowerShell: run your first G-set experiment

This page uses **PowerShell**, not Bash. The `maxkcut` CLI is the same on Windows, macOS and Linux; **only the shell's line-continuation syntax changes**.

## Start in the repository

```powershell
cd C:\Users\175199\maxkcut
git pull origin main
conda activate maxkcut
python -m pytest -q
maxkcut --list-solvers
```

If you have already activated the environment or are already in the project directory, skip those setup commands. Installing fresh uses `conda env create -f environment.yml`; updating an existing Conda environment uses `conda env update -f environment.yml --prune`.

## Graph inspection (does not solve an SDP)

```powershell
maxkcut --graphs G11 --inspect-only
```

Review its reported `n`, number of edges, and edge-weight range before running a solver.

## First G11 Max-Cut SDP

**Copy the entire line**:

```powershell
maxkcut --graphs G11 --k 2 --solver SCS --eps 1e-3 --max-iters 3000 --restarts 20 --seed 2026 --output reports/gset/g11-first.jsonl
```

Alternatively, the same command may be written on multiple lines using PowerShell's **backtick**, which must be the very last character on each continued line:

```powershell
maxkcut --graphs G11 --k 2 `
  --solver SCS `
  --eps 1e-3 `
  --max-iters 3000 `
  --restarts 20 `
  --seed 2026 `
  --output reports/gset/g11-first.jsonl
```

**Do not copy the Bash backslash `\` as a line continuation in PowerShell.** It will be passed to `maxkcut` as a literal argument (e.g. `invalid int value: '\'`), and subsequent lines beginning with `--` will cause PowerShell syntax errors.

## Change rounding seeds without re-solving the SDP

If the first SDP solve completes and its matrix is safely cached, reuse it with another random seed and more restarts:

```powershell
maxkcut --graphs G11 --k 2 --restarts 100 --seed 2027 --output reports/gset/g11-more.jsonl
```

The cache is keyed by graph file hash and solver settings. Do not reuse the same output JSONL filename when rerunning, because the experiment runner **appends** records.

To compare SCS's indirect linear system solver, run a separate experiment:

```powershell
maxkcut --graphs G11 --k 2 --scs-indirect --output reports/gset/g11-indirect.jsonl
```

This is a new, separately fingerprinted SDP solve. It is not necessarily faster than the direct solver.

## View completed results

```powershell
Get-Content reports/gset/g11-first.jsonl
```

Each line is one JSON record for one rounding restart. Fields include cut weight, solver status, factorization time, the original SDP solve time, and whether the cached Gram matrix was reused.

## Tests, warnings and solver status

- `14 passed` means the small tests passed; **it does not mean G11 was solved**.
- SciPy's old `loadmat` default sparse type and `sparse.diags` dtype warnings are compatibility warnings, not test errors. Newer code chooses a compatible sparse-array format explicitly where supported and explicitly sets the Laplacian diagonal dtype.
- A returned `optimal_inaccurate` status is an approximate numerical result, **not** a certified integer optimum or validated dual SDP upper bound.
- G11 requires an 800×800 dense PSD matrix; the solver can consume much more memory than that matrix itself.
- If you want solver iteration output, add `--verbose`, but expect substantial terminal logging.

See the [performance guide](performance.md) for solver strategies and the [G-set reproduction runbook](https://github.com/heri-espino/maxkcut/blob/main/notes/experiments/gset-cvxpy-reproduction.md) for scientific interpretation.
