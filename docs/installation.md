# Installation

## Conda (primary workflow)

From the repository root:

```bash
git clone https://github.com/heri-espino/maxkcut.git
cd maxkcut
conda env create -f environment.yml
conda activate maxkcut
python -c "import maxkcut; print(maxkcut.__version__)"
python -m pytest -q
```

The Conda environment uses **conda-forge** and `pip: -e .` for an **editable** installation: edits inside `src/maxkcut/` take effect without reinstalling. It supplies `numpy`, `scipy`, `cvxpy`, `scs`, `pytest`, Sphinx, Furo, and MyST.

To update an existing environment after changes to the manifest:

```bash
conda env update -f environment.yml --prune
conda activate maxkcut
```

Alternative pip installation: `python -m pip install -e '.[dev]'`. Pip is not the recommended path when using Conda.

## Build the documentation locally

```bash
sphinx-build -b html docs docs/_build/html -W --keep-going
```

Open `docs/_build/html/index.html`. No hosting or publication is implied merely by building HTML.

## Solver and memory

CVXPY is a **modeling layer**, not a numerical SDP solver. The default solver in the CLI is SCS. For larger graphs, PSD matrices have $n(n+1)/2$ independent entries, and solver memory may greatly exceed raw $n^2$ matrix storage.

Start with `--inspect-only` or a synthetic small-graph smoke test. **Do not** launch G22 or batch every G-set file merely to check installation. An `optimal_inaccurate` status should be treated as diagnostic evidence, not a certified optimum.

## Common problems

- `ModuleNotFoundError: maxkcut`: activate the environment and run `python -m pip install -e .`.
- `ModuleNotFoundError: cvxpy`: update `environment.yml` and verify `conda list cvxpy`.
- Mat file cannot be decoded: check that the file uses the supported MATLAB v5 structure; a v7.3 (HDF5) file may need a separate loader.
- Numerical failure or exhausted RAM: lower $n$, check solver logs and tolerances, or use a more scalable alternative **with clearly labeled non-equivalence**.
