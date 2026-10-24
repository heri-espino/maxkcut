# maxkcut

Python research library for **Max-Cut** and **Max-$k$-Cut** using semidefinite relaxations, randomized rounding, and spherical $k$-means rounding.

The project is motivated by Ángel Eduardo Rodríguez Fernández's 2021 dissertation (Chapters 6–7). It is experimental research software: **solver success does not imply an exact optimum or a new approximation guarantee**.

## Install with Conda

```bash
git clone https://github.com/heri-espino/maxkcut.git
cd maxkcut
conda env create -f environment.yml
conda activate maxkcut
python -m pytest -q
```

`environment.yml` installs the package **editable** from `src/maxkcut/` and includes CVXPY, SCS, NumPy, SciPy, pytest, Sphinx, Furo and MyST. If the environment already exists, use `conda env update -f environment.yml --prune`. Pip fallback: `python -m pip install -e '.[dev]'`.

## Inspect graphs first (no SDP solve)

```bash
maxkcut --graphs G11 G1 G22 --inspect-only
# or python -m maxkcut --graphs G11 --inspect-only
```

## Run a single controlled experiment

```bash
maxkcut --graphs G11 --k 2 --restarts 10 --solver SCS \
  --eps 1e-3 --max-iters 3000 --output reports/gset/g11-k2.jsonl
```

**Windows PowerShell users:** Bash uses a trailing backslash to continue long commands, but PowerShell does not. Use the one-line command below, or a PowerShell backtick as the last character of each continued line:

```powershell
maxkcut --graphs G11 --k 2 --solver SCS --eps 1e-3 --max-iters 3000 --restarts 20 --seed 2026 --output reports/gset/g11-first.jsonl
```

See the [Windows PowerShell guide](docs/windows-powershell.md) for the complete run, caching, solver comparison and result inspection.

The default `--max-n 1000` guard prevents unexpectedly large runs. The legacy entry point `python scripts/gset_benchmark.py` still works once the environment is active. Expensive GitHub Actions experiments remain **manual-only**.

## Import the library

```python
from pathlib import Path
import numpy as np
from maxkcut import load_graph, edge_arrays, cut_weight

graph = load_graph(Path("G-Set/G11.mat"))
edges = edge_arrays(graph)
print(graph.shape, len(edges[0]))
print(cut_weight(edges, np.zeros(graph.shape[0], dtype=int)))
```

Core public API: `load_graph`, `edge_arrays`, `cut_weight`, `solve_sdp`, `factorize`, `round_fj`, `round_spherical_kmeans`, `benchmark`.

## Performance tuning and reproducible SDP caching

**Numerical PSD note:** SCS may return a small negative Gram eigenvalue at
a loose tolerance. The rounding factorization now records and clips *small*
negative eigenvalues with a warning; it still rejects large violations. This
does **not** certify an SDP bound. To insist on a fresh, more accurate solve,
use `--eps 1e-5 --max-iters 20000 --no-sdp-cache` (potentially slower).
See [Windows PowerShell troubleshooting](docs/windows-powershell.md).

The first G11 run saves an SDP Gram matrix in `reports/gset/sdp_cache/`.
Re-running with a different seed or restart count **reuses the same SDP**
when the graph file and solver settings are unchanged.

```bash
maxkcut --list-solvers
maxkcut --graphs G11 --k 2 --restarts 10 --output reports/gset/g11-first.jsonl
maxkcut --graphs G11 --k 2 --restarts 100 --seed 2027 --output reports/gset/g11-second.jsonl
# Separately benchmark SCS iterative KKT mode
maxkcut --graphs G11 --k 2 --scs-indirect --output reports/gset/g11-indirect.jsonl
```

See [the performance guide](docs/performance.md) for bottlenecks, cache safety,
timings, and the limitations of CVXPY/SCS.

## CPU threading and optional NVIDIA GPU

Your G11 SDP uses dense semidefinite-cone projections. The number of logical
processors does not imply SCS automatically uses them all. Check hardware
visibility and use a controlled BLAS thread limit:

```powershell
maxkcut --hardware-info
maxkcut --graphs G11 --k 2 --solver SCS --eps 1e-5 --max-iters 20000 --cpu-threads 8 --no-sdp-cache --restarts 20 --output reports/gset/g11-cpu8.jsonl
```

`--cpu-threads` affects recognized BLAS/LAPACK thread pools, **not** every
solver kernel, and may be slower with too many threads. A separate
`--scs-cudss` flag requests an optional **CUDA/cuDSS-enabled custom SCS build**
for compatible NVIDIA GPUs; standard Conda SCS does not guarantee this
backend. It is not an automatic GPU speedup, particularly for SDP cone
eigenprojections.

The same `--cpu-threads N` flag now also sets SDPA's native
`numThreads=N` option. This matters for Windows users who saw
`Memory Exhausted (bad_alloc)` despite 64 GB RAM: limiting native workers
may reduce resource demands but cannot guarantee a successful SDP solve.
See the [SDPA/Windows troubleshooting section](docs/windows-powershell.md).

See the [CPU/GPU performance guide](docs/gpu-cpu-performance.md) and
[Windows PowerShell guide](docs/windows-powershell.md).

## Documentation and agent instructions

```bash
sphinx-build -b html docs docs/_build/html -W --keep-going
```

Read `docs/_build/html/index.html` for the Sphinx/Furo site, or `docs/index.md` for source. For contributors and AI agents, **start with [AGENTS.md](AGENTS.md)** and the [agent guide](docs/agent-guide.md).

- [Source-grounded research background](notes/background/README.md) — SDP mathematics, Chapter 7, prior art and open questions.
- [G-set reproduction runbook](notes/experiments/gset-cvxpy-reproduction.md) — inputs, protocol and numerical limitations.
- [Package architecture](docs/architecture.md) and [API](docs/api.md).

The `G-Set/` files are associated with Chapter 6's Max-Cut benchmark, **not** the 300 A/B/C/D random graphs used in Chapter 7. No full G-set SDP computation is claimed by this packaging refactor.
