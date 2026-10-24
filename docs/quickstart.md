# Quickstart

## Inspect input graph files before any optimization

```bash
maxkcut --graphs G11 G1 G22 --inspect-only
# Equivalent:
python -m maxkcut --graphs G11 --inspect-only
# Backward-compatible:
python scripts/gset_benchmark.py --graphs G11 --inspect-only
```

These commands read MATLAB graph data but do **not** solve the SDP.

## Solve one graph deliberately

```bash
maxkcut --graphs G11 --k 2 --restarts 10 \
  --solver SCS --eps 1e-3 --max-iters 3000 \
  --output reports/gset/g11-k2.jsonl
```

The `--max-n` guard defaults to 1000 vertices. Change it explicitly before attempting an instance larger than this, understanding that the solver may run out of memory.

For $k>2$ the SDP additionally imposes $X_{ij}\ge-1/(k-1)$.

## Use the Python library directly

```python
from pathlib import Path
import numpy as np
from maxkcut import load_graph, edge_arrays, cut_weight

adjacency = load_graph(Path("G-Set/G11.mat"))
edges = edge_arrays(adjacency)
labels = np.zeros(adjacency.shape[0], dtype=int)
print("n, m:", adjacency.shape[0], len(edges[0]))
print("All-in-one-class cut:", cut_weight(edges, labels))
```

To run a small SDP explicitly:

```python
import numpy as np
from scipy.sparse import csr_matrix
from maxkcut import solve_sdp, factorize, round_fj

triangle = csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
X, diagnostics = solve_sdp(triangle, k=2, solver="SCS", eps=1e-5, max_iters=5000)
vectors, minimum_eigenvalue = factorize(X)
labels = round_fj(vectors, 2, np.random.default_rng(2026))
print(labels, diagnostics["solver_status"])
```

**Reminder:** the primal SDP score is not an automatically verified upper bound; consult diagnostics and a dual certificate when reporting guarantees.

## Avoid unnecessary repeated SDP solves

The SDP matrix is cached by input checksum and solver settings. Different random
rounding seeds reuse the expensive matrix: `maxkcut --graphs G11 --k 2
--restarts 100 --seed 2027 --output reports/gset/g11-rounding-2027.jsonl`.

Use `--no-sdp-cache` to force a fresh solve, `--scs-indirect` to test SCS's
indirect linear system backend, or `--list-solvers` to inspect the available
solvers. **Save to a fresh JSONL file for every run**, because output appends.
See [performance and library choices](performance.md).
