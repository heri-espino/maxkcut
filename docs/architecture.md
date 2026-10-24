# Project architecture

```text
maxkcut/
├── AGENTS.md                     # canonical instructions for coding agents
├── environment.yml               # Conda environment + pip editable install
├── pyproject.toml                # src layout and package metadata
├── src/maxkcut/
│   ├── __init__.py               # stable public API
│   ├── __main__.py               # python -m maxkcut
│   ├── hardware.py               # CPU BLAS thread limits + hardware visibility
│   ├── archive.py                # compact graph/partition/reproducibility snapshots
│   ├── cache.py                  # SHA256/solver-fingerprinted SDP matrix reuse
│   ├── cli.py                    # argument parsing, inspecting graphs
│   ├── io.py                     # MATLAB matrices -> sparse weighted graph
│   ├── objective.py              # exact weighted cut evaluator
│   ├── sdp.py                    # semidefinite relaxation via CVXPY
│   ├── rounding.py               # Gram factorization + random / k-means rounding
│   └── experiments.py            # runner, reproducible seed protocol, JSONL
├── scripts/
│   └── gset_benchmark.py         # backward-compatible CLI/import wrapper
├── tests/                        # synthetic tests and optional tiny SDP test
├── docs/                         # Sphinx documentation (Furo + MyST)
├── notes/
│   ├── background/               # source-grounded theory notes
│   └── experiments/              # experimental designs, results, limitations
├── G-Set/                        # checked-in input data, do not modify casually
└── .github/workflows/            # heavy numerical job explicitly manual
```

## Data and dependency flow

```text
load_graph(...) -> sparse W -> edge_arrays(W) -> cut_weight(...)
                             |
                             +-> solve_sdp(W, k, ...) -> Gram matrix X
                                   -> factorize(X) -> row vectors V
                                          -> GW/FJ round_fj(V, k, ...)
                                          -> round_spherical_kmeans(V, k, ...)
                                              -> cut_weight(edges, labels)
```

The graph objective is always scored in `objective.py`, never substituted with k-means inertia. Spherical k-means only provides a *rounding heuristic*.

## Module boundaries

- **`io`:** no CVXPY dependency, input shape/weight validation, no network.
- **`sdp`:** no CLI parsing or file writes; heavy solve only when called explicitly.
- **`rounding`:** seeded NumPy generators passed by callers; no implicit global RNG.
- **`experiments`:** writes per-restart JSONL and logs solver diagnostics. Never mix data generation with numerical problem formulation.
- **`cli`:** user-facing flags, guards against accidental high-$n$ runs.
- **`scripts/`:** compatibility only; all new functionality should live in `src/maxkcut/`.

## Stability contract

Preserve the importable names `load_graph`, `edge_arrays`, `cut_weight`, `factorize`, `round_fj`, `round_spherical_kmeans`, `solve_sdp`, and `benchmark` unless a versioned deprecation is justified. Keep `maxkcut`, `python -m maxkcut`, and `python scripts/gset_benchmark.py` consistent.

**No experimental performance claim can be inferred from static code or tests alone.** A successful small SDP smoke test does not prove the feasibility of large G-set instances.
