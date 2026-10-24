# maxkcut

Research on semidefinite relaxations and rounding methods for Max-$k$-Cut.

Start with the [source-grounded research notes](notes/background/README.md), which review Chapter 7 of Rodríguez Fernández's 2021 dissertation, the Frieze–Jerrum and clustering rounding methods, benchmarks, limitations, and related literature.

The existing `G-Set/` directory is a separate Max-Cut benchmark collection; it is **not** the thesis chapter's A/B/C/D random-graph benchmark.

## G-set SDP experiments

A [CVXPY reproduction runbook](notes/experiments/gset-cvxpy-reproduction.md) and the implementation in [`scripts/gset_benchmark.py`](scripts/gset_benchmark.py) are available on this branch. Start by validating the MATLAB files:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python scripts/gset_benchmark.py --graphs G11 G1 G22 --inspect-only
```

To attempt the real G11 Max-Cut SDP, run `python scripts/gset_benchmark.py --graphs G11 --k 2 --restarts 10`. Runs may require substantial RAM and CPU; heavy GitHub Actions runs are manual-only. No full G-set run has been claimed or executed in this PR.
