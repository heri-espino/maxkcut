"""Backward-compatible benchmark entry point.

Prefer `maxkcut --graphs G11 --inspect-only` or `python -m maxkcut`.
The Conda environment installs the package in editable mode.
"""
from maxkcut import (benchmark, cut_weight, edge_arrays, factorize, load_graph,
                     round_fj, round_spherical_kmeans, solve_sdp)
from maxkcut.cli import main

if __name__ == "__main__":
    main()
