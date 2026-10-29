"""Compiled EVA preserves the reference sequential mathematical update.

These tests are opt-in if Numba is not installed; NumPy EVA always works.
Install supported Numba with: pip install 'numba>=0.67,<0.69'.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("numba", minversion="0.67")

from maxkcut.eva import load_eva_factor, solve_eva, vector_objective
from maxkcut.eva_numba import solve_eva_numba
from maxkcut.eva_cli import run_one
from maxkcut.eva_rounding_cli import round_saved_factor


def _signed_graph():
    a = np.array([
        [0, 2, -1, 0, 0, 0],
        [2, 0, 3, 1, -2, 0],
        [-1, 3, 0, 4, 1, 0],
        [0, 1, 4, 0, 2, 0],
        [0, -2, 1, 2, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.float64)
    return sparse.csr_matrix(a)


@pytest.mark.parametrize("seed", [1, 2026, 42])
@pytest.mark.parametrize("tolerance", [1e-3, 1e-5])
def test_compiled_matches_numpy_objective_and_factor(seed, tolerance):
    """At meaningful EVA tolerances, the two sequential algorithms agree."""
    graph = _signed_graph()
    ref_y, ref = solve_eva(
        graph, seed=seed, rank=4, max_sweeps=40, tolerance=tolerance,
    )
    fast_y, fast = solve_eva_numba(
        graph, seed=seed, rank=4, max_sweeps=40, tolerance=tolerance,
    )
    # Different floating point reduction order causes sub-ulp variation.
    np.testing.assert_allclose(fast_y, ref_y, rtol=1e-9, atol=1e-9)
    assert fast["sweeps"] == ref["sweeps"]
    assert fast["converged"] == ref["converged"]
    assert abs(fast["objective"] - ref["objective"]) < 1e-7
    assert abs(fast["max_angle"] - ref["max_angle"]) < 1e-7
    assert abs(fast["objective"] - vector_objective(graph, fast_y)) < 1e-7


@pytest.mark.parametrize("seed", [1, 2026, 42])
def test_compiled_strict_threshold_same_fixed_sweep_trajectory(seed):
    """For 1e-8 angular thresholds, compare fixed trajectories, not stop count.

    acos(dot) near dot=1 has resolution only around 1.49e-8 rad in
    double precision; one backend may see cosine==1 while the other
    sees 0.9999999999999999 and run another sweep. This is a stopping
    diagnostic issue, not a materially different optimization step.
    """
    graph = _signed_graph()
    # Five sweeps precede convergence for each seed and hence eliminate
    # the ill-conditioned stop test from the algorithmic parity check.
    ref_y, ref = solve_eva(
        graph, seed=seed, rank=4, max_sweeps=5, tolerance=1e-8,
    )
    fast_y, fast = solve_eva_numba(
        graph, seed=seed, rank=4, max_sweeps=5, tolerance=1e-8,
    )
    assert ref["sweeps"] == fast["sweeps"] == 5
    np.testing.assert_allclose(fast_y, ref_y, rtol=1e-12, atol=1e-12)
    assert abs(fast["objective"] - ref["objective"]) < 1e-9
    assert abs(fast["objective"] - vector_objective(graph, fast_y)) < 1e-9

    # At the machine-resolution boundary the implementations may stop at
    # different iterations. Check the final numerical solutions instead.
    ref_y, ref = solve_eva(
        graph, seed=seed, rank=4, max_sweeps=40, tolerance=1e-8,
    )
    fast_y, fast = solve_eva_numba(
        graph, seed=seed, rank=4, max_sweeps=40, tolerance=1e-8,
    )
    np.testing.assert_allclose(fast_y, ref_y, rtol=1e-8, atol=1e-8)
    assert abs(fast["objective"] - ref["objective"]) < 1e-8
    assert abs(fast["objective"] - vector_objective(graph, fast_y)) < 1e-8
    assert np.isfinite(fast["max_angle"])
    assert np.isfinite(ref["max_angle"])


def test_compiled_checkpoint_resume_and_backend_isolation(tmp_path):
    graph = _signed_graph()
    path = tmp_path / "G11.npz"
    first_y, first = solve_eva_numba(
        graph, rank=4, tolerance=1e-9, max_sweeps=3,
        checkpoint_every=1, checkpoint_path=path,
    )
    assert path.is_file()
    saved_y, saved = load_eva_factor(path)
    np.testing.assert_allclose(first_y, saved_y)
    assert saved["identity"]["algorithm"] == "eva-sequential-numba-v1"
    further_y, further = solve_eva_numba(
        graph, rank=4, tolerance=1e-9, max_sweeps=8,
        checkpoint_every=1, checkpoint_path=path,
    )
    fresh_y, fresh = solve_eva_numba(
        graph, rank=4, tolerance=1e-9, max_sweeps=8,
    )
    np.testing.assert_allclose(further_y, fresh_y, rtol=1e-10, atol=1e-10)
    assert abs(further["objective"] - fresh["objective"]) < 1e-8
    with pytest.raises(ValueError, match="mismatch"):
        solve_eva(graph, rank=4, tolerance=1e-9,
                  max_sweeps=9, checkpoint_path=path)


def test_numba_cli_archives_and_reusable_factors(tmp_path):
    graph_path = tmp_path / "G11.txt"
    graph_path.write_text(
        "5 7\n1 2 2\n1 3 -1\n1 4 1\n2 3 2\n"
        "2 5 1\n3 4 1\n4 5 3\n", encoding="utf8",
    )
    root = tmp_path / "eva-numba"
    args = SimpleNamespace(
        max_n=20, output_dir=root, backend="numba", rank=None,
        tolerance=1e-6, max_sweeps=12, seed=2026,
        checkpoint_every=3, rounding_seed=2026,
        restarts=2, convex_dir=None, cpu_threads=1,
        push=False, push_branch=None, push_every=0,
    )
    result = run_one(graph_path, args)
    assert result["provenance"]["optimizer_backend"] == "numba"
    factor, metadata = load_eva_factor(root / "factors" / "G11.npz")
    assert metadata["identity"]["algorithm"] == "eva-sequential-numba-v1"
    assert factor.shape[0] == 5
    again = run_one(graph_path, args)
    assert again["archive_path"] == result["archive_path"]
    extensions = tmp_path / "rounding"
    rounded = round_saved_factor(
        graph_path, source=root, dest=extensions,
        methods=["FJ_EVA"], first=2, restarts=3,
    )
    assert rounded["methods"]["FJ_EVA"]["count"] == 5
    assert (extensions / "factor-snapshots").is_dir()


def test_different_backends_cannot_reuse_same_checkpoint(tmp_path):
    graph = _signed_graph()
    path = tmp_path / "G.npz"
    solve_eva(graph, rank=4, max_sweeps=2, checkpoint_path=path)
    with pytest.raises(ValueError, match="mismatch"):
        solve_eva_numba(graph, rank=4, max_sweeps=2, checkpoint_path=path)
