"""Native SDPA worker count regression tests, with no large SDP solve."""
from __future__ import annotations

import json
import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

import maxkcut.experiments as experiments
from maxkcut.sdp import solve_sdp


def test_sdpa_num_threads_forwarded_to_cvxpy(monkeypatch):
    cp = pytest.importorskip("cvxpy")
    passed = []

    def fake_solve(self, **kwargs):
        passed.append(kwargs)
        raise RuntimeError("solve was intercepted before allocating memory")

    monkeypatch.setattr(cp, "installed_solvers", lambda: ["SDPA"])
    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    adjacency = sparse.csr_matrix([[0, 1], [1, 0]], dtype=float)

    with pytest.raises(RuntimeError, match="intercepted"):
        solve_sdp(adjacency, 2, "SDPA", 1e-3, 3000,
                  sdpa_num_threads=4)
    assert passed[-1]["solver"] == "SDPA"
    assert passed[-1]["numThreads"] == 4

    with pytest.raises(RuntimeError, match="intercepted"):
        solve_sdp(adjacency, 2, "SDPA", 1e-3, 3000)
    assert "numThreads" not in passed[-1]


def test_sdpa_num_threads_rejected_for_wrong_solver_or_value():
    adjacency = sparse.csr_matrix([[0., 1.], [1., 0.]])
    with pytest.raises(ValueError, match="positive integer"):
        solve_sdp(adjacency, 2, "SDPA", 1e-3, 3000,
                  sdpa_num_threads=0)
    with pytest.raises(ValueError, match="positive integer"):
        solve_sdp(adjacency, 2, "SCS", 1e-3, 3000,
                  sdpa_num_threads=4)


def test_benchmark_passes_native_sdpa_threads_without_sdp(monkeypatch, tmp_path):
    graph = tmp_path / "G11.mat"
    savemat(graph, {"A": sparse.csr_matrix([[0, 1], [1, 0]])})
    kwargs_seen = []

    def fake_solver(a, k, solver, eps, max_iters, verbose=False, **kwargs):
        kwargs_seen.append(kwargs)
        return np.eye(a.shape[0]), {
            "sdp_seconds": 1., "solver_status": "optimal",
            "sdp_objective_primal": 1.,
        }

    monkeypatch.setattr(experiments, "solve_sdp", fake_solver)
    result = tmp_path / "small.jsonl"
    experiments.benchmark(
        [graph], [2], result, solver="SDPA", cpu_threads=4,
        restarts=1, use_cache=False,
    )
    assert kwargs_seen[0]["sdpa_num_threads"] == 4
    entries = [json.loads(line) for line in result.read_text().splitlines()]
    assert len(entries) == 3
    assert all(entry["cpu_threads_requested"] == 4 for entry in entries)
