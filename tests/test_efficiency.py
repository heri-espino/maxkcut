"""Fast tests for disk-backed SDP reuse; never run a real large SDP."""
import json

import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

from maxkcut.cache import (
    cache_identity,
    cache_path,
    load_cached_sdp,
    save_cached_sdp,
)
import maxkcut.experiments as experiments


def test_cache_round_trip_and_graph_fingerprint(tmp_path):
    graph = tmp_path / "G11.mat"
    savemat(graph, {"A": sparse.csr_matrix([[0, 1], [1, 0]])})
    identity = cache_identity(
        graph, k=2, solver="SCS", eps=1e-3, max_iters=3000,
        scs_indirect=False
    )
    path = cache_path(graph, tmp_path / "cache", identity)
    assert load_cached_sdp(path, identity, n=2) is None
    save_cached_sdp(path, identity, np.eye(2), {"sdp_seconds": 3.0})
    x, info = load_cached_sdp(path, identity, n=2)
    np.testing.assert_allclose(x, np.eye(2))
    assert info["sdp_seconds"] == 3.0
    assert path.exists()

    another = cache_identity(
        graph, k=2, solver="SCS", eps=1e-3, max_iters=3000,
        scs_indirect=True
    )
    assert cache_path(graph, tmp_path / "cache", another) != path
    savemat(graph, {"A": sparse.csr_matrix([[0, 2], [2, 0]])})
    changed = cache_identity(
        graph, k=2, solver="SCS", eps=1e-3, max_iters=3000,
        scs_indirect=False
    )
    assert cache_path(graph, tmp_path / "cache", changed) != path


def test_cache_rejects_wrong_size(tmp_path):
    path = tmp_path / "sdp.npz"
    identity = {"k": 2}
    save_cached_sdp(path, identity, np.eye(2), {"sdp_seconds": 1.0})
    with pytest.raises(ValueError, match="Invalid cached"):
        load_cached_sdp(path, identity, n=3)


def test_benchmark_reuses_single_sdp_across_runs(monkeypatch, tmp_path):
    graph = tmp_path / "G11.mat"
    savemat(graph, {"A": sparse.csr_matrix([[0, 1], [1, 0]])})
    solve_calls = []

    def stub_solver(a, k, solver, eps, max_iters, verbose=False, *,
                    scs_indirect=False, compute_eigenvalues=True):
        solve_calls.append((k, scs_indirect, compute_eigenvalues))
        return np.eye(a.shape[0]), {
            "sdp_seconds": 12.5, "solver_status": "optimal",
            "sdp_objective_primal": 1.0
        }

    monkeypatch.setattr(experiments, "solve_sdp", stub_solver)
    settings = {
        "paths": [graph], "k_values": [2], "restarts": 2,
        "cache_dir": tmp_path / "cache", "seed": 2026,
        "solver": "SCS",
    }
    output1 = tmp_path / "one.jsonl"
    experiments.benchmark(output=output1, **settings)
    output2 = tmp_path / "two.jsonl"
    experiments.benchmark(output=output2, **settings)

    assert solve_calls == [(2, False, False)]
    first = [json.loads(line) for line in output1.read_text().splitlines()]
    second = [json.loads(line) for line in output2.read_text().splitlines()]
    assert len(first) == len(second) == 6  # 3 methods * 2 restarts
    assert all(row["cache_hit"] is False for row in first)
    assert all(row["cache_hit"] is True for row in second)
    assert all(row["sdp_seconds"] == 0 for row in second)
    assert all(row["sdp_original_solve_seconds"] == 12.5 for row in second)
    assert all(row["factorization_seconds"] >= 0 for row in second)
    assert [
        (row["method"], row["restart"], row["cut_weight"])
        for row in first
    ] == [
        (row["method"], row["restart"], row["cut_weight"])
        for row in second
    ]


def test_benchmark_cache_can_be_disabled(monkeypatch, tmp_path):
    graph = tmp_path / "G11.mat"
    savemat(graph, {"A": sparse.csr_matrix([[0, 1], [1, 0]])})
    calls = []

    def fake(a, k, solver, eps, max_iters, verbose=False, **kwargs):
        calls.append(k)
        return np.eye(a.shape[0]), {"sdp_seconds": 1.2,
                                     "solver_status": "optimal"}
    monkeypatch.setattr(experiments, "solve_sdp", fake)
    for index in range(2):
        experiments.benchmark(
            [graph], [2], tmp_path / f"uncached{index}.jsonl",
            restarts=1, use_cache=False
        )
    assert calls == [2, 2]
