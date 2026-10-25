"""No SDP solves: unit/regression tests for sparse G-set sweep."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

from maxkcut.archive import verify_archive
from maxkcut.io import edge_arrays
from maxkcut.local_search import improve_by_flips, random_flip_restart
from maxkcut.objective import cut_weight
from maxkcut.sweep import discover_graphs, inspect_gset, run_sweep
from maxkcut.sweep_cli import main


def _graph(weights):
    a = np.asarray(weights, dtype=float)
    return sparse.csr_matrix(a)


def _file(path, weights):
    savemat(path, {"A": _graph(weights)})
    return path


def test_signed_one_flip_local_search_is_monotone_and_locally_stable():
    a = _graph([[0, 2, -1], [2, 0, 3], [-1, 3, 0]])
    labels = np.array([0, 0, 0], dtype=np.int8)
    result, info = improve_by_flips(
        a, labels, np.random.default_rng(2026), max_sweeps=10
    )
    assert info["initial_cut"] == 0.
    assert info["cut_weight"] >= 0.
    assert info["termination"] == "local_optimum"
    assert info["sweeps"] <= 10
    edges = edge_arrays(a)
    final_score = cut_weight(edges, result)
    assert final_score == info["cut_weight"]
    for vertex in range(a.shape[0]):
        candidate = result.copy()
        candidate[vertex] = 1 - candidate[vertex]
        assert cut_weight(edges, candidate) <= final_score + 1e-10


def test_repeatable_signed_random_search_and_timeout():
    a = _graph([[0, -2, 3], [-2, 0, 1], [3, 1, 0]])
    one, stats1 = random_flip_restart(a, seed=11, max_sweeps=8)
    two, stats2 = random_flip_restart(a, seed=11, max_sweeps=8)
    np.testing.assert_array_equal(one, two)
    assert stats1["cut_weight"] == stats2["cut_weight"]
    output, stats = random_flip_restart(
        a, seed=11, max_sweeps=8, deadline=0.0,
    )
    assert stats["termination"] == "time_budget"
    assert len(output) == 3
    assert stats["cut_weight"] >= stats["initial_cut"]


def test_graph_discovery_is_numeric_and_inspection_has_no_solver(tmp_path):
    weights = [[0, 1], [1, 0]]
    for name in ["G10", "G2", "G1"]:
        _file(tmp_path / f"{name}.mat", weights)
    assert [p.stem for p in discover_graphs(tmp_path)] == ["G1", "G2", "G10"]
    report = inspect_gset(tmp_path)
    assert len(report) == 3
    assert all(row["n"] == 2 and row["m"] == 1 for row in report)


def test_batch_resumes_and_verifies_archives(monkeypatch, tmp_path):
    folder = tmp_path / "G-Set"
    folder.mkdir()
    for name in ["G1", "G2", "G10"]:
        _file(folder / f"{name}.mat", [[0, 1], [1, 0]])
    output = tmp_path / "reports"

    first = run_sweep(
        folder, output, max_graphs=2, restarts=3,
        max_sweeps=5, seconds_per_graph=30., seed=7,
    )
    assert first["completed"] == 2 and first["skipped"] == 0
    assert (output / "summary.csv").exists()
    for name in ["G1", "G2"]:
        result = json.loads(
            (output / "graphs" / f"{name}.json").read_text()
        )
        assert result["completed_restarts"] == 3
        assert result["status"] == "complete"
        verified = verify_archive(result["archive_path"])
        assert verified["methods"]["random_flip_local_search"]["verified_partitions"] == 3
        trials = (output / "trials" / f"{name}.jsonl").read_text().splitlines()
        assert len(trials) == 3
        assert all(json.loads(row)["sdp_used"] is False for row in trials)

    # Resuming the same settings should do no numerical work at all.
    import maxkcut.sweep as sweep

    def forbidden(*args, **kwargs):
        raise AssertionError("resume must skip completed verified experiments")

    monkeypatch.setattr(sweep, "random_flip_restart", forbidden)
    second = run_sweep(
        folder, output, max_graphs=2, restarts=3,
        max_sweeps=5, seconds_per_graph=30., seed=7,
    )
    assert second["skipped"] == 2 and second["completed"] == 0
    # New choice of budget in same output folder must fail before any trials.
    with pytest.raises(ValueError, match="fresh --output-dir"):
        run_sweep(folder, output, restarts=4, max_graphs=1)


def test_tampering_forces_rerun(monkeypatch, tmp_path):
    folder = tmp_path / "G-Set"
    folder.mkdir()
    graph_file = _file(folder / "G11.mat", [[0, 1], [1, 0]])
    output = tmp_path / "results"
    run_sweep(folder, output, restarts=1, max_sweeps=3)
    previous = json.loads((output / "graphs" / "G11.json").read_text())
    assert previous["input_sha256"]
    # Same filename, different weighted instance must not be silently skipped.
    _file(graph_file, [[0, 2], [2, 0]])
    next_run = run_sweep(folder, output, restarts=1, max_sweeps=3)
    assert next_run["completed"] == 1 and next_run["skipped"] == 0
    updated = json.loads((output / "graphs" / "G11.json").read_text())
    assert previous["input_sha256"] != updated["input_sha256"]


def test_sweep_cli_inspects_without_output_dir(monkeypatch, capsys, tmp_path):
    folder = tmp_path / "G-Set"
    folder.mkdir()
    _file(folder / "G1.mat", [[0, 1], [1, 0]])
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "maxkcut-sweep", "--gset-dir", str(folder),
        "--output-dir", str(output), "--inspect-only",
    ])
    main()
    assert json.loads(capsys.readouterr().out)[0]["graph"] == "G1"
    assert not output.exists()


def test_sweep_records_failed_graph_and_continues(tmp_path):
    folder = tmp_path / "G-Set"
    folder.mkdir()
    (folder / "G1.mat").write_bytes(b"invalid matlab bytes")
    _file(folder / "G2.mat", [[0, 1], [1, 0]])
    result = run_sweep(
        folder, tmp_path / "results", restarts=1, max_sweeps=3,
        seconds_per_graph=10.,
    )
    assert result["failed"] == 1 and result["completed"] == 1
    failed = json.loads(
        (tmp_path / "results" / "graphs" / "G1.json").read_text()
    )
    assert failed["status"] == "failed"
    assert failed["error"]


def test_soft_budget_result_is_saved_and_counted_without_false_failure(
    monkeypatch, tmp_path
):
    """A time-budgeted feasible cut must never become KeyError/failed."""
    folder = tmp_path / "G-Set"
    folder.mkdir()
    _file(folder / "G1.mat", [[0, 1], [1, 0]])
    output = tmp_path / "reports"

    import maxkcut.sweep as sweep

    def limited_restart(graph, *, seed, max_sweeps, deadline):
        return np.array([0, 1], dtype=np.int8), {
            "initial_cut": 0.,
            "cut_weight": 1.,
            "flips": 1,
            "sweeps": 1,
            "termination": "time_budget",
            "seconds": 0.,
        }

    monkeypatch.setattr(sweep, "random_flip_restart", limited_restart)
    first = run_sweep(
        folder, output, restarts=3, max_sweeps=2,
        seconds_per_graph=30.,
    )
    assert first["completed"] == 0
    assert first["budget_limited"] == 1
    assert first["failed"] == 0

    record = json.loads((output / "graphs" / "G1.json").read_text())
    assert record["status"] == "budget_limited"
    assert record["completed_restarts"] == 1
    verification = verify_archive(record["archive_path"])
    assert verification["status"] == "budget_limited"
    assert verification["methods"]["random_flip_local_search"]["best_cut"] == 1.

    def forbidden(*args, **kwargs):
        raise AssertionError("verified budget-limited experiment must be skipped")

    monkeypatch.setattr(sweep, "random_flip_restart", forbidden)
    resumed = run_sweep(
        folder, output, restarts=3, max_sweeps=2,
        seconds_per_graph=30.,
    )
    assert resumed["skipped"] == 1 and resumed["failed"] == 0
