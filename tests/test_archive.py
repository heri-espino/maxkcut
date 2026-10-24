"""Archive regression tests require no CVXPY and no large graph solver."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

from maxkcut.archive import (
    load_method, method_archive_path, new_archive, new_run_id,
    save_method, update_manifest, verify_archive,
)
import maxkcut.experiments as experiments
from maxkcut.cli import main


def _test_edges():
    # Includes a negative weight; a valid cut need not include each edge.
    return (
        np.array([0, 0, 1], dtype=int),
        np.array([1, 2, 2], dtype=int),
        np.array([2., -1., 3.]),
    )


def test_two_way_roundtrip_preserves_all_labels_and_signed_cut(tmp_path):
    graph_path = tmp_path / "G11.mat"
    graph_path.write_bytes(b"original external input")
    archive = new_archive(
        tmp_path / "artifacts", new_run_id(), graph_path, 3,
        _test_edges(), 2, {"seed": 123}, jsonl_output=tmp_path / "runs.jsonl",
    )
    labels = [
        np.array([0, 1, 1]),
        np.array([1, 0, 1]),
        np.array([0, 0, 0]),
    ]
    scores = [2., 2., 0.]
    summary = save_method(archive, "FJ", labels, scores, k=2)
    assert summary["best_cut"] == 2.
    assert summary["best_restart"] == 0
    assert summary["restarts"] == 3
    assert summary["sample_std_cut"] is not None
    saved, raw_scores = load_method(method_archive_path(archive, "FJ"))
    np.testing.assert_array_equal(saved, np.stack(labels))
    np.testing.assert_allclose(raw_scores, scores)
    update_manifest(archive, status="complete")
    verification = verify_archive(archive["path"])
    assert verification["status"] == "complete"
    assert verification["methods"]["FJ"]["verified_partitions"] == 3

    with np.load(archive["path"] / "graph.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["weights"], [2., -1., 3.])
    manifest = json.loads((archive["path"] / "manifest.json").read_text())
    assert manifest["graph_sha256"]
    assert manifest["settings"]["seed"] == 123


def test_general_k_roundtrip_and_mismatched_scores_rejected(tmp_path):
    file = tmp_path / "G10.mat"
    file.write_bytes(b"graph")
    archive = new_archive(
        tmp_path / "archives", new_run_id(), file, 3,
        _test_edges(), 3, {"seed": 1}, jsonl_output=tmp_path / "out.jsonl"
    )
    labels = [np.array([0, 1, 2]), np.array([1, 1, 0])]
    save_method(archive, "spherical_kmeans", labels, [4., 100.], k=3)
    stored, values = load_method(
        method_archive_path(archive, "spherical_kmeans")
    )
    assert stored.shape == (2, 3)
    np.testing.assert_allclose(values, [4, 100])
    with pytest.raises(ValueError, match="do not match"):
        verify_archive(archive["path"])


def test_archive_records_incomplete_state_before_methods(tmp_path):
    file = tmp_path / "G3.mat"
    file.write_bytes(b"graph")
    archive = new_archive(
        tmp_path / "archives", new_run_id(), file, 3,
        _test_edges(), 2, {}, jsonl_output=tmp_path / "out.jsonl"
    )
    manifest = json.loads((archive["path"] / "manifest.json").read_text())
    assert manifest["status"] == "started"
    assert manifest["methods"] == {}


def test_benchmark_artifacts_can_be_verified_without_solver(monkeypatch, tmp_path):
    graph_path = tmp_path / "G11.mat"
    a = sparse.csr_matrix([
        [0., 2., -1.],
        [2., 0., 3.],
        [-1., 3., 0.],
    ])
    savemat(graph_path, {"A": a})

    def mock_sdp(a, k, solver, eps, max_iters, verbose=False, **kwargs):
        return np.eye(a.shape[0]), {
            "sdp_seconds": 3., "solver_status": "optimal",
            "sdp_objective_primal": 4.,
        }

    monkeypatch.setattr(experiments, "solve_sdp", mock_sdp)
    output = tmp_path / "gset" / "experiment.jsonl"
    experiments.benchmark(
        [graph_path], [2], output, restarts=2, use_cache=False,
        seed=42, argv=["maxkcut", "--graphs", "G11"]
    )
    cases = list((tmp_path / "gset" / "artifacts").glob("*/G11-k2"))
    assert len(cases) == 1
    result = verify_archive(cases[0])
    assert result["status"] == "complete"
    assert sorted(result["methods"]) == ["FJ", "K-Means2N", "K-MeansNM"]
    assert all(row["verified_partitions"] == 2 for row in result["methods"].values())
    rows = [
        json.loads(line) for line in output.read_text().splitlines()
    ]
    assert len(rows) == 6
    assert all(Path(row["partition_archive"]).is_file() for row in rows)
    assert all(row["artifact_run_id"] for row in rows)
    assert all(row["partition_index"] in (0, 1) for row in rows)

    monkeypatch.setattr(sys, "argv", [
        "maxkcut", "--verify-archive", str(cases[0]),
    ])
    main()


def test_disabling_archive_preserves_legacy_jsonl(monkeypatch, tmp_path):
    graph_path = tmp_path / "G11.mat"
    savemat(graph_path, {"A": sparse.csr_matrix([[0., 1.], [1., 0.]])})

    def mock_sdp(a, k, solver, eps, max_iters, verbose=False, **kwargs):
        return np.eye(a.shape[0]), {
            "sdp_seconds": 1., "solver_status": "optimal",
        }

    monkeypatch.setattr(experiments, "solve_sdp", mock_sdp)
    output = tmp_path / "gset" / "runs.jsonl"
    experiments.benchmark(
        [graph_path], [2], output, restarts=1,
        use_cache=False, save_artifacts=False,
    )
    assert not (output.parent / "artifacts").exists()
    entry = json.loads(output.read_text().splitlines()[0])
    assert entry["partition_archive"] is None
