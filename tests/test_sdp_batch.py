"""Full-SDP batch orchestration tests; no large SDP solve or CUDA required."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

import maxkcut.sdp_batch as batch
from maxkcut.archive import (
    new_archive, new_run_id, save_method, update_manifest,
)
from maxkcut.io import edge_arrays, load_graph


def _create_graph(tmp_path: Path, name: str, n: int = 3) -> Path:
    folder = tmp_path / "data"
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.txt"
    path.write_text(f"{n} 1\n1 2 1\n", encoding="utf8")
    return path


def _write_complete_sdp_archive(graph: Path, jsonl: Path, restarts: int = 2):
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    adjacency = load_graph(graph)
    edges = edge_arrays(adjacency)
    n = adjacency.shape[0]
    archive = new_archive(
        jsonl.parent / "artifacts", new_run_id(), graph, n, edges, 2,
        {"solver": "SCS"}, jsonl_output=jsonl,
    )
    rows = []
    scores = [1.0] * restarts
    labels = [np.r_[0, 1, np.zeros(n-2, dtype=int)] for _ in range(restarts)]
    for method in batch.EXPECTED:
        save_method(archive, method, labels, scores, k=2)
        for index in range(restarts):
            rows.append({
                "graph": graph.stem, "n": n, "k": 2,
                "method": method, "restart": index,
                "solver": "SCS", "solver_status": "optimal",
                "cut_weight": 1., "artifact_run_id": archive["manifest"]["run_id"],
            })
    update_manifest(archive, status="complete")
    jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf8",
    )
    return archive["path"]


def test_inventory_only_reads_canonical_headers(tmp_path):
    _create_graph(tmp_path, "G10", 20000)
    _create_graph(tmp_path, "G2", 800)
    rows = batch.inventory(tmp_path / "data")
    assert [row["graph"] for row in rows] == ["G2", "G10"]
    assert rows[0]["vertices"] == 800
    assert rows[1]["one_float64_gram_gib"] == pytest.approx(2.98, abs=.01)
    assert not (tmp_path / "reports").exists()


def test_full_sdp_child_command_is_same_convex_pipeline(tmp_path):
    graph = _create_graph(tmp_path, "G11", 800)
    command = batch._child_command(
        path=graph, jsonl=tmp_path / "output.jsonl",
        solver="SCS", eps=1e-3, max_iters=3000, restarts=20,
        seed=2026, cpu_threads=8, scs_indirect=False, n=800,
    )
    assert command[:3] == [batch.sys.executable, "-m", "maxkcut"]
    assert "--solver" in command
    assert command[command.index("--solver") + 1] == "SCS"
    assert command[command.index("--max-n") + 1] == "800"
    assert "--cpu-threads" in command
    assert "--scs-cudss" not in command
    assert "maxkcut-gpu" not in command


def test_completed_archive_independently_verified_and_corruption_detected(tmp_path):
    graph = _create_graph(tmp_path, "G1")
    jsonl = tmp_path / "runs" / "G1.jsonl"
    archive = _write_complete_sdp_archive(graph, jsonl)
    result = batch._read_saved_result(jsonl, "G1", 2, graph)
    assert result["best_cut"] == 1
    assert result["best_method"] == "FJ"
    assert result["sdp_solver_status"] == "optimal"
    assert Path(result["archive_path"]) == archive

    graph.write_text("3 1\n1 2 -1\n", encoding="utf8")
    with pytest.raises(ValueError, match="SHA256"):
        batch._read_saved_result(jsonl, "G1", 2, graph)

    graph.write_text("3 1\n1 2 1\n", encoding="utf8")
    (archive / "K-MeansNM.npz").unlink()
    with pytest.raises((OSError, FileNotFoundError)):
        batch._read_saved_result(jsonl, "G1", 2, graph)


def test_batch_runs_subprocess_and_resume_skips_verified_success(
    monkeypatch, tmp_path
):
    graph = _create_graph(tmp_path, "G1")
    target = tmp_path / "full-sdp"
    calls = []

    def fake_run(command, *, stdout, stderr, timeout, check):
        calls.append(list(command))
        assert stderr == subprocess.STDOUT
        assert 0 < timeout <= 10
        assert check is False
        jsonl = Path(command[command.index("--output") + 1])
        _write_complete_sdp_archive(graph, jsonl, restarts=2)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("maxkcut.archive._git_metadata", lambda: {"commit": None, "dirty": None})
    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    options = {
        "restarts": 2, "seconds_per_graph": 10.,
        "total_seconds": 120.,
    }
    first = batch.run_full_sdp_batch(
        graph.parent, target, **options
    )
    assert first["complete"] == 1 and first["failed"] == 0
    assert len(calls) == 1
    second = batch.run_full_sdp_batch(
        graph.parent, target, **options
    )
    assert second["skipped_verified"] == 1
    assert len(calls) == 1
    case = json.loads((target / "cases" / "G1.json").read_text())
    assert case["status"] == "complete"
    assert case["best_cut"] == 1.
    with (target / "summary.csv").open(newline="", encoding="utf8") as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 1
    assert saved[0]["status"] == "complete"
    with pytest.raises(ValueError, match="new --output-dir"):
        batch.run_full_sdp_batch(
            graph.parent, target, **{**options, "eps": 1e-5}
        )


def test_skips_graphs_above_cap_before_launching_subprocess(monkeypatch, tmp_path):
    _create_graph(tmp_path, "G1", 800)
    _create_graph(tmp_path, "G81", 20000)

    def forbidden(*args, **kwargs):
        raise AssertionError("should not invoke SDP child")

    monkeypatch.setattr(batch.subprocess, "run", forbidden)
    # Select the huge instance explicitly; it is marked skipped_size.
    result = batch.run_full_sdp_batch(
        tmp_path / "data", tmp_path / "out", names=["G81"],
    )
    assert result["skipped_size"] == 1
    assert result["complete"] == 0
    assert not (tmp_path / "out" / "runs").exists()


def test_unsafe_size_requires_two_opt_ins(tmp_path):
    path = _create_graph(tmp_path, "G81", 20000)
    with pytest.raises(ValueError, match="allow-large"):
        batch.run_full_sdp_batch(
            path.parent, tmp_path / "out",
            names=["G81"], max_n=20000
        )
    with pytest.raises(ValueError, match="acknowledge-huge-risk"):
        batch.run_full_sdp_batch(
            path.parent, tmp_path / "out",
            names=["G81"], max_n=20000, allow_large=True
        )
    assert not (tmp_path / "out").exists()


def test_timeout_captured_and_only_retried_when_explicit(
    monkeypatch, tmp_path
):
    graph = _create_graph(tmp_path, "G1")
    target = tmp_path / "out"
    calls = []
    def timed_out(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(batch.subprocess, "run", timed_out)
    options = {
        "restarts": 2, "seconds_per_graph": 5.,
        "total_seconds": 30.,
    }
    first = batch.run_full_sdp_batch(graph.parent, target, **options)
    assert first["timeout"] == 1 and len(calls) == 1
    second = batch.run_full_sdp_batch(graph.parent, target, **options)
    assert second["skipped_failed"] == 1 and len(calls) == 1
    third = batch.run_full_sdp_batch(
        graph.parent, target, retry_failed=True, **options,
    )
    assert third["timeout"] == 1 and len(calls) == 2
    assert (target / "runs" / "G1" / "attempt-001").is_dir()
    assert (target / "runs" / "G1" / "attempt-002").is_dir()
