"""GitHub result publication tests; no SDP solves and no network calls."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maxkcut.archive import new_archive, save_method, update_manifest
from maxkcut.io import edge_arrays, load_graph
from maxkcut.sdp_publish import publish_snapshot
from maxkcut.sdp_publish_cli import _push_publication


def _study(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    graph = dataset / "G11.txt"
    graph.write_text("3 1\n1 2 1\n", encoding="utf8")

    batch = tmp_path / "reports" / "gset" / "full-sdp-batch"
    cases = batch / "cases"
    cases.mkdir(parents=True)
    (batch / "plan.json").write_text(
        json.dumps({"input_dir": str(dataset.resolve()),
                    "settings": {"solver": "SCS"}}), encoding="utf8",
    )

    output = batch / "runs" / "G11" / "attempt-001" / "G11.jsonl"
    output.parent.mkdir(parents=True)
    adjacency = load_graph(graph)
    archive = new_archive(
        output.parent / "artifacts", "synthetic-run-123", graph, 3,
        edge_arrays(adjacency), 2,
        {"solver": "SCS", "eps": 1e-3, "max_iters": 3000,
         "restarts": 2, "seed": 2026, "cpu_threads": 8},
        jsonl_output=output,
    )
    method_cuts = {
        "FJ": [1.0, 1.0], "K-Means2N": [1.0, 0.0],
        "K-MeansNM": [0.0, 1.0],
    }
    rows = []
    for method, values in method_cuts.items():
        partitions = (
            [np.array([0, 1, 1]), np.array([0, 1, 1])] if method == "FJ"
            else [np.array([0, 1, 1]), np.array([0, 0, 0])]
            if method == "K-Means2N"
            else [np.array([0, 0, 0]), np.array([0, 1, 1])]
        )
        save_method(archive, method, partitions, values, k=2)
        for rep, value in enumerate(values):
            rows.append({
                "graph": "G11", "n": 3, "k": 2, "method": method,
                "restart": rep, "cut_weight": value, "solver": "SCS",
                "solver_status": "optimal", "solver_iterations": 123,
                "sdp_original_solve_seconds": 2.5,
                "sdp_objective_primal": 1.3,
                "cache_path": "C:\\Users\\SECRET\\gram.npz",
                "partition_archive": "C:\\Users\\SECRET\\rounding.npz",
                "artifact_run_id": "synthetic-run-123",
            })
    update_manifest(archive, status="complete")
    output.write_text(
        "".join(json.dumps(item) + "\n" for item in rows),
        encoding="utf8",
    )
    (cases / "G11.json").write_text(json.dumps({
        "graph": "G11", "vertices": 3, "edges": 1,
        "status": "complete", "best_cut": 1.,
        "best_method": "FJ", "sdp_solver_status": "optimal",
        "settings": {"restarts": 2},
        "attempts": 1, "wall_seconds": 5.,
        "source_jsonl": str(output.resolve()),
        "archive_path": str(archive["path"].resolve()),
        "log_path": "C:\\Users\\SECRET\\solver.log",
        "input_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
    }), encoding="utf8")
    (cases / "G55.json").write_text(json.dumps({
        "graph": "G55", "vertices": 5000, "edges": 10000,
        "status": "started", "attempts": 1,
        "log_path": "C:\\Users\\SECRET\\solver.log",
    }), encoding="utf8")
    (cases / "G81.json").write_text(json.dumps({
        "graph": "G81", "vertices": 20000, "edges": 40000,
        "status": "skipped_size",
        "reason": "n exceeds limit",
    }), encoding="utf8")
    return batch


def test_publish_is_sanitized_idempotent_and_only_completed_archives(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "maxkcut.archive._git_metadata",
        lambda: {"commit": "abc123", "dirty": False},
    )
    batch = _study(tmp_path)
    target = tmp_path / "results" / "gset" / "full-sdp"
    first = publish_snapshot(batch, target)
    assert first["verified_complete"] == 1
    assert first["total_cases"] == 3
    assert first["statuses"] == {
        "complete": 1, "skipped_size": 1, "started": 1,
    }
    assert len(first["updated_files"]) == 5
    assert sorted(p.name for p in target.iterdir()) == [
        "README.md", "method-summary.csv", "provenance.csv",
        "rounding-trials.csv", "summary.csv",
    ]
    with (target / "summary.csv").open(newline="", encoding="utf8") as file:
        summary = {row["graph"]: row for row in csv.DictReader(file)}
    assert summary["G11"]["best_cut"] == "1.0"
    assert summary["G11"]["status"] == "complete"
    assert summary["G55"]["status"] == "started"
    assert summary["G55"]["best_cut"] == ""
    with (target / "method-summary.csv").open(
        newline="", encoding="utf8",
    ) as file:
        methods = {row["method"]: row for row in csv.DictReader(file)}
    assert len(methods) == 3
    assert float(methods["K-Means2N"]["mean_cut"]) == 0.5
    with (target / "rounding-trials.csv").open(
        newline="", encoding="utf8",
    ) as file:
        trials = list(csv.DictReader(file))
    assert len(trials) == 6
    assert all(row["graph"] == "G11" for row in trials)
    assert all(row["solver_status"] == "optimal" for row in trials)

    published = "\n".join(
        file.read_text(encoding="utf8") for file in target.iterdir()
    )
    assert "C:\\Users\\SECRET" not in published
    assert str(tmp_path.resolve()) not in published
    assert "proof of the integer optimum" in published
    assert not (target / "sdp-vs-cuda.csv").exists()
    assert publish_snapshot(batch, target)["updated_files"] == []


def test_corrupted_complete_archive_is_not_published_as_success(tmp_path):
    batch = _study(tmp_path)
    path = batch / "cases" / "G11.json"
    case = json.loads(path.read_text())
    (Path(case["archive_path"]) / "FJ.npz").unlink()
    target = tmp_path / "results"
    output = publish_snapshot(batch, target)
    assert output["verified_complete"] == 0
    assert len(output["unverified"]) == 1
    with (target / "summary.csv").open(newline="", encoding="utf8") as file:
        rows = list(csv.DictReader(file))
    entry = next(row for row in rows if row["graph"] == "G11")
    assert entry["status"] == "unverified_complete"
    assert entry["best_cut"] == ""
    with (target / "rounding-trials.csv").open(newline="", encoding="utf8") as file:
        assert list(csv.DictReader(file)) == []


def test_push_requires_main_and_rejects_unrecognized_files(
    monkeypatch, tmp_path
):
    from maxkcut import sdp_publish_cli as cli
    dest = tmp_path / "results" / "gset" / "full-sdp"
    dest.mkdir(parents=True)
    (dest / "summary.csv").write_text("graph,status\n", encoding="utf8")
    calls = []

    def fake_git(repo, *args, check=True):
        calls.append(args)
        if args[0] == "rev-parse":
            return SimpleNamespace(stdout=str(tmp_path) + "\n", returncode=0)
        if args[0] == "symbolic-ref":
            return SimpleNamespace(stdout="main\n", returncode=0)
        if args[0] == "diff":
            return SimpleNamespace(stdout="", returncode=1)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(cli, "_git", fake_git)
    assert _push_publication(dest, 54) is True
    assert ("push", "origin", "HEAD:main") in calls
    assert all("--force" not in call for call in calls)
    assert all("pull" not in call for call in calls)

    (dest / "private.txt").write_text("should not publish", encoding="utf8")
    with pytest.raises(ValueError, match="unrecognized"):
        _push_publication(dest, 55)


def test_push_refuses_feature_branch(monkeypatch, tmp_path):
    from maxkcut import sdp_publish_cli as cli
    dest = tmp_path / "results" / "gset" / "full-sdp"
    dest.mkdir(parents=True)
    (dest / "summary.csv").write_text("graph,status\n", encoding="utf8")

    def fake_git(repo, *args, check=True):
        if args[0] == "rev-parse":
            return SimpleNamespace(stdout=str(tmp_path), returncode=0)
        if args[0] == "symbolic-ref":
            return SimpleNamespace(stdout="some-feature\n", returncode=0)
        raise AssertionError("should not stage/commit or push feature branch")

    monkeypatch.setattr(cli, "_git", fake_git)
    with pytest.raises(RuntimeError, match="not 'main'"):
        _push_publication(dest, 54)


def test_rejects_individual_jsonl_score_mismatch_even_if_best_agrees(tmp_path):
    batch = _study(tmp_path)
    case = json.loads(
        (batch / "cases" / "G11.json").read_text(encoding="utf8")
    )
    jsonl = Path(case["source_jsonl"])
    rows = [
        json.loads(line) for line in jsonl.read_text(encoding="utf8").splitlines()
    ]
    # K-Means2N best remains 1, but the *other* restart is corrupted.
    changed = False
    for row in rows:
        if row["method"] == "K-Means2N" and row["restart"] == 1:
            assert row["cut_weight"] == 0.0
            row["cut_weight"] = 1.0
            changed = True
    assert changed
    jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf8",
    )
    target = tmp_path / "tracked"
    output = publish_snapshot(batch, target)
    assert output["verified_complete"] == 0
    assert output["unverified"]
    assert output["statuses"]["unverified_complete"] == 1
    assert "JSONL rounding values differ" in output["unverified"][0]

    # The top-level archive reader approved the original method's best
    # score, but a single corrupted nonbest restart invalidates the ENTIRE
    # graph for public reporting. No stale numeric data may escape.
    with (target / "summary.csv").open(newline="", encoding="utf8") as stream:
        summary = {row["graph"]: row for row in csv.DictReader(stream)}
    assert summary["G11"]["status"] == "unverified_complete"
    assert summary["G11"]["best_cut"] == ""
    assert summary["G11"]["best_method"] == ""
    assert summary["G11"]["sdp_solver_status"] == ""

    for filename in ("rounding-trials.csv", "method-summary.csv", "provenance.csv"):
        with (target / filename).open(newline="", encoding="utf8") as stream:
            assert list(csv.DictReader(stream)) == [], filename
    assert "unverified_complete" in (target / "README.md").read_text(
        encoding="utf8"
    )
