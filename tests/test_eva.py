"""Fast, deterministic tests for the EVA vector-relaxation reproduction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from maxkcut.archive import verify_archive
from maxkcut.eva import load_eva_factor, solve_eva, vector_objective
from maxkcut.eva_cli import _full_sdp_comparison, run_one


def triangle():
    return sparse.csr_matrix(
        [[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]
    )


def test_objective_counts_edges_once_and_respects_sign():
    a = sparse.csr_matrix(
        [[0., 2., -1.], [2., 0., 3.], [-1., 3., 0.]]
    )
    y = np.array([[1., 0.], [-1., 0.], [1., 0.]])
    # The 2-weight edge cuts; the -1 edge does not; the 3-weight cuts.
    assert vector_objective(a, y) == pytest.approx(5.0)


def test_eva_monotone_and_unit_rows():
    history = []
    y, info = solve_eva(
        triangle(), tolerance=1e-8, max_sweeps=75, seed=17,
        on_sweep=lambda s: history.append(s["objective"]),
    )
    assert y.shape == (3, 3)  # ceil(sqrt(2*(3+1)))
    assert np.allclose(np.linalg.norm(y, axis=1), 1)
    assert all(b >= a - 1e-7 for a, b in zip(history, history[1:]))
    assert info["objective"] == pytest.approx(vector_objective(triangle(), y))
    assert info["objective"] >= 2.0  # triangle discrete Max-Cut optimum
    assert info["objective"] <= 2.25 + 1e-5  # triangle SDP optimum


def test_eva_checkpoint_resume_and_invalidation(tmp_path):
    cache = tmp_path / "eva.npz"
    y1, first = solve_eva(
        triangle(), tolerance=1e-12, max_sweeps=2, seed=7,
        checkpoint_path=cache, checkpoint_every=1,
    )
    assert cache.is_file()
    assert first["sweeps"] >= 1
    y2, second = solve_eva(
        triangle(), tolerance=1e-12, max_sweeps=25, seed=7,
        checkpoint_path=cache, checkpoint_every=1,
    )
    saved, info = load_eva_factor(cache)
    assert np.array_equal(saved, y2)
    assert info["sweeps"] == second["sweeps"]
    assert second["objective"] >= first["objective"] - 1e-7
    with pytest.raises(ValueError, match="mismatch"):
        solve_eva(
            triangle(), rank=2, seed=7, max_sweeps=25,
            checkpoint_path=cache,
        )


def test_eva_rejects_asymmetric_graph():
    with pytest.raises(ValueError, match="symmetric"):
        solve_eva(sparse.csr_matrix([[0., 1.], [2., 0.]]))


def test_cli_archives_every_partition(tmp_path):
    graph_path = tmp_path / "G11.txt"
    graph_path.write_text(
        "4 5\n1 2 2\n1 3 -1\n1 4 1\n2 3 2\n3 4 1\n",
        encoding="utf8",
    )
    root = tmp_path / "results" / "eva"
    args = SimpleNamespace(
        max_n=20, output_dir=root, rank=None,
        tolerance=1e-6, max_sweeps=12, seed=2026,
        checkpoint_every=3, rounding_seed=2026,
        restarts=2, convex_dir=None, cpu_threads=1, push=False,
        push_branch=None, push_every=0,
    )
    result = run_one(graph_path, args)
    assert result["graph"] == "G11"
    assert result["factor_rank"] == 4
    assert set(result["methods"]) == {
        "FJ_EVA", "K-Means2N_EVA", "K-MeansNM_EVA"
    }
    assert sum(item["restarts"] for item in result["methods"].values()) == 6
    rows = [
        json.loads(line) for line in
        (root / "trials" / "G11.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 6
    verified = verify_archive(Path(result["archive_path"]))
    assert verified["status"] == "complete"
    assert sum(item["verified_partitions"] for item in verified["methods"].values()) == 6
    assert (root / "factors" / "G11.npz").exists()
    assert (root / "cases" / "G11.json").exists()


def test_convex_comparison_rejects_graph_mismatch(tmp_path):
    full = tmp_path / "full"
    (full / "cases").mkdir(parents=True)
    (full / "cases" / "G11.json").write_text(json.dumps({
        "status": "complete", "input_sha256": "other",
        "source_jsonl": "not-accessed.jsonl"
    }))
    result = _full_sdp_comparison("G11", "expected", full, {})
    assert result["status"] == "graph_mismatch"


def test_eva_second_run_skips_verified_factor_and_roundings(tmp_path, monkeypatch):
    """All-graph reruns must never recalculate already completed partitions."""
    graph_path = tmp_path / "G1.txt"
    graph_path.write_text("3 3\n1 2 1\n2 3 1\n1 3 1\n", encoding="utf8")
    root = tmp_path / "results"
    args = SimpleNamespace(
        max_n=10, output_dir=root, rank=None, tolerance=1e-3,
        max_sweeps=10, seed=2026, checkpoint_every=2,
        rounding_seed=2026, restarts=2, convex_dir=None,
        cpu_threads=1, push=False, push_branch=None, push_every=0,
    )
    first = run_one(graph_path, args)
    def should_not_solve(*_args, **_kwargs):
        raise AssertionError("Cached EVA run should not solve again")
    monkeypatch.setattr("maxkcut.eva_cli.solve_eva", should_not_solve)
    second = run_one(graph_path, args)
    assert first["archive_path"] == second["archive_path"]
    assert first["factor_sweeps"] == second["factor_sweeps"]
    assert first["methods"] == second["methods"]


def test_all_graph_summary_pending_failure_and_completed(tmp_path):
    from maxkcut.eva_cli import _write_batch_summary
    root = tmp_path / "results"
    paths = [tmp_path / f"G{j}.txt" for j in (1, 2, 3)]
    (root / "cases").mkdir(parents=True)
    (root / "errors").mkdir(parents=True)
    (root / "errors" / "G2.json").write_text("{}")
    (root / "cases" / "G1.json").write_text(json.dumps({
        "graph": "G1", "status": "converged", "vertices": 4, "edges": 4,
        "factor_rank": 4, "factor_sweeps": 5,
        "objective_vector_relaxation": 5.5, "factor_elapsed_seconds": 2.0,
        "graph_sha256": "abc",
        "methods": {"FJ_EVA": {"best": 4}, "K-Means2N_EVA": {"best": 5},
                    "K-MeansNM_EVA": {"best": 5}},
        "comparison_to_full_sdp": {
            "status": "compared", "convex_source": "old",
            "methods": {"FJ": {"convex_best": 3},
                        "K-Means2N": {"convex_best": 4},
                        "K-MeansNM": {"convex_best": 4}},
        },
    }))
    _write_batch_summary(root, paths)
    with (root / "summary.csv").open(newline="", encoding="utf8") as stream:
        import csv
        rows = list(csv.DictReader(stream))
    assert [row["status"] for row in rows] == ["converged", "failed", "pending"]
    assert float(rows[0]["best_cut_eva_minus_convex"]) == 1
    assert rows[1]["convex_status"] == ""
