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
        restarts=2, convex_dir=None, push=False,
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
