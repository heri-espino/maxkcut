"""Experiment013: exact paired signed Max-Cut grid, concurrency and resume."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("numba")

from maxkcut.archive import load_method, verify_archive
from maxkcut.eva_cli import run_one
from maxkcut.eva_pipeline_grid import (
    STAGES, evaluate_pipeline, expand_grid,
)
from maxkcut.eva_pipeline_grid_cli import run_graph
from maxkcut.eva_rounding_cli import round_saved_factor, _round_one
from maxkcut.eva import load_eva_factor
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight


def _sample(n=9, seed=4):
    rng = np.random.default_rng(seed)
    matrix = rng.integers(-4, 5, size=(n, n)).astype(float)
    matrix[np.tril_indices(n)] = 0.
    matrix += matrix.T
    y = rng.normal(size=(n, 5))
    y /= np.linalg.norm(y, axis=1)[:, None]
    return sparse.csr_matrix(matrix), y


def _one_cfg(rounder, **overrides):
    return expand_grid({
        "version": 1, "families": [{
            "name": "test", "rounder": rounder,
            "base": overrides,
        }],
    })[0]


def test_grid_expansion_cartesian_and_deterministic_identity():
    spec = {"version": 1, "families": [{
        "name": "KM", "rounder": "kmeansnm",
        "axes": {"flip_sweeps": [0, 20], "kl_passes": [0, 10]},
    }, {
        "name": "Aware", "rounder": "cutaware-kmeans",
        "axes": {"direction_trials": [0, 8]},
    }]}
    configs = expand_grid(spec)
    assert len(configs) == 6
    assert len({x["id"] for x in configs}) == 6
    assert [x["id"] for x in configs] == [
        x["id"] for x in expand_grid(spec)
    ]
    assert sorted(set(c["rounder"] for c in configs)) == [
        "cutaware-kmeans", "kmeansnm",
    ]
    with pytest.raises(ValueError, match="only apply"):
        expand_grid({"version": 1, "families": [{
            "name": "invalid", "rounder": "gw",
            "base": {"direction_trials": 5},
        }]})
    with pytest.raises(ValueError, match="requires"):
        _one_cfg("kmeansnm", kl_passes=0, kl_repeats=2)


@pytest.mark.parametrize("rounder", [
    "saved-kmeans-flip", "kmeansnm", "kmeans2n", "gw",
    "cutaware-kmeans", "gw-threshold",
])
def test_every_rounder_has_non_decreasing_signed_refinement(rounder):
    graph, y = _sample()
    edges = edge_arrays(graph)
    saved = np.random.default_rng(3).integers(
        0, 2, size=graph.shape[0], dtype=np.int8,
    )
    cfg = _one_cfg(
        rounder,
        direction_trials=5 if rounder == "cutaware-kmeans" else 0,
        flip_sweeps=10, kl_passes=5, kl_repeats=2,
        ils_cycles=3, kick_sizes=[2, 3],
    )
    stages, scores, details, times = evaluate_pipeline(
        graph, y, saved_labels=saved,
        saved_score=cut_weight(edges, saved),
        restart=5, graph_number=11,
        rounding_seed=2026, search_seed=31,
        config=cfg, target=1000.,
    )
    assert list(scores) == list(STAGES)
    assert all(
        scores[b] >= scores[a] - 1e-8
        for a, b in zip(STAGES, STAGES[1:])
    )
    assert scores["Final"] >= scores["Round"]
    assert len(details["kl_repeat_scores"]) == 2
    assert times["total_seconds"] >= 0
    for stage in STAGES:
        assert stages[stage].shape == (graph.shape[0],)
        assert cut_weight(edges, stages[stage]) == scores[stage]
    # Each stage's replay uses the same independent RNGs.
    replay, again, _, _ = evaluate_pipeline(
        graph, y, saved_labels=saved,
        saved_score=cut_weight(edges, saved),
        restart=5, graph_number=11,
        rounding_seed=2026, search_seed=31,
        config=cfg, target=1000.,
    )
    assert scores == again
    for stage in STAGES:
        np.testing.assert_array_equal(stages[stage], replay[stage])


def test_original_kmeans_restart_schedule_matches_same_seed_rounding():
    graph, y = _sample(12, 8)
    edges = edge_arrays(graph)
    saved = np.zeros(graph.shape[0], dtype=np.int8)
    for method, original in (
        ("kmeansnm", "K-MeansNM_EVA"),
        ("kmeans2n", "K-Means2N_EVA"),
        ("gw", "FJ_EVA"),
    ):
        original_label, _ = _round_one(
            y, graph, edges, original, 11, 7, 2026,
        )
        result, scores, details, times = evaluate_pipeline(
            graph, y, saved_labels=saved,
            saved_score=0.,
            restart=7, graph_number=11, rounding_seed=2026,
            search_seed=31, config=_one_cfg(method),
        )
        np.testing.assert_array_equal(original_label, result["Round"])
        assert scores["Round"] == cut_weight(edges, original_label)




def test_saved_kmeans_kl_zero_repeat_reproduces_historical_experiment010_seed():
    # experiment010 used SeedSequence([seed, graph_id, restart, 101])
    from maxkcut.eva_escape import optimize_kl
    graph, y = _sample(12, 51)
    edges = edge_arrays(graph)
    saved = np.random.default_rng(9).integers(
        0, 2, size=12, dtype=np.int8,
    )
    expected, _ = optimize_kl(
        graph, saved.copy(),
        np.random.default_rng(np.random.SeedSequence(
            [2026, 11, 17, 101],
        )), passes=6,
    )
    cfg = _one_cfg(
        "saved-kmeans-flip", kl_passes=6, kl_repeats=1,
    )
    stages, scores, _, _ = evaluate_pipeline(
        graph, y, saved_labels=saved,
        saved_score=cut_weight(edges, saved),
        restart=17, graph_number=11,
        rounding_seed=2026, search_seed=2026,
        config=cfg,
    )
    np.testing.assert_array_equal(stages["KL"], expected)
    assert scores["KL"] == cut_weight(edges, expected)

def _source_archive(tmp):
    path = tmp / "G11.txt"
    path.write_text(
        "7 10\n1 2 2\n1 3 -1\n1 4 1\n2 3 2\n"
        "2 5 1\n3 4 -2\n4 5 3\n4 6 2\n5 7 -1\n6 7 2\n",
        encoding="utf8",
    )
    factor_dir = tmp / "eva"
    args = SimpleNamespace(
        max_n=20, output_dir=factor_dir, rank=None,
        tolerance=1e-6, max_sweeps=16, seed=2026,
        checkpoint_every=2, rounding_seed=2026,
        restarts=2, cpu_threads=1, convex_dir=None,
        push=False, push_branch=None, push_every=0,
    )
    run_one(path, args)
    round_dir = tmp / "round"
    round_saved_factor(
        path, source=factor_dir, dest=round_dir,
        methods=["K-MeansNM-Flip_EVA"], first=0,
        restarts=8, max_flip_sweeps=8,
    )
    return path, factor_dir, round_dir


def test_full_parallel_grid_resumes_and_promotes_top_configuration(tmp_path):
    path, factor, round_dir = _source_archive(tmp_path)
    factor_file = factor / "factors" / "G11.npz"
    original_bytes = factor_file.read_bytes()
    grid_file = tmp_path / "grid.json"
    grid = {"version": 1, "families": [
        {"name": "Saved", "rounder": "saved-kmeans-flip",
         "axes": {"kl_passes": [0, 3]}},
        {"name": "Aware", "rounder": "cutaware-kmeans",
         "base": {"direction_trials": 4, "kl_passes": 3}},
    ]}
    grid_file.write_text(json.dumps(grid), encoding="utf8")
    dest = tmp_path / "grid-results"
    options = dict(
        factor_dir=factor, rounding_dir=round_dir,
        output_dir=dest, grid_path=grid_file,
        chunk_size=2, workers=2, seed=13, target=1000.,
    )
    first, leaderboard = run_graph(path, top_starts=4, **options)
    assert first["configurations_total"] == 3
    assert all(x["count"] == 4 for x in first["leaderboard"])
    assert len(list((dest / "cases").glob("*.json"))) == 6
    for item in (dest / "cases").glob("G*.json"):
        record = json.loads(item.read_text(encoding="utf8"))
        checked = verify_archive(record["archive_path"])
        assert checked["status"] == "complete"
        assert all(
            checked["methods"][f"Grid-{stage}_EVA"]["verified_partitions"] == 2
            for stage in STAGES
        )
        for stage in STAGES:
            part, scores = load_method(
                Path(record["archive_path"]) /
                f"Grid-{stage}_EVA.npz",
            )
            for i, trial in enumerate(record["trials"]):
                actual = cut_weight(edge_arrays(load_graph(path)), part[i])
                assert actual == scores[i] == trial["scores"][stage]
    assert factor_file.read_bytes() == original_bytes
    count_before = len(list((dest / "archives").iterdir()))
    identical, _ = run_graph(path, top_starts=4, **options)
    assert identical["leaderboard"] == first["leaderboard"]
    assert len(list((dest / "archives").iterdir())) == count_before

    # Promoting the winning configuration to more paired sources
    # reuses completed first chunks and adds only the extra chunk.
    promoted, _ = run_graph(
        path, top_starts=6, promote_from=leaderboard,
        promote_top=1, **options,
    )
    assert promoted["configurations_total"] == 1
    assert promoted["leaderboard"][0]["count"] == 6
    assert len(list((dest / "archives").iterdir())) == count_before + 1
