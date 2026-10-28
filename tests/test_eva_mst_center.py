"""Tests for central one-edge cuts of exact original-space EVA MSTs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("numba", minversion="0.67")

from maxkcut.archive import load_method, verify_archive
from maxkcut.eva_cli import run_one
from maxkcut.eva_mst_center import (
    balanced_edge, diameter_center_edge, all_tree_edge_cut_scores,
    best_cut_edge, labels_cutting_edge,
)
from maxkcut.eva_mst_center_cli import run_central_graph, write_center_summary
from maxkcut.io import load_graph, edge_arrays
from maxkcut.objective import cut_weight


def test_most_balanced_edge_is_edge_betweenness_maximizer():
    # rooted tree:     0-1-2, 1-3-4, 3-5, 4-6
    parent = np.array([-1, 0, 1, 1, 3, 3, 4])
    weights = np.array([0.0, 1., 1., 1., 1., 1., 1.])
    child, stats = balanced_edge(parent, weights)
    labels = labels_cutting_edge(parent, child)
    assert child == 3
    assert sorted(np.bincount(labels, minlength=2).tolist()) == [3, 4]
    assert stats["edge_betweenness_pairs"] == 12
    assert stats["imbalance_vertices"] == 1


def test_metric_center_is_weighted_diameter_midpoint():
    parent = np.array([-1, 0, 1, 2, 3])
    lengths = np.array([0., 2., 2., 2., 2.])
    child, stats = diameter_center_edge(parent, lengths)
    assert child == 2
    assert stats["tree_diameter_length"] == 8.
    assert stats["midpoint_at_vertex"] is True
    assert sorted(np.bincount(labels_cutting_edge(parent, child)).tolist()) == [2, 3]


def test_metric_center_zero_length_tree_uses_balance():
    parent = np.array([-1, 0, 1, 2, 3])
    lengths = np.zeros(5)
    child, stats = diameter_center_edge(parent, lengths)
    assert child == 2
    assert stats["degenerate_zero_diameter"] is True


@pytest.mark.parametrize("seed", [4, 7, 14])
def test_lca_path_scores_match_brute_force_for_signed_weights(seed):
    rng = np.random.default_rng(seed)
    n = 25
    # random valid rooted tree, node indices are topological
    parent = np.array([-1] + [int(rng.integers(i)) for i in range(1, n)])
    lengths = np.array([0.] + [float(rng.uniform(0.1, 2)) for _ in range(n - 1)])
    pairs = np.array(
        [(i, j) for i in range(n) for j in range(i + 1, n)]
    )
    rng.shuffle(pairs)
    pairs = pairs[:100]
    edge_u, edge_v = pairs[:, 0], pairs[:, 1]
    # Signed integer weights including negative and zero.
    w = rng.integers(-10, 11, size=len(pairs)).astype(float)
    edges = edge_u, edge_v, w
    scores = all_tree_edge_cut_scores(parent, lengths, edges)
    expected = np.full(n, np.nan)
    for child in range(1, n):
        labels = labels_cutting_edge(parent, child)
        expected[child] = cut_weight(edges, labels)
    np.testing.assert_allclose(scores[1:], expected[1:], atol=1e-10)
    selected, details = best_cut_edge(parent, lengths, edges)
    assert selected == int(np.argmax(expected[1:]) + 1)
    assert details["best_tree_edge_cut_weight"] == expected[selected]


def _completed_eva(tmp):
    graph_path = tmp / "G11.txt"
    graph_path.write_text(
        "5 7\n1 2 2\n1 3 -1\n1 4 1\n2 3 2\n"
        "2 5 1\n3 4 1\n4 5 3\n", encoding="utf8",
    )
    source = tmp / "eva"
    args = SimpleNamespace(
        max_n=20, output_dir=source, rank=None,
        tolerance=1e-6, max_sweeps=20, seed=2026,
        checkpoint_every=2, rounding_seed=2026,
        restarts=2, cpu_threads=1, convex_dir=None,
        push=False, push_branch=None, push_every=0,
    )
    run_one(graph_path, args)
    return graph_path, source, tmp / "mst"


def test_combined_tree_reuse_verified_archives_and_no_eva_rerun(tmp_path):
    path, source, dest = _completed_eva(tmp_path)
    original_factor = (source / "factors" / "G11.npz").read_bytes()
    result = run_central_graph(
        path, source=source, dest=dest,
        metrics=["l1", "l2", "linf"],
        rules=["balanced", "diameter_center", "bestcut"],
    )
    assert len(result) == 9
    original_msts = list((dest / "trees").glob("*.npz"))
    assert len(original_msts) == 3
    for item in result:
        archive = Path(item["archive_path"])
        verified = verify_archive(archive)
        assert verified["status"] == "complete"
        assert verified["methods"][item["method"]]["verified_partitions"] == 1
        labels, scores = load_method(archive / f"{item['method']}.npz")
        assert labels.shape == (1, 5)
        assert item["cut_weight"] == scores[0]
        with np.load(item["tree_path"], allow_pickle=False) as tree:
            recon = labels_cutting_edge(tree["parent"],
                                        item["removed_edge_child"])
            np.testing.assert_array_equal(recon, labels[0])

    assert (source / "factors" / "G11.npz").read_bytes() == original_factor
    again = run_central_graph(
        path, source=source, dest=dest,
        metrics=["l1", "l2", "linf"],
        rules=["balanced", "diameter_center", "bestcut"],
    )
    assert again == result
    assert len(list((dest / "center-archives").iterdir())) == 9
    assert len(list((dest / "trees").glob("*.npz"))) == 3
    write_center_summary(dest)
    with (dest / "center-summary.csv").open(
        newline="", encoding="utf8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert {row["rule"] for row in rows} == {
        "balanced", "diameter_center", "bestcut"
    }
