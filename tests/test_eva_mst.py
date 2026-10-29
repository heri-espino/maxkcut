"""Exact, original-EVA-ambient-space MST rounding and archive tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

pytest.importorskip("numba", minversion="0.67")

from maxkcut.archive import load_method, verify_archive
from maxkcut.eva_cli import run_one
from maxkcut.eva_mst import (
    METHODS, _cut_longest_tree_edge, mst_binary_partition,
)
from maxkcut.eva_mst_cli import run_graph, write_summary
from maxkcut.io import load_graph, edge_arrays
from maxkcut.objective import cut_weight


@pytest.mark.parametrize("metric,scipy_metric", [
    ("l1", "cityblock"), ("l2", "euclidean"), ("linf", "chebyshev"),
])
def test_exact_complete_mst_matches_scipy_reference(metric, scipy_metric):
    points = np.random.default_rng(33).normal(size=(13, 6))
    labels, report, parents, lengths = mst_binary_partition(points, metric)
    full_distances = cdist(points, points, metric=scipy_metric)
    reference = minimum_spanning_tree(full_distances).toarray()
    expected = reference[reference > 0]
    # Independent SciPy Kruskal reference, operating on all pairwise
    # distances; our Numba Prim routine must be exact for every metric.
    assert np.isclose(np.sum(lengths), np.sum(expected),
                      rtol=1e-10, atol=1e-10)
    assert np.isclose(max(lengths), max(expected),
                      rtol=1e-10, atol=1e-10)
    assert report["tree_edges"] == len(points) - 1
    assert report["pca_projection"] is False
    assert report["ambient_dimensions"] == points.shape[1]
    assert 0 < np.sum(labels) < len(points)
    assert parents[0] == -1
    for child in range(1, len(points)):
        expected_distance = full_distances[child, parents[child]]
        assert np.isclose(lengths[child], expected_distance, atol=1e-10)


@pytest.mark.parametrize("metric", ["l1", "l2", "linf"])
def test_cut_single_longest_edge_gives_exactly_two_components(metric):
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [15.0, 15.0, 30.0],
        [15.0, 15.1, 30.0],
    ])
    labels, report, parents, lengths = mst_binary_partition(points, metric)
    assert labels.tolist() == [0, 0, 1, 1]
    assert report["component_sizes"] == [2, 2]
    assert report["ambient_dimensions"] == 3  # third original feature retained
    assert report["largest_edge_length"] == float(max(lengths))
    assert np.array_equal(
        _cut_longest_tree_edge(parents, lengths)[0], labels
    )
    assert report["is_randomized"] is False


def test_identical_points_have_deterministic_tie_break():
    points = np.zeros((6, 4))
    first = mst_binary_partition(points, "l2")
    second = mst_binary_partition(points, "l2")
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[2], second[2])
    assert sorted(first[1]["component_sizes"]) == [1, 5]
    assert first[1]["removed_edge_child"] == 1


@pytest.mark.parametrize("invalid", [
    np.ones((1, 4)), np.full((4, 3), np.nan),
    np.ones((3, 0)), np.ones(4),
])
def test_invalid_point_cloud(invalid):
    with pytest.raises(ValueError):
        mst_binary_partition(invalid, "l1")


def test_invalid_metric():
    with pytest.raises(ValueError, match="metric"):
        mst_binary_partition(np.ones((3, 3)), "cosine")


def _completed_eva(tmp_path):
    path = tmp_path / "G11.txt"
    path.write_text(
        "5 7\n1 2 2\n1 3 -1\n1 4 1\n2 3 2\n"
        "2 5 1\n3 4 1\n4 5 3\n", encoding="utf8",
    )
    source = tmp_path / "saved-eva"
    args = SimpleNamespace(
        max_n=20, output_dir=source, rank=None,
        tolerance=1e-6, max_sweeps=20, seed=2026,
        checkpoint_every=2, rounding_seed=2026,
        restarts=2, cpu_threads=1, convex_dir=None,
        push=False, push_branch=None, push_every=0,
    )
    run_one(path, args)
    return path, source, tmp_path / "experiment005"


def test_archived_exact_msts_are_verified_and_reusable(tmp_path):
    path, source, dest = _completed_eva(tmp_path)
    original_factor = (source / "factors" / "G11.npz").read_bytes()
    results = run_graph(
        path, source=source, dest=dest, metrics=["l1", "l2", "linf"]
    )
    assert len(results) == 3
    assert {row["method"] for row in results} == set(METHODS.values())
    graph = load_graph(path)
    edges = edge_arrays(graph)
    for row in results:
        report = verify_archive(Path(row["archive_path"]))
        assert report["status"] == "complete"
        assert report["methods"][row["method"]]["verified_partitions"] == 1
        labels, scores = load_method(
            Path(row["archive_path"]) / f"{row['method']}.npz"
        )
        assert len(labels) == 1
        assert scores[0] == cut_weight(edges, labels[0])
        assert row["cut_weight"] == scores[0]
        with np.load(row["tree_path"], allow_pickle=False) as tree:
            verified_labels, child, sizes = _cut_longest_tree_edge(
                tree["parent"], tree["edge_lengths"]
            )
            np.testing.assert_array_equal(verified_labels, labels[0])
            assert row["component_sizes"] == list(sizes)
            assert row["removed_edge_child"] == child

    assert (source / "factors" / "G11.npz").read_bytes() == original_factor
    snapshots = list((dest / "factor-snapshots").glob("G11-*.npz"))
    assert len(snapshots) == 1
    assert snapshots[0].read_bytes() == original_factor

    # Reexecution does not produce new archives or solve EVA again.
    again = run_graph(
        path, source=source, dest=dest, metrics=["l1", "l2", "linf"]
    )
    assert again == results
    assert len(list((dest / "archives").iterdir())) == 3

    summary = write_summary(dest)
    with summary.open(newline="", encoding="utf8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3
    assert set(row["metric"] for row in rows) == {"l1", "l2", "linf"}
    assert all(int(row["component_size_0"]) >= 1 for row in rows)


def test_wrong_factor_identity_does_not_silently_reuse_archive(tmp_path):
    path, source, dest = _completed_eva(tmp_path)
    result = run_graph(path, source=source, dest=dest, metrics=["l2"])[0]
    record_path = next((dest / "cases").glob("*-mst-l2.json"))
    record = json.loads(record_path.read_text(encoding="utf8"))
    record["graph_sha256"] = "wrong"
    record_path.write_text(json.dumps(record), encoding="utf8")
    with pytest.raises(ValueError, match="different|mismatch"):
        run_graph(path, source=source, dest=dest, metrics=["l2"])
    assert Path(result["archive_path"]).is_dir()
