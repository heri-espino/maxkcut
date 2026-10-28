"""Correctness and reproducibility tests for nested multi-edge EVA MST rounding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from maxkcut.archive import (
    _atomic_npz, load_method, new_archive, new_run_id, save_method,
    update_manifest, verify_archive,
)
from maxkcut.eva_mst_multicut import (
    longest_edge_forest, contracted_adjacency, multi_edge_candidates,
)
from maxkcut.eva_mst_multicut_cli import run_graph, write_summary
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight


def _small_signed_graph():
    edges = [
        (0, 1, 2.), (0, 2, -1.), (0, 3, 3.),
        (1, 4, 2.), (1, 5, -3.), (2, 3, 2.),
        (2, 4, 4.), (3, 5, 1.), (4, 5, 2.),
    ]
    left = [u for u, v, w in edges] + [v for u, v, w in edges]
    right = [v for u, v, w in edges] + [u for u, v, w in edges]
    weights = [w for u, v, w in edges] * 2
    graph = sparse.coo_matrix((weights, (left, right)), shape=(6, 6)).tocsr()
    return graph


def _path_tree():
    return (
        np.array([-1, 0, 1, 2, 3, 4], dtype=np.int32),
        np.array([0., 12., 9., 10., 8., 11.], dtype=float),
    )


def test_forest_component_counts_and_nested_longest_edge_rule():
    parent, distances = _path_tree()
    previous = None
    for q in [1, 2, 3, 4, 5]:
        components, removed = longest_edge_forest(parent, distances, q)
        assert len(np.unique(components)) == q + 1
        assert len(removed) == q
        assert list(removed) == [1, 5, 3, 2, 4][:q]
        assert np.bincount(components).sum() == 6
        if previous is not None:
            # Each new forest component is contained in an old one.
            for comp in np.unique(components):
                assert len(np.unique(previous[components == comp])) == 1
        previous = components


def test_coarse_signed_contraction_exactly_preserves_any_binary_cut():
    graph = _small_signed_graph()
    parent, distances = _path_tree()
    original = edge_arrays(graph)
    rng = np.random.default_rng(31)
    for q in [1, 2, 4]:
        comps, removed = longest_edge_forest(parent, distances, q)
        coarse_graph = contracted_adjacency(graph, comps)
        assert coarse_graph.shape == (q + 1, q + 1)
        assert coarse_graph.nnz <= (q + 1) * q
        for _ in range(30):
            c = rng.integers(0, 2, size=q + 1, dtype=np.int8)
            assert cut_weight(original, c[comps]) == pytest.approx(
                cut_weight(edge_arrays(coarse_graph), c), abs=1e-10
            )


def test_nested_coarse_best_never_decreases_and_final_refinement_monotone():
    graph = _small_signed_graph()
    parent, distances = _path_tree()
    results = list(multi_edge_candidates(
        graph, parent, distances, [1, 2, 3, 4, 5],
        coarse_restarts=6, coarse_sweeps=12,
        flip_sweeps=20, seed=2026, graph_number=11,
    ))
    assert len(results) == 5
    assert len(results[0]["coarse_trials"]) == 1
    assert all(len(result["coarse_trials"]) == 7 for result in results[1:])
    coarse = [result["best_coarse_cut"] for result in results]
    assert all(b >= a for a, b in zip(coarse, coarse[1:]))
    for result in results:
        assert result["final_cut"] >= result["best_coarse_cut"] - 1e-8
        assert result["vertex_improvement"] >= -1e-8
        assert result["forest_component_count"] == result["q"] + 1
        assert result["final_labels"].shape == (6,)
        assert result["final_cut"] == pytest.approx(
            cut_weight(edge_arrays(graph), result["final_labels"]),
        )
        assert result["best_coarse_cut"] == max(result["coarse_scores"])


def test_q_validation_and_deterministic_replay():
    graph = _small_signed_graph()
    parent, distances = _path_tree()
    options = dict(
        coarse_restarts=3, coarse_sweeps=10,
        flip_sweeps=10, seed=34, graph_number=11,
    )
    first = list(multi_edge_candidates(
        graph, parent, distances, [1, 3, 5], **options,
    ))
    second = list(multi_edge_candidates(
        graph, parent, distances, [1, 3, 5], **options,
    ))
    for a, b in zip(first, second):
        assert a["coarse_scores"] == b["coarse_scores"]
        assert a["final_cut"] == b["final_cut"]
        np.testing.assert_array_equal(a["final_labels"], b["final_labels"])
    for invalid in [[2, 3], [1, 1, 2], [1, 3, 2], [1, 6]]:
        with pytest.raises(ValueError):
            list(multi_edge_candidates(
                graph, parent, distances, invalid, **options,
            ))


def _create_archived_tree(tmp_path):
    # Signed input, with the original graph itself independently archived.
    graph = _small_signed_graph()
    upper = edge_arrays(graph)
    graph_path = tmp_path / "G11.txt"
    graph_path.write_text(
        "6 9\n" + "".join(
            f"{u+1} {v+1} {int(w)}\n"
            for u, v, w in zip(*upper)
        ),
        encoding="utf8",
    )
    loaded = load_graph(graph_path)
    assert cut_weight(upper, np.arange(6) % 2) == pytest.approx(
        cut_weight(edge_arrays(loaded), np.arange(6) % 2)
    )
    parent, distances = _path_tree()
    first_component, removed = longest_edge_forest(parent, distances, 1)
    labels = np.asarray(first_component != first_component[0], dtype=np.int8)
    longest_cut = float(cut_weight(edge_arrays(loaded), labels))
    factor_sha = "d" * 64
    graph_sha = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    mst_dir = tmp_path / "mst"
    tree_path = mst_dir / "trees" / f"G11-{factor_sha[:16]}-mst-l2.npz"
    _atomic_npz(
        tree_path, parent=parent, edge_lengths=distances,
        removed_edge_child=np.asarray(int(removed[0]), dtype=np.int64),
    )
    trial_path = mst_dir / "test-trial.jsonl"
    archive = new_archive(
        mst_dir / "archives", new_run_id(), graph_path, 6,
        edge_arrays(loaded), 2, {"synthetic": True},
        jsonl_output=trial_path,
    )
    method = "MST-L2_EVA"
    save_method(archive, method, [labels], [longest_cut], k=2)
    update_manifest(archive, status="complete")
    record = {
        "status": "complete",
        "graph": "G11",
        "metric": "l2",
        "method": method,
        "graph_sha256": graph_sha,
        "factor_sha256": factor_sha,
        "factor_rank": 3,
        "cut_weight": longest_cut,
        "archive_path": str(archive["path"]),
        "tree_path": str(tree_path),
        "removed_edge_child": int(removed[0]),
        "component_sizes": list(np.bincount(labels, minlength=2)),
    }
    case_path = mst_dir / "cases" / (
        f"G11-{factor_sha[:16]}-mst-l2.json"
    )
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text(json.dumps(record), encoding="utf8")
    return graph_path, mst_dir, tmp_path / "multicut"


def test_archives_all_trials_refined_partition_and_resumes(tmp_path):
    path, mst_dir, dest = _create_archived_tree(tmp_path)
    one = run_graph(
        path, mst_dir=mst_dir, dest=dest, metrics=("l2",),
        q_values=(1, 2, 4), coarse_restarts=3,
        coarse_sweeps=10, vertex_sweeps=20, seed=72,
    )
    assert len(one) == 3
    assert len(list((dest / "archives").iterdir())) == 3
    for record in one:
        verified = verify_archive(Path(record["archive_path"]))
        assert verified["status"] == "complete"
        coarse_method, final_method = record["method_names"]
        trials = 1 if record["q"] == 1 else 4
        assert verified["methods"][coarse_method]["verified_partitions"] == trials
        assert verified["methods"][final_method]["verified_partitions"] == 1
        _, coarse_scores = load_method(
            Path(record["archive_path"]) / f"{coarse_method}.npz"
        )
        _, final_scores = load_method(
            Path(record["archive_path"]) / f"{final_method}.npz"
        )
        assert record["best_coarse_cut"] == max(coarse_scores)
        assert record["final_cut"] == final_scores[0]
        assert record["final_cut"] >= record["best_coarse_cut"]
    second = run_graph(
        path, mst_dir=mst_dir, dest=dest, metrics=("l2",),
        q_values=(1, 2, 4), coarse_restarts=3,
        coarse_sweeps=10, vertex_sweeps=20, seed=72,
    )
    assert second == one
    assert len(list((dest / "archives").iterdir())) == 3
    summary = write_summary(dest)
    assert sum(1 for _ in summary.open(encoding="utf8")) == 4
