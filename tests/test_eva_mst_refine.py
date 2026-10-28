"""Experiment007: verify a second Max-Cut optimization after archived MST-BestCut.

No EVA or MST numerical optimization is run in these tests; a tiny
source MST tree and its independently verified single-edge cut are saved.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from maxkcut.archive import (
    load_method, new_archive, new_run_id, save_method, update_manifest,
    verify_archive,
)
from maxkcut.eva_mst_center import (
    best_cut_edge, labels_cutting_edge,
)
from maxkcut.eva_mst_refine_cli import (
    _find_bestcut_record, refine_saved_bestcut, write_summary,
)
from maxkcut.io import edge_arrays, load_graph
from maxkcut.local_search import improve_by_flips
from maxkcut.objective import cut_weight


def _saved_star_tree_bestcut(tmp_path):
    path = tmp_path / "G11.txt"
    # Complete graph K5 with unit weights: a star MST cut of one
    # singleton has score 4, whereas a 2:3 partition scores 6.
    edges_txt = [
        (i, j, 1) for i in range(1, 6) for j in range(i + 1, 6)
    ]
    path.write_text(
        "5 10\n" + "".join(
            f"{i} {j} {w}\n" for i, j, w in edges_txt
        ),
        encoding="utf8",
    )
    original_graph = load_graph(path)
    edges = edge_arrays(original_graph)
    parent = np.array([-1, 0, 0, 0, 0], dtype=np.int32)
    lengths = np.array([0., 0.5, 0.8, 0.9, 1.], dtype=float)
    removed_child, details = best_cut_edge(parent, lengths, edges)
    original_labels = labels_cutting_edge(parent, removed_child)
    initial_cut = float(cut_weight(edges, original_labels))
    assert initial_cut == 4.
    source_dir = tmp_path / "mst-results"
    factor_sha = "a" * 64
    graph_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    tree_path = source_dir / "trees" / f"G11-{factor_sha[:16]}-mst-l2.npz"
    tree_path.parent.mkdir(parents=True)
    np.savez_compressed(
        tree_path, parent=parent, edge_lengths=lengths,
        removed_edge_child=np.array(removed_child),
    )
    trial = source_dir / "dummy.jsonl"
    src_archive = new_archive(
        source_dir / "center-archives", new_run_id(), path, 5,
        edges, 2, {"origin": "synthetic MST-BestCut"}, jsonl_output=trial,
    )
    method = "MST-BestCut-L2_EVA"
    save_method(src_archive, method, [original_labels], [initial_cut], k=2)
    update_manifest(src_archive, status="complete")
    assert verify_archive(src_archive["path"])["status"] == "complete"
    record = {
        "status": "complete",
        "graph": "G11",
        "graph_sha256": graph_sha,
        "factor_sha256": factor_sha,
        "metric": "l2",
        "method": method,
        "rule": "bestcut",
        "cut_weight": initial_cut,
        "archive_path": str(src_archive["path"]),
        "tree_path": str(tree_path),
        "removed_edge_child": removed_child,
        "component_sizes": [4, 1],
        "factor_rank": 3,
    }
    case_dir = source_dir / "center-cases"
    case_dir.mkdir(parents=True)
    (case_dir / f"G11-{factor_sha[:16]}-mst-l2-bestcut.json").write_text(
        json.dumps(record), encoding="utf8",
    )
    return path, source_dir, tmp_path / "refined-results"


def test_refinement_strictly_improves_and_preserves_the_original_cut(tmp_path):
    path, mst_dir, dest = _saved_star_tree_bestcut(tmp_path)
    result = refine_saved_bestcut(
        path, mst_dir=mst_dir, dest=dest, metrics=("l2",),
        max_flip_sweeps=10, flip_seed=2026,
    )
    assert len(result) == 1
    row = result[0]
    assert row["initial_cut"] == 4.
    assert row["refined_cut"] == 6.
    assert row["improvement"] == 2.
    assert row["flip_count"] >= 1
    assert row["termination"] == "local_optimum"
    assert row["graph_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert row["factor_sha256"] == "a" * 64
    assert row["source_factor_rank"] == 3
    verified = verify_archive(Path(row["archive_path"]))
    assert verified["status"] == "complete"
    assert verified["methods"][row["method"]]["verified_partitions"] == 1
    refined, scores = load_method(
        Path(row["archive_path"]) / f"{row['method']}.npz"
    )
    assert scores[0] == 6.
    assert cut_weight(edge_arrays(load_graph(path)), refined[0]) == 6.

    # Second identical invocation independently verifies and skips.
    again = refine_saved_bestcut(
        path, mst_dir=mst_dir, dest=dest, metrics=("l2",),
        max_flip_sweeps=10, flip_seed=2026,
    )
    assert again == result
    assert len(list((dest / "archives").iterdir())) == 1

    summary = write_summary(dest)
    with summary.open(newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["initial_cut"] == "4.0"
    assert rows[0]["refined_cut"] == "6.0"


def test_different_seed_is_a_new_archived_local_search_trial(tmp_path):
    path, mst_dir, dest = _saved_star_tree_bestcut(tmp_path)
    a = refine_saved_bestcut(
        path, mst_dir=mst_dir, dest=dest, metrics=("l2",),
        max_flip_sweeps=5, flip_seed=10,
    )
    b = refine_saved_bestcut(
        path, mst_dir=mst_dir, dest=dest, metrics=("l2",),
        max_flip_sweeps=5, flip_seed=11,
    )
    assert len(list((dest / "cases").glob("*.json"))) == 2
    assert len(list((dest / "archives").iterdir())) == 2
    assert a[0]["refined_cut"] >= a[0]["initial_cut"]
    assert b[0]["refined_cut"] >= b[0]["initial_cut"]


def test_missing_and_ambiguous_factors_are_not_silently_selected(tmp_path):
    path, mst_dir, dest = _saved_star_tree_bestcut(tmp_path)
    with pytest.raises(FileNotFoundError, match="No archived MST-BestCut"):
        _find_bestcut_record(mst_dir, "G11", "l1")
    first_file = next((mst_dir / "center-cases").glob("*-l2-bestcut.json"))
    original = json.loads(first_file.read_text(encoding="utf8"))
    duplicate = dict(original, factor_sha256="b" * 64)
    second_file = mst_dir / "center-cases" / (
        "G11-bbbbbbbbbbbbbbbb-mst-l2-bestcut.json"
    )
    second_file.write_text(json.dumps(duplicate), encoding="utf8")
    with pytest.raises(ValueError, match="versions"):
        refine_saved_bestcut(
            path, mst_dir=mst_dir, dest=dest, metrics=("l2",),
        )
    selected = _find_bestcut_record(mst_dir, "G11", "l2", "a" * 10)
    assert selected[1]["factor_sha256"] == "a" * 64


def test_signed_weight_local_search_never_decreases_objective(tmp_path):
    path = tmp_path / "G11.txt"
    path.write_text(
        "5 7\n1 2 2\n1 3 -1\n1 4 1\n2 3 2\n"
        "2 5 1\n3 4 -3\n4 5 3\n",
        encoding="utf8",
    )
    graph = load_graph(path)
    edges = edge_arrays(graph)
    start = np.array([0, 0, 0, 1, 1], dtype=np.int8)
    score_before = cut_weight(edges, start)
    after, diagnostic = improve_by_flips(
        graph, start, np.random.default_rng(2026), max_sweeps=20,
    )
    assert cut_weight(edges, after) >= score_before
    assert diagnostic["cut_weight"] >= diagnostic["initial_cut"]
