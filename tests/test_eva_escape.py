"""Experiment010: KL and Tabu correctness on signed and archived instances.

Both methods are allowed non-improving INTERMEDIATE moves, but the
RETURNED best-so-far cut must score at least the saved starting cut
under the ORIGINAL signed weighted graph. Independent verification
also checks exact bit-packed partition archives and Windows-safe
two-process execution on identical seeded source partitions.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("numba")

from maxkcut.archive import (
    load_method, new_archive, new_run_id, save_method,
    update_manifest, verify_archive,
)
from maxkcut.eva_escape import optimize_kl, optimize_tabu
from maxkcut.eva_escape_cli import run_graph, _read_initials, write_summary
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight


def _graph_signed():
    # Signed graph with a known 1-flip local optimum that is NOT
    # globally optimal: [0,0,1,0] has cut 1; another cut attains 2.
    raw = [(0, 1, -1.), (0, 3, -4.),
           (1, 2, 1.), (1, 3, 2.)]
    u = np.asarray([i for i, j, w in raw] +
                   [j for i, j, w in raw])
    v = np.asarray([j for i, j, w in raw] +
                   [i for i, j, w in raw])
    w = np.asarray([wt for i, j, wt in raw] * 2)
    graph = sparse.coo_matrix((w, (u, v)), shape=(4, 4)).tocsr()
    return graph


def _all_flip_gains(graph, partition):
    edges = edge_arrays(graph)
    base = cut_weight(edges, partition)
    return np.array([
        cut_weight(edges, np.array([
            (1-label if j == i else label)
            for j, label in enumerate(partition)
        ])) - base
        for i in range(len(partition))
    ])


@pytest.mark.parametrize("method", ["KL", "Tabu"])
def test_signed_heuristics_are_independently_scored_and_reproducible(method):
    g = _graph_signed()
    first = np.asarray([0, 0, 1, 0], dtype=np.int8)
    assert cut_weight(edge_arrays(g), first) == 1.
    assert _all_flip_gains(g, first).max() <= 0.
    def solve():
        rng = np.random.default_rng(7)
        return (
            optimize_kl(g, first, rng, passes=10)
            if method == "KL"
            else optimize_tabu(
                g, first, rng, steps=500, tenure=2, stagnation=100,
            )
        )
    labels, diag = solve()
    other, other_diag = solve()
    np.testing.assert_array_equal(labels, other)
    assert diag["cut_weight"] == other_diag["cut_weight"]
    actual = cut_weight(edge_arrays(g), labels)
    assert actual == diag["cut_weight"]
    assert actual >= diag["initial_cut"]
    assert actual == diag["initial_cut"] + diag["improvement"]
    assert np.all(np.isin(labels, [0, 1]))
    if method == "Tabu":
        assert diag["nonpositive_moves"] > 0
        assert 1 <= diag["steps"] <= 500
        assert diag["termination"] in ("stagnation", "step_limit")
    else:
        assert diag["termination"] in (
            "no_improving_gain_sequence", "pass_limit",
        )


def test_empty_pass_returns_original_labels_without_losing_cut():
    # Complete +1 K4; balanced cut is globally optimal.
    arr = np.ones((4, 4), dtype=np.float64)
    np.fill_diagonal(arr, 0)
    g = sparse.csr_matrix(arr)
    start = np.asarray([0, 1, 0, 1], dtype=np.int8)
    end, diag = optimize_kl(g, start, np.random.default_rng(3), passes=4)
    assert diag["initial_cut"] == 4.
    assert diag["cut_weight"] == 4.
    np.testing.assert_array_equal(start, end)
    assert diag["passes_improved"] == 0


def test_binary_input_and_budget_validation():
    g = _graph_signed()
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="binary"):
        optimize_kl(g, np.array([0, 2, 0, 1]), rng)
    with pytest.raises(ValueError, match=">= 1"):
        optimize_kl(g, np.array([0, 1, 0, 1]), rng, passes=0)
    with pytest.raises(ValueError, match=">= 1"):
        optimize_tabu(g, np.array([0, 1, 0, 1]), rng, tenure=0)


def _source_archives(tmp_path):
    graph = _graph_signed()
    source_graph = tmp_path / "G11.txt"
    source_graph.write_text(
        "4 4\n1 2 -1\n1 4 -4\n2 3 1\n2 4 2\n",
        encoding="utf8",
    )
    loaded = load_graph(source_graph)
    assert loaded.shape == graph.shape
    input_sha = hashlib.sha256(source_graph.read_bytes()).hexdigest()
    factor_sha = "d" * 64
    source = tmp_path / "roundings"
    method = "K-MeansNM-Flip_EVA"
    labels = [
        np.array(x, dtype=np.int8) for x in itertools.islice(
            itertools.product([0, 1], repeat=4), 6
        )
    ]
    scores = [float(cut_weight(edge_arrays(loaded), x)) for x in labels]
    archive = new_archive(
        source / "archives", new_run_id(), source_graph, 4,
        edge_arrays(loaded), 2, {"test": True},
        jsonl_output=source / "dummy.jsonl",
    )
    save_method(archive, method, labels, scores, k=2)
    update_manifest(archive, status="complete")
    root = source / "segments" / f"G11-{factor_sha[:16]}" / method
    root.mkdir(parents=True)
    segment = {
        "graph": "G11", "graph_sha256": input_sha,
        "factor_sha256": factor_sha, "method": method,
        "rounding_seed": 2026, "max_flip_sweeps": 20,
        "start_restart": 0, "stop_restart_exclusive": 6,
        "archive_path": str(archive["path"]),
    }
    (root / "r00000-r00006.json").write_text(
        json.dumps(segment), encoding="utf8",
    )
    return source_graph, source, tmp_path / "refined"


def test_parallel_workers_start_identically_and_archive_all_trials(tmp_path):
    path, source, dest = _source_archives(tmp_path)
    sha, parts, provenance = _read_initials(
        path, source, "K-MeansNM-Flip_EVA",
    )
    assert len(parts) == 6
    assert len(provenance) == 1
    options = dict(
        source_dir=source, output_dir=dest,
        methods=("KL", "Tabu"), top_starts=4,
        chunk_size=2, workers=2, seed=2026, kl_passes=3,
        tabu_steps=50, tabu_tenure=2, tabu_stagnation=20,
        benchmark=2.,
    )
    first = run_graph(path, **options)
    assert first["source_count"] == 6
    assert len(first["selected_restart_ids"]) == 4
    assert first["methods"]["KL"]["count"] == 4
    assert first["methods"]["Tabu"]["count"] == 4
    assert len(list((dest / "cases").glob("G*.json"))) == 4
    all_indices = {}
    for file in (dest / "cases").glob("G*.json"):
        record = json.loads(file.read_text(encoding="utf8"))
        method = record["method"]
        all_indices.setdefault(method, set()).update(
            trial["restart"] for trial in record["trials"]
        )
        checked = verify_archive(record["archive_path"])
        assert checked["status"] == "complete"
        assert checked["methods"][record["archive_method"]][
            "verified_partitions"] == 2
        final_labels, scores = load_method(
            Path(record["archive_path"]) /
            f"{record['archive_method']}.npz"
        )
        for j, trial in enumerate(record["trials"]):
            assert trial["initial_cut"] == parts[trial["restart"]][1]
            assert trial["final_cut"] == scores[j]
            assert trial["final_cut"] >= trial["initial_cut"]
            assert cut_weight(edge_arrays(load_graph(path)), final_labels[j]) \
                == trial["final_cut"]
    assert all(
        all_indices[method] == set(first["selected_restart_ids"])
        for method in ("KL", "Tabu")
    )
    second = run_graph(path, **options)
    assert first["methods"] == second["methods"]
    assert len(list((dest / "archives").iterdir())) == 4
    summary = write_summary(dest, benchmark=2.)
    assert len(summary.read_text(encoding="utf8").splitlines()) == 9


def test_duplicate_start_segments_fail_loudly(tmp_path):
    path, source, dest = _source_archives(tmp_path)
    original = next(source.rglob("r00000-r00006.json"))
    second = original.with_name("r00000-r00005.json")
    second.write_text(original.read_text(encoding="utf8"), encoding="utf8")
    with pytest.raises(ValueError, match="Overlapping"):
        _read_initials(path, source, "K-MeansNM-Flip_EVA")
