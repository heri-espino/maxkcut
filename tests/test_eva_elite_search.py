"""Experiment011: indexed-heap KL correctness, elite diversification, archives."""
from __future__ import annotations

import hashlib
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
from maxkcut.eva_escape import optimize_kl
from maxkcut.eva_elite_search import (
    _aligned_partner, _elite_mix, optimize_elite, optimize_kl_fast,
)
from maxkcut.eva_elite_cli import (
    _source_kl, _select_distinct_elites, run_graph, write_summary,
)
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight


def _signed_graph(n=9, seed=12):
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 5, size=(n, n)).astype(float)
    a[np.tril_indices(n, -1)] = 0
    a = a + a.T
    np.fill_diagonal(a, 0)
    a[a == -0.0] = 0
    return sparse.csr_matrix(a)


@pytest.mark.parametrize("seed", [0, 1, 11, 27])
def test_heap_kl_exactly_matches_legacy_kl(seed):
    """Heap must give same signed Max-Cut result and labels as full scan."""
    g = _signed_graph(n=9, seed=seed+80)
    start = np.random.default_rng(seed+200).integers(
        0, 2, size=g.shape[0], dtype=np.int8,
    )
    old, old_diag = optimize_kl(
        g, start, np.random.default_rng(seed+1), passes=8,
    )
    fast, fast_diag = optimize_kl_fast(
        g, start, np.random.default_rng(seed+1), passes=8,
    )
    np.testing.assert_array_equal(fast, old)
    assert fast_diag["cut_weight"] == pytest.approx(old_diag["cut_weight"])
    assert fast_diag["passes_improved"] == old_diag["passes_improved"]
    assert fast_diag["flips_retained"] == old_diag["flips_retained"]
    assert fast_diag["cut_weight"] >= fast_diag["initial_cut"]


def test_heap_kl_matches_legacy_for_dense_signed_and_tied_edges():
    graph = sparse.csr_matrix(
        np.ones((7, 7), dtype=np.float64) -
        np.eye(7, dtype=np.float64)
    )
    start = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    for seed in [0, 2, 3, 4]:
        old, _ = optimize_kl(
            graph, start, np.random.default_rng(seed), passes=5,
        )
        fast, _ = optimize_kl_fast(
            graph, start, np.random.default_rng(seed), passes=5,
        )
        np.testing.assert_array_equal(old, fast)


@pytest.mark.parametrize("method", [
    "ILS-KL", "EliteMix-KL", "Tabu-KL",
])
def test_extended_methods_monotone_best_and_explore_signed_graph(method):
    g = _signed_graph(n=12, seed=17)
    original_edges = edge_arrays(g)
    rng = np.random.default_rng(47)
    starts = [
        rng.integers(0, 2, size=g.shape[0], dtype=np.int8)
        for _ in range(4)
    ]
    elites = [
        (x.copy(), float(cut_weight(original_edges, x)))
        for x in starts
    ]
    start = starts[0].copy()
    best, detail = optimize_elite(
        g, start, elites, np.random.default_rng(18),
        method=method, cycles=9, kl_passes=5,
        kick_sizes=(2, 3, 4), tabu_steps=80,
        tabu_tenure=3, tabu_stagnation=40, target=10000.,
    )
    expected = float(cut_weight(original_edges, start))
    actual = float(cut_weight(original_edges, best))
    assert actual == detail["cut_weight"]
    assert actual >= expected
    assert actual == detail["initial_cut"] + detail["improvement"]
    assert detail["cycles_completed"] == 9
    assert detail["termination"] == "cycle_limit"
    assert all(
        b["best_cut"] >= a["best_cut"]
        for a, b in zip(detail["trace"], detail["trace"][1:])
    )
    # Replay with the same seed, ignoring only nondeterministic timing.
    repeat, repeated = optimize_elite(
        g, start, elites, np.random.default_rng(18),
        method=method, cycles=9, kl_passes=5,
        kick_sizes=(2, 3, 4), tabu_steps=80,
        tabu_tenure=3, tabu_stagnation=40, target=10000.,
    )
    np.testing.assert_array_equal(best, repeat)
    assert detail["cut_weight"] == repeated["cut_weight"]


def test_complement_alignment_prevents_spurious_distance():
    a = np.array([0, 1, 0, 1, 0, 0], dtype=np.int8)
    np.testing.assert_array_equal(_aligned_partner(a, 1-a), a)
    proposal, changes = _elite_mix(
        a, 1-a, np.random.default_rng(4), 0.5,
    )
    assert changes == 0
    np.testing.assert_array_equal(proposal, a)


def _synthetic_verified_exp010(tmp_path):
    graph_path = tmp_path / "G11.txt"
    g = _signed_graph(n=9, seed=39)
    i, j, w = edge_arrays(g)
    graph_path.write_text(
        f"9 {len(w)}\n" +
        "".join(
            f"{u+1} {v+1} {int(weight)}\n"
            for u, v, weight in zip(i, j, w)
        ),
        encoding="utf8",
    )
    g = load_graph(graph_path)
    scores_edges = edge_arrays(g)
    start = np.random.default_rng(72)
    starts = [
        start.integers(0, 2, size=9, dtype=np.int8)
        for _ in range(6)
    ]
    # The warm starts are already 1-flip (or KL) improved, and archive
    # contains exactly the saved result for each synthetic restart.
    improved = [
        optimize_kl_fast(
            g, partition, np.random.default_rng(900+i), passes=5,
        )[0] for i, partition in enumerate(starts)
    ]
    scores = [float(cut_weight(scores_edges, x)) for x in improved]
    source = tmp_path / "experiment010"
    name = "K-MeansNM-Flip_EVA-KL_EVA"
    archived = new_archive(
        source / "archives", new_run_id(), graph_path, 9,
        scores_edges, 2, {"synthetic": True},
        jsonl_output=source / "dummy.jsonl",
    )
    save_method(archived, name, improved, scores, k=2)
    update_manifest(archived, status="complete")
    assert verify_archive(archived["path"])["status"] == "complete"
    factor = "d"*64
    record = {
        "status": "complete",
        "graph": "G11",
        "method": "KL",
        "source_method": "K-MeansNM-Flip_EVA",
        "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "factor_sha256": factor,
        "settings": {
            "seed": 2026, "kl_passes": 30,
            "source_restarts": list(range(6)),
        },
        "archive_method": name,
        "archive_path": str(archived["path"]),
        "trials": [
            {"restart": i, "initial_cut": scores[i],
             "final_cut": scores[i]} for i in range(6)
        ],
    }
    folder = source / "cases"
    folder.mkdir(parents=True)
    (folder / "G11-K-MeansNM-Flip_EVA-KL-synthetic.json").write_text(
        json.dumps(record), encoding="utf8",
    )
    return graph_path, source, tmp_path / "experiment011"


def test_kl_archive_source_provenance_and_distinct_elites(tmp_path):
    path, source, dest = _synthetic_verified_exp010(tmp_path)
    factor, data, provenance = _source_kl(
        path, source, source_seed=2026, source_passes=30,
    )
    assert factor == "d"*64
    assert len(data) == 6
    assert len(provenance) == 1
    chosen = _select_distinct_elites(data, 5)
    assert 1 <= len(chosen) <= 5
    assert chosen[0]["initial_cut"] == max(x[1] for x in data.values())
    with pytest.raises(FileNotFoundError, match="KL results absent"):
        _source_kl(path, source, source_seed=4444, source_passes=30)


def test_parallel_resumable_episodes_verify_all_archived_cuts(tmp_path):
    path, source, dest = _synthetic_verified_exp010(tmp_path)
    options = dict(
        source_dir=source, dest=dest,
        methods=("ILS-KL", "EliteMix-KL"), top_elites=3,
        replicates=2, chunk_size=2, workers=2,
        cycles=5, kl_passes=3, kick_sizes=(2,3,4),
        tabu_steps=60, tabu_tenure=3, tabu_stagnation=20,
        target=9999.,
    )
    first = run_graph(path, **options)
    assert first["target"] == 9999.
    assert "ILS-KL" in first["methods"]
    assert "EliteMix-KL" in first["methods"]
    for method, stats in first["methods"].items():
        assert stats["episodes"] >= 2
        assert stats["best"] >= first["selected_best"] - 1e-8
    original_graph = edge_arrays(load_graph(path))
    archives = list((dest / "cases").glob("*.json"))
    assert len(archives) >= 2
    for file in archives:
        record = json.loads(file.read_text(encoding="utf8"))
        checked = verify_archive(record["archive_path"])
        assert checked["status"] == "complete"
        method = record["archive_method"]
        labels, scores = load_method(
            Path(record["archive_path"]) / f"{method}.npz"
        )
        assert len(scores) == len(record["episodes"])
        for j, trial in enumerate(record["episodes"]):
            assert trial["final_cut"] == scores[j]
            assert trial["final_cut"] >= trial["initial_cut"]
            assert cut_weight(original_graph, labels[j]) == scores[j]
    previous = len(list((dest / "archives").iterdir()))
    second = run_graph(path, **options)
    assert first["methods"] == second["methods"]
    assert len(list((dest / "archives").iterdir())) == previous
    summary_path = write_summary(dest, target=9999.)
    lines = summary_path.read_text(encoding="utf8").splitlines()
    assert len(lines) > 1
