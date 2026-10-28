"""Experiment012: exact signed threshold optimization and paired KL stage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("numba")

from maxkcut.archive import load_method, verify_archive
from maxkcut.eva_cli import run_one
from maxkcut.eva_cutaware_kmeans import (
    improve_kmeans_direction, optimal_threshold,
)
from maxkcut.eva_cutaware_kmeans_cli import METHODS, run_graph, write_summary
from maxkcut.eva_rounding_cli import _round_one, round_saved_factor
from maxkcut.eva import load_eva_factor
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight


def _signed_graph(n=9, seed=10):
    rng = np.random.default_rng(seed)
    a = rng.integers(-5, 6, size=(n, n)).astype(float)
    a[np.tril_indices(n)] = 0
    a += a.T
    return sparse.csr_matrix(a)


@pytest.mark.parametrize("seed", [0, 1, 7, 11, 29])
def test_exact_threshold_matches_brute_force_for_signed_graph(seed):
    rng = np.random.default_rng(seed)
    graph = _signed_graph(12, seed+11)
    projection = rng.normal(size=12)
    # Exact ties restrict legal thresholds: whole groups move together.
    projection[:4] = -2
    projection[5:9] = 0
    result, diag = optimal_threshold(graph, projection)
    observed = float(cut_weight(edge_arrays(graph), result))
    scores = [float(cut_weight(
        edge_arrays(graph), np.asarray(projection <= value, dtype=np.int8),
    )) for value in np.unique(projection)]
    scores.append(0.)
    assert observed == pytest.approx(max(scores), abs=1e-8)
    assert diag["cut"] == observed
    assert diag["group_sizes"] == np.bincount(result, minlength=2).tolist()
    # Threshold decisions are monotone by ordering.
    order = np.argsort(projection, kind="stable")
    assert not np.any(np.diff(result[order]) > 0)


def test_negative_graph_can_select_empty_partition_and_equal_projections():
    a = -np.ones((5, 5))
    np.fill_diagonal(a, 0)
    g = sparse.csr_matrix(a)
    partition, diag = optimal_threshold(g, np.ones(5))
    assert diag["cut"] == 0.
    assert diag["sorted_prefix"] == 0
    assert np.all(partition == 0)


@pytest.mark.parametrize("seed", [2, 5, 13])
def test_cut_aware_kmeans_preserves_same_seed_raw_kmeans(seed):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(14, 7))
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    graph = _signed_graph(14, seed)
    graph_edges = edge_arrays(graph)
    raw_labels, iterations = _round_one(
        vectors, graph, graph_edges, "K-MeansNM_EVA",
        11, 0, seed,
    )
    normalized_rounding_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 11, 2, 0, 0])
    )
    result, detail = improve_kmeans_direction(
        vectors, graph, normalized_rounding_rng,
        direction_trials=18, kmeans_iters=100,
    )
    assert detail["raw_kmeans_best_cut"] == pytest.approx(
        cut_weight(graph_edges, raw_labels),
    )
    assert detail["cut"] >= detail["raw_kmeans_best_cut"] - 1e-8
    assert cut_weight(graph_edges, result) == detail["cut"]
    assert detail["direction_candidates"] >= detail["kmeans_iterations"]
    assert detail["uses_full_eva_dimensions"] is True


def _synthetic_eva_and_roundings(tmp_path):
    path = tmp_path / "G11.txt"
    path.write_text(
        "7 10\n1 2 2\n1 3 -1\n1 4 1\n2 3 2\n"
        "2 5 1\n3 4 -2\n4 5 3\n4 6 2\n5 7 -1\n6 7 2\n",
        encoding="utf8",
    )
    factor_dir = tmp_path / "eva"
    args = SimpleNamespace(
        max_n=20, output_dir=factor_dir, rank=None,
        tolerance=1e-6, max_sweeps=16, seed=2026,
        checkpoint_every=2, rounding_seed=2026,
        restarts=2, cpu_threads=1, convex_dir=None,
        push=False, push_branch=None, push_every=0,
    )
    run_one(path, args)
    rounding_dir = tmp_path / "rounding"
    round_saved_factor(
        path, source=factor_dir, dest=rounding_dir,
        methods=["K-MeansNM-Flip_EVA"], first=0, restarts=6,
        max_flip_sweeps=10,
    )
    return path, factor_dir, rounding_dir, tmp_path / "cut-aware"


def test_paired_control_concurrent_rounding_kl_and_archive_resume(tmp_path):
    path, factor_dir, rounding_dir, dest = _synthetic_eva_and_roundings(tmp_path)
    original_factor = (factor_dir / "factors" / "G11.npz").read_bytes()
    graph = load_graph(path)
    original_edges = edge_arrays(graph)
    options = dict(
        factor_dir=factor_dir, rounding_dir=rounding_dir,
        dest=dest, top_starts=4, chunk_size=2, workers=2,
        direction_trials=8, kmeans_iters=100,
        perturb_scales=(0.03, 0.12),
        kl_passes=5, seed=2026, target=10000.,
    )
    first = run_graph(path, **options)
    assert first["selected_count"] == 4
    assert (factor_dir / "factors" / "G11.npz").read_bytes() == original_factor
    assert len(list((dest / "cases").glob("G*.json"))) == 2
    assert len(list((dest / "archives").iterdir())) == 2
    original = {}
    for case_path in (dest / "cases").glob("G*.json"):
        case = json.loads(case_path.read_text(encoding="utf8"))
        checked = verify_archive(case["archive_path"])
        assert checked["status"] == "complete"
        for name in METHODS:
            archive_name = f"Exp012-{name}_EVA"
            assert checked["methods"][archive_name]["verified_partitions"] == 2
            parts, scores = load_method(
                Path(case["archive_path"]) / f"{archive_name}.npz"
            )
            for trial_index, trial in enumerate(case["trials"]):
                assert cut_weight(original_edges, parts[trial_index]) == scores[trial_index]
                assert trial["scores"][name] == scores[trial_index]
                original.setdefault(trial["restart"], {})[name] = scores[trial_index]
        for trial in case["trials"]:
            scores = trial["scores"]
            assert scores["KMeansNMFlipKL"] >= scores["KMeansNMFlipInput"]
            assert scores["CutAwareKMeansKL"] >= scores["CutAwarePreKL"]
            assert scores["CutAwarePreKL"] >= trial["kmeans_raw_cut"]
    second = run_graph(path, **options)
    assert second["stages"] == first["stages"]
    assert len(list((dest / "archives").iterdir())) == 2
    written = write_summary(dest, target=10000.)
    assert len(written.read_text(encoding="utf8").splitlines()) == 1 + 4*4


def test_poor_parameters_fail_and_keep_sources(tmp_path):
    path, factor_dir, rounding_dir, dest = _synthetic_eva_and_roundings(tmp_path)
    graph = load_graph(path)
    y, _ = load_eva_factor(factor_dir / "factors" / "G11.npz")
    with pytest.raises(ValueError):
        optimal_threshold(graph, np.zeros(2))
    with pytest.raises(ValueError):
        improve_kmeans_direction(y, graph, np.random.default_rng(3),
                                 perturb_scales=[0])
    with pytest.raises(ValueError):
        run_graph(path, factor_dir=factor_dir, rounding_dir=rounding_dir,
                  dest=dest, top_starts=-1)
