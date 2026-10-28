"""Experiment015: direct hyperplane rounding + stochastic gradient correctness."""
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
from maxkcut.eva_direct_rounding import (
    optimize_hyperplane, optimize_semicircle,
)
from maxkcut.eva_rounding_sga import (
    optimize_es_sga, optimize_soft_sga,
    soft_signed_cut_and_gradient,
)
from maxkcut.eva_direct_rounding_cli import run_graph, STAGES
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight


def _signed_graph(n=10, seed=13):
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 6, size=(n, n)).astype(np.float64)
    a[np.tril_indices(n)] = 0.
    a = a+a.T
    return sparse.csr_matrix(a)


def _factor(n=10, p=5, seed=7):
    y = np.random.default_rng(seed).normal(size=(n,p))
    return y/np.linalg.norm(y, axis=1)[:,None]


@pytest.mark.parametrize("seed", [1, 2, 9, 33])
def test_exact_semicircle_signed_cut_dominates_grid_and_source(seed):
    g = _signed_graph(11, seed)
    y = _factor(11, 5, seed+3)
    rng = np.random.default_rng(seed)
    r = rng.normal(size=5)
    r /= np.linalg.norm(r)
    v = rng.normal(size=5)
    v -= float(v@r)*r
    v /= np.linalg.norm(v)
    improved_normal, improved_labels, diagnostics = optimize_semicircle(
        g, y, r, v,
    )
    edges = edge_arrays(g)
    score = float(cut_weight(edges, improved_labels))
    source = float(cut_weight(edges, (y@r>0).astype(np.int8)))
    assert score >= source
    assert score == diagnostics["realized_arc_cut"]
    assert diagnostics["predicted_arc_max"] >= score-1e-8
    np.testing.assert_array_equal(
        improved_labels, (y@improved_normal>0).astype(np.int8),
    )
    # Exhaustive collection of intervals (the exact sweep tests all)
    # should dominate a dense independent numeric angle grid.
    best_grid = max(
        cut_weight(edges, (
            y@(np.cos(theta)*r+np.sin(theta)*v)>0
        ).astype(np.int8))
        for theta in np.linspace(-1.56, 1.56, 301)
    )
    assert score >= best_grid-1e-8


def test_exact_semicircle_moves_equal_angle_roots_as_one_group():
    g = _signed_graph(7, 13)
    y = _factor(7, 4, 16)
    y[1] = y[0]
    y[3] = y[2]
    r = np.array([1.,0.,0.,0.])
    v = np.array([0.,1.,0.,0.])
    normal, labels, info = optimize_semicircle(g, y, r, v)
    assert info["distinct_events"] < info["events"]
    assert info["realized_arc_cut"] >= info["initial_cut"]-1e-8
    np.testing.assert_array_equal(
        labels, (y@normal>0).astype(np.int8),
    )


def test_stochastic_soft_cut_exact_gradient_finite_difference_signed():
    g = _signed_graph(9, 30)
    y = _factor(9, 4, 31)
    r = np.array([.3,-.4,.1,.6])
    score, gradient = soft_signed_cut_and_gradient(
        g, y, r, .23,
    )
    h = 1e-6
    for j in range(len(r)):
        rplus = r.copy()
        rminus = r.copy()
        rplus[j] += h
        rminus[j] -= h
        plus = soft_signed_cut_and_gradient(g,y,rplus,.23)[0]
        minus = soft_signed_cut_and_gradient(g,y,rminus,.23)[0]
        assert gradient[j] == pytest.approx(
            (plus-minus)/(2*h), abs=1e-6,
        )


@pytest.mark.parametrize("method", ["es-sga", "soft-sga"])
def test_stochastic_rounding_best_hard_signed_cut_and_replay(method):
    g = _signed_graph(14, 41)
    y = _factor(14, 6, 42)
    original = y.copy()
    rng = np.random.default_rng(3)
    start = rng.normal(size=y.shape[1])
    start /= np.linalg.norm(start)
    init_cut = float(cut_weight(
        edge_arrays(g), (y@start>0).astype(np.int8),
    ))
    def run():
        if method == "es-sga":
            return optimize_es_sga(
                g, y, np.random.default_rng(2026),
                initial=start, steps=24, population=6,
                sigma_start=.4, sigma_end=.05,
                learning_rate=.09, restart_patience=9,
                target=10000.,
            )
        return optimize_soft_sga(
            g, y, np.random.default_rng(2026),
            initial=start, steps=24, edge_batch=32,
            temperature_start=.5, temperature_end=.04,
            learning_rate=.09, restart_patience=9,
            target=10000.,
        )
    best_r, labels, details = run()
    again_r, again_labels, again = run()
    assert np.array_equal(y, original)
    np.testing.assert_array_equal(best_r, again_r)
    np.testing.assert_array_equal(labels, again_labels)
    assert details["optimized_raw_cut"] == again["optimized_raw_cut"]
    assert len(details["trace"]) == 24
    assert details["optimized_raw_cut"] >= init_cut
    assert details["optimized_raw_cut"] == cut_weight(
        edge_arrays(g), labels,
    )
    np.testing.assert_array_equal(
        labels, (y@best_r > 0).astype(np.int8),
    )
    assert all(
        b["best_cut"] >= a["best_cut"]-1e-8
        for a,b in zip(details["trace"], details["trace"][1:])
    )


@pytest.mark.parametrize("method", ["es-sga", "soft-sga"])
def test_stochastic_input_validation(method):
    g = _signed_graph(7)
    y = _factor(7, 3)
    r = np.array([1.,0.,0.])
    if method == "es-sga":
        with pytest.raises(ValueError):
            optimize_es_sga(
                g,y,np.random.default_rng(1),initial=r,
                population=0,
            )
    else:
        with pytest.raises(ValueError):
            optimize_soft_sga(
                g,y,np.random.default_rng(1),initial=r,
                edge_batch=0,
            )


def _small_completed_eva(tmp):
    path = tmp/"G11.txt"
    path.write_text(
        "7 10\n1 2 2\n1 3 -1\n1 4 1\n2 3 2\n"
        "2 5 1\n3 4 -2\n4 5 3\n4 6 2\n5 7 -1\n6 7 2\n",
        encoding="utf8",
    )
    factor_dir = tmp/"saved-EVA"
    opts = SimpleNamespace(
        max_n=20, output_dir=factor_dir, rank=None,
        tolerance=1e-6, max_sweeps=12, seed=2026,
        checkpoint_every=2, rounding_seed=2026,
        restarts=2, cpu_threads=1, convex_dir=None,
        push=False, push_branch=None, push_every=0,
    )
    run_one(path, opts)
    return path, factor_dir


def test_parallel_all_methods_archives_and_resume(tmp_path):
    path, factor_dir = _small_completed_eva(tmp_path)
    original_file = factor_dir/"factors"/"G11.npz"
    original_sha = hashlib.sha256(original_file.read_bytes()).hexdigest()
    dest = tmp_path/"optimized-rounding"
    methods = (
        "linesweep", "es-sga", "es-sga-sweep",
        "soft-sga", "soft-sga-sweep",
    )
    options = dict(
        factor_dir=factor_dir, dest=dest, methods=methods,
        first=0, restarts=4, chunk_size=2, workers=2,
        line_directions=4, sga_steps=6, sga_population=3,
        edge_batch=6, kl_passes=3, kl_repeats=2,
        restart_patience=4, target=10000.,
    )
    first = run_graph(path, **options)
    assert len(first) == len(methods)*2
    assert hashlib.sha256(original_file.read_bytes()).hexdigest() == original_sha
    archives = sorted((dest/"cases").glob("G*.json"))
    assert len(archives) == len(methods)*2
    graph = load_graph(path)
    edges = edge_arrays(graph)
    from maxkcut.eva_rounding_cli import _factor_and_case
    _, y, _, _, _ = _factor_and_case(path, factor_dir)
    for record_file in archives:
        record = json.loads(record_file.read_text(encoding="utf8"))
        ver = verify_archive(record["archive_path"])
        assert ver["status"] == "complete"
        assert all(
            ver["methods"][f"Exp015-{stage}_EVA"][
                "verified_partitions"] == 2
            for stage in STAGES
        )
        with np.load(record["directions_path"], allow_pickle=False) as d:
            old = d["original_normals"]
            new = d["optimized_normals"]
        for stage in STAGES:
            parts, cuts = load_method(
                Path(record["archive_path"])/
                f"Exp015-{stage}_EVA.npz",
            )
            for index, (labels,cut) in enumerate(zip(parts,cuts)):
                assert cut == cut_weight(edges,labels)
                if stage == "GWRaw":
                    np.testing.assert_array_equal(
                        labels,(y@old[index]>0).astype(np.int8),
                    )
                if stage == "OptimizedRaw":
                    np.testing.assert_array_equal(
                        labels,(y@new[index]>0).astype(np.int8),
                    )
    archive_count = len(list((dest/"archives").iterdir()))
    repeated = run_graph(path, **options)
    assert [rec["best_by_stage"] for rec in repeated] == [
        rec["best_by_stage"] for rec in first
    ]
    assert len(list((dest/"archives").iterdir())) == archive_count
    assert (dest/"summary.csv").is_file()
