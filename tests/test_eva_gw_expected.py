"""Experiment014 GW expected-raw-rounding factor optimization tests.

Tests distinguish the mathematical expectation from actual sampled
signed Max-Cut and verify no K-Means or EVA re-solve happens.
"""
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
from maxkcut.eva_gw_expected import (
    gw_expected_cut, optimize_gw_factor, raw_gw_pair,
)
from maxkcut.eva_gw_expected_cli import run_graph
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight


def _signed_graph(n=8, seed=42):
    rng = np.random.default_rng(seed)
    a = rng.integers(-3, 5, size=(n, n)).astype(np.float64)
    a[np.tril_indices(n)] = 0
    a = a + a.T
    return sparse.csr_matrix(a)


def _unit_factor(n=8, p=4, seed=21):
    y = np.random.default_rng(seed).standard_normal((n, p))
    return y / np.linalg.norm(y, axis=1)[:, None]


def test_exact_analytical_GW_expectation_for_signed_two_vertex_graph():
    y = np.asarray([[1., 0.], [0., 1.]])
    positive = (np.array([0]), np.array([1]), np.array([4.]))
    negative = (np.array([0]), np.array([1]), np.array([-6.]))
    assert gw_expected_cut(positive, y) == pytest.approx(2.)
    assert gw_expected_cut(negative, y) == pytest.approx(-3.)
    # Correlated vectors + epsilon noise have the *exact* stated
    # augmented Gaussian correlation.
    z = np.array([[1., 0.], [0.8, 0.6]])
    epsilon = .1
    expected = 4*np.arccos((1-epsilon)*.8)/np.pi
    assert gw_expected_cut(positive, z, epsilon=epsilon) == pytest.approx(
        expected, abs=1e-10,
    )


@pytest.mark.parametrize("epsilon", [0., .035])
def test_signed_expected_cut_matches_raw_GW_monte_carlo(epsilon):
    g = _signed_graph(n=6, seed=6)
    edges = edge_arrays(g)
    y = _unit_factor(6, 3, 4)
    expect = gw_expected_cut(edges, y, epsilon=epsilon)
    rng = np.random.default_rng(8)
    samples = [
        cut_weight(edges, raw_gw_pair(y, y, rng, epsilon=epsilon)[0])
        for _ in range(5000)
    ]
    assert abs(np.mean(samples) - expect) < 0.5


@pytest.mark.parametrize("epsilon,jitter", [
    (0., 0.), (.02, 0.), (0., .07), (.08, .1),
])
def test_expected_factor_optimization_is_non_decreasing(epsilon, jitter):
    g = _signed_graph(9, 12)
    y = _unit_factor(9, 5, 26)
    old = y.copy()
    result, diag = optimize_gw_factor(
        g, y, np.random.default_rng(72),
        epsilon=epsilon, max_sweeps=13,
        step=.12, jitter=jitter, patience=6,
    )
    np.testing.assert_array_equal(y, old)
    assert np.allclose(np.linalg.norm(result, axis=1), 1)
    old_expected = gw_expected_cut(edge_arrays(g), y, epsilon=epsilon)
    new_expected = gw_expected_cut(edge_arrays(g), result, epsilon=epsilon)
    assert diag["initial_expected_cut"] == pytest.approx(old_expected)
    assert diag["optimized_expected_cut"] == pytest.approx(new_expected)
    assert new_expected >= old_expected - 1e-7
    assert all(b["best_expected"] >= a["best_expected"] - 1e-8
               for a, b in zip(diag["trace"], diag["trace"][1:]))
    replay, rdiag = optimize_gw_factor(
        g, y, np.random.default_rng(72),
        epsilon=epsilon, max_sweeps=13,
        step=.12, jitter=jitter, patience=6,
    )
    np.testing.assert_array_equal(result, replay)
    assert rdiag["optimized_expected_cut"] == diag["optimized_expected_cut"]


def test_optimizer_tangent_gradient_agrees_with_finite_difference():
    """Gradient is for *the original expected GW cut*, not EVA's surrogate."""
    from maxkcut.eva_gw_expected import _compiled
    g = _signed_graph(7, 22)
    edges = edge_arrays(g)
    y = _unit_factor(7, 4, 33)
    u, v, weights = edges
    epsilon = .03
    obj, tangent = _compiled()(
        u.astype(np.int64), v.astype(np.int64),
        weights.astype(np.float64), y.copy(), epsilon, True,
    )
    a = 2
    vec = np.random.default_rng(17).normal(size=y.shape[1])
    vec -= (vec@y[a])*y[a]
    vec /= np.linalg.norm(vec)
    h = 1e-5
    pos, neg = y.copy(), y.copy()
    pos[a] = (y[a]+h*vec)/np.linalg.norm(y[a]+h*vec)
    neg[a] = (y[a]-h*vec)/np.linalg.norm(y[a]-h*vec)
    finite = (
        gw_expected_cut(edges, pos, epsilon=epsilon) -
        gw_expected_cut(edges, neg, epsilon=epsilon)
    )/(2*h)
    assert float(tangent[a]@vec) == pytest.approx(finite, abs=1e-6)


def _completed_small_eva(tmp_path):
    path = tmp_path / "G11.txt"
    path.write_text(
        "7 10\n1 2 2\n1 3 -1\n1 4 1\n2 3 2\n"
        "2 5 1\n3 4 -2\n4 5 3\n4 6 2\n5 7 -1\n6 7 2\n",
        encoding="utf8",
    )
    source = tmp_path / "eva"
    args = SimpleNamespace(
        max_n=20, output_dir=source, rank=None,
        tolerance=1e-6, max_sweeps=12, seed=2026,
        checkpoint_every=2, rounding_seed=2026,
        restarts=2, cpu_threads=1, convex_dir=None,
        push=False, push_branch=None, push_every=0,
    )
    run_one(path, args)
    return path, source


def test_full_parallel_factor_and_raw_GW_archives_resume(tmp_path):
    path, source = _completed_small_eva(tmp_path)
    source_factor = source / "factors" / "G11.npz"
    original_bytes = source_factor.read_bytes()
    dest = tmp_path / "expected-gw"
    options = dict(
        factor_dir=source, dest=dest,
        epsilons=(0., .05), jitters=(0.,),
        steps=(.08,), optimize_seeds=(2026,),
        num_rounds=12, kl_top=4, kl_passes=4,
        max_sweeps=8, patience=4,
        rounding_seed=2026, workers=2, target=500.,
    )
    original_edges = edge_arrays(load_graph(path))
    first = run_graph(path, **options)
    assert len(first) == 2
    assert source_factor.read_bytes() == original_bytes
    assert len(list((dest/"cases").glob("*.json"))) == 2
    for record in first:
        assert Path(record["optimized_factor"]).is_file()
        assert hashlib.sha256(
            Path(record["optimized_factor"]).read_bytes(),
        ).hexdigest() == record["optimized_factor_sha256"]
        assert record["optimized_expected_cut"] >= (
            record["source_expected_cut"] - 1e-8
        )
        report = verify_archive(record["archive_path"])
        assert report["status"] == "complete"
        assert report["graph_sha256"] == hashlib.sha256(
            path.read_bytes(),
        ).hexdigest()
        for stage, count in (
            ("OldRawGW", 12), ("NewRawGW", 12),
            ("OldRawGWKL", 4), ("NewRawGWKL", 4),
        ):
            method = f"Exp014-{stage}_EVA"
            assert report["methods"][method]["verified_partitions"] == count
            labels, scores = load_method(
                Path(record["archive_path"])/f"{method}.npz",
            )
            for row, score in zip(labels, scores):
                assert cut_weight(original_edges, row) == score
    count_before = len(list((dest/"archives").iterdir()))
    second = run_graph(path, **options)
    assert [x["best_cut_by_stage"] for x in first] == [
        x["best_cut_by_stage"] for x in second
    ]
    assert len(list((dest/"archives").iterdir())) == count_before
    assert (dest/"summary.csv").is_file()


def test_gw_pair_uses_same_random_direction_for_old_new():
    y = _unit_factor(9, 4, 6)
    a, b = raw_gw_pair(y, y, np.random.default_rng(7), epsilon=0.)
    np.testing.assert_array_equal(a, b)
    a, b = raw_gw_pair(y, y, np.random.default_rng(7), epsilon=.03)
    np.testing.assert_array_equal(a, b)
