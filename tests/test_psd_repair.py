"""Regression tests for numerical SDP projection, motivated by G11/SCS."""
import json

import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

from maxkcut.rounding import factorize
import maxkcut.experiments as experiments


def test_scs_scale_negative_eigenvalue_can_be_repaired():
    # A diagonal-one near-correlation matrix with lambda_min = -8.37e-4.
    x = np.array([[1.0, 1.000837374814667],
                  [1.000837374814667, 1.0]])
    with pytest.warns(RuntimeWarning, match="clipped negative"):
        vectors, eigenmin, diagnostics = factorize(
            x, return_diagnostics=True
        )
    assert eigenmin == pytest.approx(-0.000837374814667, abs=1e-12)
    assert diagnostics["psd_projection_applied"]
    assert diagnostics["psd_relative_negative_eigenvalue"] < 1e-3
    assert diagnostics["psd_negative_eigenvalue_count"] == 1
    gram = vectors @ vectors.T
    assert np.allclose(np.diag(gram), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(gram).min() >= -1e-12


def test_strict_projection_rejects_negative_matrix():
    x = np.array([[1.0, 1.000837], [1.000837, 1.0]])
    with pytest.raises(ValueError, match="substantially non-PSD"):
        factorize(x, relative_negative_tolerance=0.0)


def test_substantial_infeasibility_still_rejected():
    x = np.array([[1., 2.], [2., 1.]])
    with pytest.raises(ValueError, match="substantially non-PSD"):
        factorize(x, relative_negative_tolerance=1e-3)


def test_factorize_requires_finite_square_input():
    with pytest.raises(ValueError, match="finite and square"):
        factorize(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite and square"):
        factorize(np.array([[1., np.nan], [0., 1.]]))


def test_approximate_sdp_is_cached_only_after_successful_repair(
    monkeypatch, tmp_path
):
    graph = tmp_path / "G11.mat"
    savemat(graph, {"A": sparse.csr_matrix([[0, 1], [1, 0]])})
    calls = []

    def fake_solver(a, k, solver, eps, max_iters, verbose=False, **kwargs):
        calls.append(k)
        x = np.array([[1., 1.000837], [1.000837, 1.]])
        return x, {
            "sdp_seconds": 10.0,
            "solver_status": "optimal_inaccurate",
            "sdp_objective_primal": 1.25,
        }

    monkeypatch.setattr(experiments, "solve_sdp", fake_solver)
    params = dict(
        paths=[graph], k_values=[2], restarts=1,
        cache_dir=tmp_path / "cache"
    )
    with pytest.warns(RuntimeWarning, match="clipped negative"):
        experiments.benchmark(output=tmp_path / "first.jsonl", **params)
    with pytest.warns(RuntimeWarning, match="clipped negative"):
        experiments.benchmark(output=tmp_path / "second.jsonl", **params)

    assert calls == [2]  # no second numerical SDP solve
    rows = [
        json.loads(line)
        for line in (tmp_path / "second.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert all(row["cache_hit"] for row in rows)
    assert all(row["psd_projection_applied"] for row in rows)
    assert all(row["min_eigenvalue"] < 0 for row in rows)
    assert all(row["solver_status"] == "optimal_inaccurate" for row in rows)


def test_substantial_indefiniteness_does_not_pollute_cache(monkeypatch, tmp_path):
    graph = tmp_path / "G11.mat"
    savemat(graph, {"A": sparse.csr_matrix([[0, 1], [1, 0]])})

    def fake_solver(a, k, solver, eps, max_iters, verbose=False, **kwargs):
        return np.array([[1., 1.5], [1.5, 1.]]), {
            "sdp_seconds": 1.0, "solver_status": "optimal_inaccurate"
        }

    monkeypatch.setattr(experiments, "solve_sdp", fake_solver)
    with pytest.raises(ValueError, match="substantially non-PSD"):
        experiments.benchmark(
            [graph], [2], tmp_path / "runs.jsonl",
            restarts=1, cache_dir=tmp_path / "cache"
        )
    assert not list((tmp_path / "cache").glob("*.npz"))
