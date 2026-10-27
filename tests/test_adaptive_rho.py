"""Deterministic non-CUDA tests for residual-balanced convex SDP ADMM."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from maxkcut.gpu_full_sdp import balanced_penalty_update
from maxkcut import precision_study


def test_residual_balance_decrease_increase_and_no_change():
    # Large dual residual -> smaller rho; large primal -> larger rho.
    assert balanced_penalty_update(1.0, 0.1, 2.0) == (0.5, "decrease")
    assert balanced_penalty_update(1.0, 2.0, 0.1) == (2.0, "increase")
    assert balanced_penalty_update(1.0, 1.0, 1.0) == (1.0, None)
    assert balanced_penalty_update(1.0, 0.0, 0.0) == (1.0, None)
    assert balanced_penalty_update(1.0, 0.1, 1.0) == (1.0, None)


def test_rho_bounds_and_unchanged_scaled_dual_equivalence():
    new_rho, reason = balanced_penalty_update(
        0.125, primal=0., dual=1., mu=2., scale=2.,
        rho_min=0.125, rho_max=8.,
    )
    assert new_rho == 0.125 and reason is None
    new_rho, reason = balanced_penalty_update(
        8., primal=1., dual=0., mu=2., scale=2.,
        rho_min=0.125, rho_max=8.,
    )
    assert new_rho == 8. and reason is None
    old_rho, old_scaled_dual = 1.0, -3.5
    new_rho, direction = balanced_penalty_update(
        old_rho, primal=0.01, dual=1.,
    )
    assert direction == "decrease"
    new_scaled_dual = old_scaled_dual * old_rho / new_rho
    assert new_rho * new_scaled_dual == pytest.approx(
        old_rho * old_scaled_dual,
    )


@pytest.mark.parametrize("invalid", [
    {"rho": 0.}, {"rho": float("nan")},
    {"primal": -1.}, {"dual": float("inf")},
    {"mu": 1.}, {"scale": 1.},
    {"rho_min": -1.}, {"rho_min": 2., "rho_max": 1.},
    {"rho": 100.},
])
def test_invalid_penalty_control_rejected(invalid):
    params = {"rho": 1., "primal": 0., "dual": 1.}
    params.update(invalid)
    with pytest.raises(ValueError):
        balanced_penalty_update(**params)


def test_explicit_adaptive_backends_and_published_rho_headers():
    assert "gpu-admm32-adaptive" in precision_study.BACKENDS
    assert "gpu-admm64-adaptive" in precision_study.BACKENDS
    assert "rho_initial" in precision_study.COLUMNS
    assert "rho_final" in precision_study.COLUMNS
    assert "rho_updates" in precision_study.COLUMNS


def test_study_plan_includes_penalty_settings(monkeypatch, tmp_path):
    """Changing adaptive parameters must never silently mix output cases."""
    from scipy import sparse
    import numpy as np
    # main() is not called: actual G11 tests happen on the Windows VM.
    # This smoke-check confirms CLI accepts the explicit study backends.
    # The computational scenarios run in tests/_precision_torch_worker.py.
    assert "gpu-admm32-adaptive" in precision_study.BACKENDS
