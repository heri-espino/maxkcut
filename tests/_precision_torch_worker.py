"""Windows-safe subprocess tests: import torch BEFORE scientific libraries.

Run by test_precision_psd_benchmark.py in separate native processes
rather than importing PyTorch into pytest's long-lived DLL environment.
"""
from __future__ import annotations

import sys

# Do not move this below numpy, scipy or maxkcut imports on Windows!
import torch

import numpy as np
from scipy import sparse

from maxkcut.gpu_full_sdp import full_sdp_admm_torch, psd_projection_eigh


def two_vertex() -> None:
    adjacency = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    for precision in ("float32", "float64"):
        matrix, info = full_sdp_admm_torch(
            adjacency, device="cpu", precision=precision,
            max_iters=100, eps_abs=1e-6, eps_rel=1e-6,
            log_every=1,
        )
        assert matrix.shape == (2, 2)
        assert np.max(np.abs(np.diag(matrix) - 1)) <= 1e-6
        assert np.linalg.eigvalsh(matrix).min() > -2e-5
        value = (matrix[0, 0] + matrix[1, 1] - 2 * matrix[0, 1]) / 4
        assert abs(value - 1) <= 2e-3, (precision, value)
        assert info["optimization_type"] == "full_convex_psd_admm"
        assert info["device"] == "cpu"
        assert info["iterations"] <= 100


def projector() -> None:
    raw = np.array([
        [1, 2, -1], [2, -1, 0], [-1, 0, 3],
    ], dtype=np.float64)
    matrix = (raw + raw.T) / 2
    values, vecs = np.linalg.eigh(matrix)
    expected = (vecs * values.clip(min=0)[None, :]) @ vecs.T
    for precision, limit in (("float32", 1e-5), ("float64", 1e-10)):
        projected, metrics = psd_projection_eigh(
            matrix, device="cpu", precision=precision,
        )
        assert np.linalg.norm(projected - expected) < limit
        assert np.linalg.eigvalsh(projected).min() > -limit
        assert metrics["kernel_seconds"] >= 0
        assert metrics["end_to_end_seconds"] >= metrics["kernel_seconds"]


def validation() -> None:
    valid = sparse.csr_matrix([[0., 1.], [1., 0.]])
    asymmetric = sparse.csr_matrix([[0., 1.], [0., 0.]])

    def check_bad(f, keyword):
        try:
            f()
        except ValueError as error:
            assert keyword in str(error), str(error)
        else:
            raise AssertionError(f"Expected ValueError containing {keyword}")

    check_bad(lambda: full_sdp_admm_torch(
        valid, device="cpu", precision="float16",
    ), "precision")
    check_bad(lambda: full_sdp_admm_torch(
        asymmetric, device="cpu",
    ), "symmetric")
    check_bad(lambda: full_sdp_admm_torch(
        valid, device="cpu", max_iters=0,
    ), "Invalid ADMM")


def adaptive_tiny() -> None:
    """Exercise actual scaled-dual rho changes on a solved convex 1-edge SDP."""
    a = sparse.csr_matrix([[0., 1.], [1., 0.]])
    for precision in ("float32", "float64"):
        matrix, meta = full_sdp_admm_torch(
            a, device="cpu", precision=precision,
            rho=1., adaptive_rho=True,
            rho_update_interval=2, rho_balance_mu=2.,
            rho_scale=2., rho_min=0.125, rho_max=8.,
            max_iters=100, eps_abs=1e-6, eps_rel=1e-6,
            log_every=1,
        )
        assert meta["adaptive_rho"] is True
        assert meta["rho_update_count"] >= 1, meta
        assert len(meta["rho_events"]) == meta["rho_update_count"]
        for event in meta["rho_events"]:
            assert 0.125 <= event["new_rho"] <= 8.
            assert abs(
                event["previous_rho"] /
                event["new_rho"] - event["scaled_dual_multiplier"]
            ) <= 1e-15
        assert meta["rho_final"] > 0
        assert abs(matrix[0, 0]-1) < 1e-5
        assert abs(matrix[1, 1]-1) < 1e-5
        assert np.linalg.eigvalsh(matrix).min() > -2e-5
        objective = (matrix[0, 0] + matrix[1, 1]-2*matrix[0, 1])/4
        assert abs(objective-1) < 2e-3, (precision, objective)


def fixed_rho_legacy() -> None:
    """Do not silently change old fixed-rho results or clipping behavior."""
    a = sparse.csr_matrix([[0., 1.], [1., 0.]])
    for rho in (0.01, 1., 100.):
        matrix, meta = full_sdp_admm_torch(
            a, device="cpu", precision="float64",
            rho=rho, adaptive_rho=False,
            max_iters=20, eps_abs=1e-4, eps_rel=1e-3,
            log_every=1,
        )
        assert meta["rho"] == rho
        assert meta["rho_final"] == rho
        assert meta["rho_update_count"] == 0
        assert not meta["rho_events"]
        assert np.isfinite(matrix).all()


def gpu_smoke() -> None:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device", flush=True)
        raise SystemExit(77)
    graph = sparse.csr_matrix([[0., 1.], [1., 0.]])
    matrix, metrics = full_sdp_admm_torch(
        graph, device="cuda", precision="float32",
        max_iters=20, log_every=1,
    )
    assert metrics["device"] == "cuda"
    assert np.isfinite(matrix).all()


if __name__ == "__main__":
    actions = {
        "two_vertex": two_vertex,
        "projector": projector,
        "validation": validation,
        "gpu_smoke": gpu_smoke,
        "adaptive_tiny": adaptive_tiny,
        "fixed_rho_legacy": fixed_rho_legacy,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        raise SystemExit("Expected exactly one test scenario")
    actions[sys.argv[1]]()
