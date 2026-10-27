"""G77 solver-only preflight must never trigger huge CPU matrix factorization."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from maxkcut import precision_study


def test_solver_only_does_not_round_or_compute_cpu_psd_spectrum(monkeypatch, tmp_path):
    graph_file = tmp_path / "G77.txt"
    graph_file.write_text("2 1\n1 2 1\n", encoding="utf8")
    a = sparse.csr_matrix([[0., 1.], [1., 0.]])
    monkeypatch.setattr(precision_study, "load_graph", lambda p: a)

    # This mock preserves the actual backend's metadata interface but
    # excludes CUDA startup from ordinary CPU pytest.
    from maxkcut import gpu_full_sdp
    seen = {}

    def fake_gpu_solver(adjacency, **kwargs):
        seen.update(kwargs)
        return np.array([[1., -1.], [-1., 1.]]), {
            "solver_status": "admm_iteration_limit",
            "sdp_seconds": 1.234, "sdp_objective_primal_uncertified": 1.,
            "rho": 1., "rho_final": 0.5,
            "rho_update_count": 1,
            "converged": False,
            "residuals": {"primal_residual": 0.2, "dual_residual": 0.3},
        }

    monkeypatch.setattr(gpu_full_sdp, "full_sdp_admm_torch", fake_gpu_solver)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU PSD spectral decomposition or rounding invoked")

    monkeypatch.setattr(precision_study, "_round", forbidden)
    monkeypatch.setattr(precision_study.np.linalg, "eigvalsh", forbidden)

    args = SimpleNamespace(
        backend="gpu-admm32-adaptive", rho=1., admm_iters=1,
        admm_abs=1e-5, admm_rel=1e-4, log_every=1,
        rho_update_interval=25, rho_balance_mu=10.,
        rho_scale=2., rho_min=0.0625, rho_max=16.,
        restarts=1, seed=2026, skip_rounding=True,
    )
    result = precision_study._solver_worker(args, graph_file)
    assert result["status"] == "computed"
    assert result["rounding_validation"] == "intentionally_skipped"
    assert result["skip_rounding"] is True
    assert result["rounding"] == {}
    assert result["trials"] == []
    assert result["min_eigenvalue"] is None
    assert result["continuous_primal_uncertified"] == 1.
    assert result["rho_final"] == 0.5
    assert seen["device"] == "cuda"
    assert seen["adaptive_rho"] is True


def test_preflight_flag_survives_subprocess_argument_contract():
    # Omission of --skip-rounding would invoke massive CPU factorization.
    import inspect
    source = inspect.getsource(precision_study.main)
    assert '*(["--skip-rounding"] if args.skip_rounding else [])' in source
    assert '"skip_rounding": args.skip_rounding' in source
