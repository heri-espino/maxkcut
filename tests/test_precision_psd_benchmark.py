"""Small deterministic precision/PSD comparison tests. No large G-set solve."""
from __future__ import annotations

import csv
import json
import numpy as np
import pytest
from scipy import sparse

from maxkcut.precision_study import atomic_json, _summary, _publish


def test_two_vertex_full_convex_sdp_has_optimum_one_cpu_torch():
    torch = pytest.importorskip("torch")
    from maxkcut.gpu_full_sdp import full_sdp_admm_torch
    graph = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    for dtype in ("float32", "float64"):
        matrix, info = full_sdp_admm_torch(
            graph, device="cpu", precision=dtype,
            max_iters=100, eps_abs=1e-6, eps_rel=1e-6,
            log_every=1,
        )
        assert matrix.shape == (2, 2)
        assert np.max(np.abs(np.diag(matrix) - 1)) <= 1e-6
        assert np.min(np.linalg.eigvalsh(matrix)) > -2e-5
        # For the 1-edge graph L=[[1,-1],[-1,1]], max Tr(LX)/4 = 1.
        objective = (matrix[0, 0] + matrix[1, 1]
                     - 2 * matrix[0, 1]) / 4
        assert objective == pytest.approx(1, abs=2e-3)
        assert info["optimization_type"] == "full_convex_psd_admm"
        assert info["device"] == "cpu"
        assert info["iterations"] <= 100


def test_psd_projection_float32_vs64_on_cpu():
    pytest.importorskip("torch")
    from maxkcut.gpu_full_sdp import psd_projection_eigh
    raw = np.array([[1, 2, -1],
                    [2, -1, 0],
                    [-1, 0, 3]], dtype=np.float64)
    mat = (raw + raw.T) / 2
    exact_eig, q = np.linalg.eigh(mat)
    reference = (q * exact_eig.clip(min=0)[None, :]) @ q.T
    for precision, limit in (("float32", 1e-5), ("float64", 1e-10)):
        projection, meta = psd_projection_eigh(
            mat, device="cpu", precision=precision,
        )
        assert np.linalg.norm(projection-reference) < limit
        assert min(np.linalg.eigvalsh(projection)) > -limit
        assert meta["kernel_seconds"] > 0
        assert meta["end_to_end_seconds"] >= meta["kernel_seconds"]


def test_input_validation_rejects_invalid_precision_and_graph():
    pytest.importorskip("torch")
    from maxkcut.gpu_full_sdp import full_sdp_admm_torch
    valid = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    with pytest.raises(ValueError, match="precision"):
        full_sdp_admm_torch(valid, device="cpu", precision="float16")
    nonsymmetric = sparse.csr_matrix(np.array([[0., 1.], [0., 0.]]))
    with pytest.raises(ValueError, match="symmetric"):
        full_sdp_admm_torch(nonsymmetric, device="cpu")
    with pytest.raises(ValueError, match="Invalid ADMM"):
        full_sdp_admm_torch(valid, device="cpu", max_iters=0)


def test_publication_includes_failure_and_all_rounding_trials(tmp_path):
    raw = tmp_path / "reports"
    dest = tmp_path / "results"
    raw.mkdir()
    (raw / "cpu-scs64.log").write_text("SCS example solver info\n")
    completed = {
        "backend": "cpu-scs64", "status": "computed",
        "type": "full_convex_sdp", "solver_status": "optimal",
        "solve_seconds": 3.5,
        "rounding": {
            "FJ": {"best": 5., "mean": 4., "std": 0.},
            "K-Means2N": {"best": 5., "mean": 5., "std": 0.},
            "K-MeansNM": {"best": 6., "mean": 6., "std": 0.},
        },
        "trials": [
            {"method": "FJ", "restart": 0, "cut_weight": 5.},
            {"method": "K-Means2N", "restart": 0, "cut_weight": 5.},
            {"method": "K-MeansNM", "restart": 0, "cut_weight": 6.},
        ],
    }
    failed = {
        "backend": "cpu-sdpa64", "type": "full_convex_sdp",
        "status": "failed", "solver_status": "failed",
        "error": "MemoryExhausted",
    }
    _publish(raw, dest, "G11", [completed, failed])
    with (dest / "comparison.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["cut_best"] == "6.0"
    assert rows[1]["status"] == "failed"
    assert (dest / "cases" / "cpu-scs64.json").is_file()
    with (dest / "rounding-trials.csv").open(newline="") as file:
        assert len(list(csv.DictReader(file))) == 3
    assert (dest / "logs" / "cpu-scs64.txt").is_file()
    assert "SDPA" in (dest / "README.md").read_text()
    assert "not a patch to SCS" in (dest / "README.md").read_text()


def test_atomic_json_writes_no_nan(tmp_path):
    path = tmp_path / "case.json"
    atomic_json(path, {"a": np.float64(1.25), "b": float("nan")})
    assert json.loads(path.read_text()) == {"a": 1.25, "b": "nan"}


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="PyTorch optional",
)
def test_gpu_smoke_when_cuda_available():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("No NVIDIA CUDA device in test environment")
    from maxkcut.gpu_full_sdp import full_sdp_admm_torch
    small = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    matrix, meta = full_sdp_admm_torch(
        small, device="cuda", precision="float32",
        max_iters=20, log_every=1,
    )
    assert meta["device"] == "cuda"
    assert np.isfinite(matrix).all()
