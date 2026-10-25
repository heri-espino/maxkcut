"""Fast regression tests: optional CUDA backend, never requires a GPU."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

from maxkcut.archive import verify_archive
from maxkcut.io import edge_arrays
from maxkcut.lowrank import (
    lowrank_objective, optimize_torch_lowrank, torch_device_info,
)
from maxkcut.lowrank_cache import (
    factor_cache_path, factor_identity, load_factor, save_factor,
)
from maxkcut.lowrank_experiments import benchmark_lowrank


def test_vector_objective_signed_edges_and_unit_factors():
    edges = (np.array([0, 1]), np.array([1, 2]),
             np.array([3., -2.]))
    v = np.array([[1., 0.], [-1., 0.], [1., 0.]])
    # positive edge is cut (3), negative edge is uncut (0)
    assert lowrank_objective(edges, v) == pytest.approx(3.)


def test_explicit_cuda_refuses_silent_cpu_fallback(monkeypatch):
    import maxkcut.lowrank as module

    class FakeCUDA:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        __version__ = "fake CPU"
        cuda = FakeCUDA()
        version = type("Version", (), {"cuda": None})

    monkeypatch.setattr(module, "_torch", lambda: FakeTorch())
    with pytest.raises(RuntimeError, match="Refusing silent CPU fallback"):
        torch_device_info("cuda")
    info = torch_device_info("cpu")
    assert info["device"] == "cpu"
    assert not info["cuda_available"]
    with pytest.raises(ValueError, match="device"):
        torch_device_info("tpu")


def test_numpy_cache_only_factors_and_never_dense_gram(tmp_path):
    graph = tmp_path / "G81.txt"
    graph.write_text("3 2\n1 2 3\n2 3 -2\n")
    settings = {
        "backend": "torch_lowrank", "device": "cuda", "rank": 2,
        "epochs": 10, "seed": 123, "torch_version": "mock",
    }
    identity = factor_identity(graph, settings)
    path = factor_cache_path(tmp_path / "cache", graph, identity)
    vectors = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    info = {"backend": "torch_lowrank", "optimizer_seconds": 1.2}
    assert load_factor(path, identity, n=3, rank=2) is None
    save_factor(path, identity, vectors, info)
    retrieved, recorded = load_factor(path, identity, n=3, rank=2)
    np.testing.assert_array_equal(retrieved, vectors)
    assert recorded == info
    with np.load(path, allow_pickle=False) as archive:
        assert archive["vectors"].shape == (3, 2)
        assert "X" not in archive.files

    other = factor_identity(graph, {**settings, "epochs": 20})
    assert factor_cache_path(tmp_path / "cache", graph, other) != path
    graph.write_text("3 2\n1 2 3\n2 3 2\n")
    changed = factor_identity(graph, settings)
    assert changed != identity
    assert factor_cache_path(tmp_path / "cache", graph, changed) != path


def test_cpu_torch_optimizer_never_builds_dense_gram():
    pytest.importorskip("torch")
    a = sparse.csr_matrix([
        [0, 2, -1], [2, 0, 1], [-1, 1, 0]
    ], dtype=float)
    factors, info = optimize_torch_lowrank(
        a, rank=2, epochs=30, optimizer_starts=2,
        learning_rate=0.04, seed=42, device="cpu",
        log_every=5,
    )
    assert factors.shape == (3, 2)
    assert np.allclose(np.linalg.norm(factors, axis=1), 1, atol=1e-5)
    assert info["backend"] == "torch_lowrank"
    assert info["device"] == "cpu"
    assert info["sdp_upper_bound_certified"] is False
    assert info["lowrank_vector_objective"] == pytest.approx(
        lowrank_objective(edge_arrays(a), factors), abs=1e-6
    )
    assert len(info["optimizer_history"]) >= 2
    assert info["lowrank_vector_objective"] >= max(
        x["vector_objective"] for x in info["optimizer_history"]
    ) - 1e-5


@pytest.mark.parametrize("kwargs", [
    {"rank": 0}, {"rank": 5}, {"epochs": 0},
    {"optimizer_starts": 0}, {"learning_rate": -1},
    {"log_every": 0}, {"seed": -1},
])
def test_optimizer_invalid_settings_fail_before_torch_import(kwargs):
    graph = sparse.csr_matrix([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        optimize_torch_lowrank(graph, device="cpu", **kwargs)


def test_full_gpu_benchmark_archive_and_reuse_without_real_optimizer(
    monkeypatch, tmp_path
):
    import maxkcut.lowrank_experiments as experiments

    graph_path = tmp_path / "G11.mat"
    a = sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    savemat(graph_path, {"A": a})
    calls = []

    def device_info(device):
        return {
            "torch_version": "mockTorch",
            "torch_cuda_version": "mockCUDA",
            "gpu_name": "Mock GPU" if device == "cuda" else None,
        }

    def optimizer(graph, **params):
        calls.append(params)
        return np.array([
            [1., 0.], [-1., 0.], [1., 0.],
        ], dtype=np.float32), {
            "backend": "torch_lowrank",
            "optimizer_seconds": 3.,
            "torch_version": "mockTorch",
            "gpu_name": "Mock GPU",
            "lowrank_vector_objective": 2.0,
            "optimizer_history": [{"epoch": 0, "vector_objective": 1.}],
        }

    monkeypatch.setattr(experiments, "torch_device_info", device_info)
    monkeypatch.setattr(experiments, "optimize_torch_lowrank", optimizer)
    options = {
        "device": "cuda", "rank": 2, "epochs": 2,
        "optimizer_starts": 1, "restarts": 2, "max_n": 10,
        "factor_cache_dir": tmp_path / "factor-cache",
    }
    first_output = tmp_path / "run1.jsonl"
    second_output = tmp_path / "run2.jsonl"
    benchmark_lowrank([graph_path], first_output, **options)
    benchmark_lowrank([graph_path], second_output, **options)
    assert len(calls) == 1

    rows1 = [json.loads(x) for x in first_output.read_text().splitlines()]
    rows2 = [json.loads(x) for x in second_output.read_text().splitlines()]
    assert len(rows1) == len(rows2) == 6
    assert all(r["is_full_convex_sdp"] is False for r in rows1)
    assert all(r["sdp_upper_bound_certified"] is False for r in rows1)
    assert all(r["factor_cache_hit"] is False for r in rows1)
    assert all(r["factor_cache_hit"] is True for r in rows2)
    assert all(r["optimizer_seconds"] == 0.0 for r in rows2)
    for restart in (0, 1):
        gw = next(x["cut_weight"] for x in rows1
                  if x["method"] == "GW_torchlowrank"
                  and x["restart"] == restart)
        nm = next(x["cut_weight"] for x in rows1
                  if x["method"] == "K-MeansNM_torchlowrank"
                  and x["restart"] == restart)
        assert nm >= gw

    case_dir = list((tmp_path / "artifacts").glob("*/G11-k2"))
    assert len(case_dir) == 2
    for directory in case_dir:
        assert (directory / "lowrank_factors.npz").is_file()
        verified = verify_archive(directory)
        assert verified["status"] == "complete"
        assert len(verified["methods"]) == 3
        assert all(
            x["verified_partitions"] == 2
            for x in verified["methods"].values()
        )


def test_gpu_cli_reports_cuda_failure_before_writing_files(
    monkeypatch, tmp_path
):
    import maxkcut.lowrank_cli as cli
    monkeypatch.setattr(
        sys, "argv", ["maxkcut-gpu", "--device-info", "--device", "cuda"]
    )
    def unavailable(device):
        raise RuntimeError("CUDA was explicitly requested")

    monkeypatch.setattr(cli, "torch_device_info", unavailable)
    with pytest.raises(RuntimeError, match="explicitly requested"):
        cli.main()
    assert not (tmp_path / "reports").exists()
