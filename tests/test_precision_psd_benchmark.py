"""Small deterministic precision/PSD comparison tests. No large G-set solve."""
from __future__ import annotations

import csv
import json
import numpy as np
import pytest
from scipy import sparse

from maxkcut.precision_study import atomic_json, _summary, _publish


@pytest.mark.parametrize(
    "scenario", [
        "two_vertex", "projector", "validation", "gpu_smoke",
        "adaptive_tiny", "fixed_rho_legacy", "gpu_gw_cut",
    ]
)
def test_torch_backend_in_fresh_process(scenario):
    """PyTorch must not enter pytest's existing SDPA/MKL Windows DLL process.

    The subprocess imports torch first, then SciPy/CVXPY/maxkcut. This
    tests the actual CUDA/ADMM math, rather than skipping native failures.
    """
    from importlib.util import find_spec
    from pathlib import Path
    import subprocess
    import sys

    if find_spec("torch") is None:
        pytest.skip("Optional torch is not installed")
    worker = Path(__file__).with_name("_precision_torch_worker.py")
    result = subprocess.run(
        [sys.executable, "-u", str(worker), scenario],
        capture_output=True, text=True, timeout=120, check=False,
    )
    if result.returncode == 77:
        pytest.skip(result.stdout.strip() or "No NVIDIA CUDA GPU")
    assert result.returncode == 0, (
        f"Fresh-interpreter torch {scenario} failed with "
        f"exit code {result.returncode}:\\n"
        f"stdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}"
    )


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
    assert "patch to SCS" in (dest / "README.md").read_text()


def test_atomic_json_writes_no_nan(tmp_path):
    path = tmp_path / "case.json"
    atomic_json(path, {"a": np.float64(1.25), "b": float("nan")})
    assert json.loads(path.read_text()) == {"a": 1.25, "b": "nan"}


def test_push_publication_only_stages_study_folder(monkeypatch, tmp_path):
    from maxkcut import precision_study as study
    dest = tmp_path / "results" / "gset" / "precision-psd-benchmark"
    dest.mkdir(parents=True)
    (dest / "comparison.csv").write_text("backend,status\n")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["git", "rev-parse", "--show-toplevel"]:
            return __import__("subprocess").CompletedProcess(
                cmd, 0, stdout=str(tmp_path) + "\n",
            )
        if cmd[:3] == ["git", "branch", "--show-current"]:
            return __import__("subprocess").CompletedProcess(
                cmd, 0, stdout="main\n",
            )
        if cmd[:4] == ["git", "diff", "--cached", "--quiet"]:
            return __import__("subprocess").CompletedProcess(cmd, 1)
        return __import__("subprocess").CompletedProcess(cmd, 0)

    monkeypatch.setattr(study.subprocess, "run", fake_run)
    study._push_publication(dest)
    assert ["git", "add", "-A", "--",
            "results/gset/precision-psd-benchmark"] in calls
    assert ["git", "push", "origin", "HEAD:main"] in calls
    assert not any("--force" in cmd for cmd in calls)
    assert not any("pull" in cmd for cmd in calls)


def test_gpu_rounding_flags_reject_incompatible_modes():
    """No accidental CPU backend or silent solver-only no-cut run."""
    from maxkcut.precision_study import main

    for flags in (
        ["--backends", "cpu-scs64", "--gpu-gw-rounding"],
        ["--backends", "gpu-admm32", "--skip-rounding",
         "--gpu-gw-rounding"],
    ):
        with pytest.raises(SystemExit) as exc:
            main(["--graph", "G11", *flags])
        assert exc.value.code == 2


def test_gpu_labels_archive_does_not_break_score_csv(tmp_path):
    """Per-trial packed assignments remain in JSON, not CSV columns."""
    import base64

    record = {
        "backend": "gpu-admm32-adaptive",
        "status": "computed", "type": "full_convex_sdp",
        "solver_status": "admm_iteration_limit",
        "rounding": {"FJ": {"best": 3., "mean": 3., "std": None}},
        "trials": [{
            "method": "FJ", "restart": 0, "cut_weight": 3.,
            "labels_packed_b64": base64.b64encode(bytes([160])).decode(),
        }],
    }
    dest = tmp_path / "results"
    _publish(tmp_path, dest, "G77", [record])
    saved = json.loads((dest / "cases" / "gpu-admm32-adaptive.json").read_text())
    assert saved["trials"][0]["labels_packed_b64"] == "oA=="
    with (dest / "rounding-trials.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["cut_weight"] == "3.0"
    assert len(rows) == 1
