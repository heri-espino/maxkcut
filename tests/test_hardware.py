"""No heavy SDP/GPU calls: CPU thread control, cache identity and CLI tests."""
from __future__ import annotations

import json
import sys

import pytest

from maxkcut.cache import cache_identity, cache_path
from maxkcut.cli import main
from maxkcut.hardware import blas_limit, hardware_info


def test_blas_thread_context_is_reentrant_and_optional():
    with blas_limit(None):
        pass
    with blas_limit(2):
        with blas_limit(1):
            pass
    with pytest.raises(ValueError, match="positive"):
        blas_limit(0)


def test_cpu_threads_and_cudss_have_distinct_cache_identity(tmp_path):
    path = tmp_path / "G11.mat"
    path.write_bytes(b"sample matrix bytes; no need for a solver")
    common = dict(k=2, solver="SCS", eps=1e-3,
                  max_iters=3000, scs_indirect=False)
    original = cache_identity(path, **common)
    eight_threads = cache_identity(path, **common, cpu_threads=8)
    sixteen_threads = cache_identity(path, **common, cpu_threads=16)
    gpu = cache_identity(path, **common, scs_cudss=True)
    assert original != eight_threads != sixteen_threads
    assert gpu != original
    assert len({
        str(cache_path(path, tmp_path, identity))
        for identity in (original, eight_threads, sixteen_threads, gpu)
    }) == 4


def test_hardware_info_does_not_require_nvidia_gpu(monkeypatch):
    from maxkcut import hardware
    monkeypatch.setattr(hardware.shutil, "which", lambda binary: None)
    report = hardware_info()
    assert isinstance(report["logical_cpus"], (int, type(None)))
    assert isinstance(report["blas_pools"], list)
    assert report["nvidia_gpus"] == []
    assert "nvidia-smi" in report["gpu_note"]


def test_cli_hardware_info_has_no_sdp_calls(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["maxkcut", "--hardware-info"])
    main()
    report = json.loads(capsys.readouterr().out)
    assert "logical_cpus" in report
    assert "blas_pools" in report


def test_cli_rejects_incompatible_gpu_and_indirect_options(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        sys, "argv", ["maxkcut", "--scs-cudss", "--scs-indirect"]
    )
    with pytest.raises(SystemExit):
        main()
    assert "--scs-cudss requires" in capsys.readouterr().err


def test_cli_rejects_zero_threads(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["maxkcut", "--cpu-threads", "0"])
    with pytest.raises(SystemExit):
        main()
    assert "--cpu-threads must be positive" in capsys.readouterr().err
