"""Archive atomic replace tests, including simulated Windows file locks."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from maxkcut import atomic_fs
from maxkcut.archive import _atomic_json


def test_atomic_replace_retry_on_transient_windows_access_denied(
    monkeypatch, tmp_path
):
    source = tmp_path / "source.tmp"
    target = tmp_path / "target.json"
    source.write_text("new")
    target.write_text("old")
    original = os.replace
    attempts = []
    sleeps = []

    def flaky(src, dst):
        attempts.append(len(attempts))
        if len(attempts) <= 2:
            raise PermissionError("simulated transient WinError 5")
        return original(src, dst)

    # Patch only our module-local hook, never the global os module.
    # Other pytest/Windows file operations must not affect this counter.
    monkeypatch.setattr(atomic_fs, "_replace_file", flaky)
    monkeypatch.setattr(atomic_fs, "_retryable", lambda exc: True)
    monkeypatch.setattr(atomic_fs, "_retry_sleep", lambda sec: sleeps.append(sec))
    atomic_fs.replace_atomic(source, target)
    assert attempts == [0, 1, 2]
    assert sleeps == pytest.approx([0.05, 0.10])
    assert os.replace is original  # Other modules were not monkeypatched.
    assert not source.exists()
    assert target.read_text() == "new"


def test_monkeypatch_does_not_intercept_unrelated_renames(monkeypatch, tmp_path):
    outside_source = tmp_path / "outside.tmp"
    outside_dest = tmp_path / "outside.json"
    outside_source.write_text("outside")
    seen = []

    def patch_only_atomic(src, dst):
        seen.append((src, dst))
        raise PermissionError("unrelated failure")

    monkeypatch.setattr(atomic_fs, "_replace_file", patch_only_atomic)
    # This unrelated file operation must call the original system primitive.
    os.replace(outside_source, outside_dest)
    assert outside_dest.read_text() == "outside"
    assert seen == []


def test_atomic_replace_permanent_failure_does_not_truncate_destination(
    monkeypatch, tmp_path
):
    source = tmp_path / "source.tmp"
    target = tmp_path / "target.json"
    source.write_text("new")
    target.write_text("original")

    def deny(src, dst):
        raise PermissionError("permanent lock")

    monkeypatch.setattr(atomic_fs, "_replace_file", deny)
    monkeypatch.setattr(atomic_fs, "_retryable", lambda exc: True)
    monkeypatch.setattr(atomic_fs, "_retry_sleep", lambda sec: None)
    with pytest.raises(PermissionError):
        atomic_fs.replace_atomic(source, target, attempts=3)
    assert target.read_text() == "original"
    assert source.read_text() == "new"


def test_non_transient_error_is_not_retried(monkeypatch, tmp_path):
    count = []

    def fail(src, dst):
        count.append(1)
        raise PermissionError("not a retryable lock")

    monkeypatch.setattr(atomic_fs, "_replace_file", fail)
    monkeypatch.setattr(atomic_fs, "_retryable", lambda exc: False)
    with pytest.raises(PermissionError):
        atomic_fs.replace_atomic(tmp_path / "src", tmp_path / "dest")
    assert len(count) == 1


def test_archive_manifest_still_valid_after_atomic_write(tmp_path):
    dest = tmp_path / "manifest.json"
    _atomic_json(dest, {"graph": "G1", "status": "started"})
    _atomic_json(dest, {"graph": "G1", "status": "complete"})
    import json
    assert json.loads(dest.read_text()) == {
        "graph": "G1", "status": "complete"
    }
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_torch_uses_fresh_worker_process():
    from maxkcut.precision_study import _backend_worker_prefix

    for name in ("gpu-admm32", "gpu-admm64",
                 "gpu-eigh32", "gpu-eigh64", "cpu-eigh64"):
        cmd = _backend_worker_prefix(name)
        assert cmd[2] == "-c"
        assert cmd[3].startswith("import torch;")
    for name in ("cpu-scs64", "cpu-sdpa64"):
        assert _backend_worker_prefix(name)[2:] == [
            "-m", "maxkcut.precision_study"
        ]
