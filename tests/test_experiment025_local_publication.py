"""No-compute regression tests for local-only Experiment025 execution and preview."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_mode_preserves_checkpoint_and_never_calls_git_publisher(tmp_path, monkeypatch):
    driver = load_script("experiment025-parallel.py")
    driver.STOP.clear()
    monkeypatch.setattr(driver, "ROOT", tmp_path)

    def fake_run(command, **kwargs):
        target = tmp_path / "full" / "G1" / "gw-rr"
        target.mkdir(parents=True, exist_ok=True)
        (target / "status.json").write_text(json.dumps(dict(
            starts_completed=16,
            status="complete_16_best_seen",
            completed_seed_indices=list(range(16))
        )))
        return SimpleNamespace(returncode=0)

    def forbidden_publish(*args):
        raise AssertionError("Local mode must not call Git publisher")

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    monkeypatch.setattr(driver, "publish", forbidden_publish)
    assert driver.run_graph("G1", "full", ("gw-rr",), 16, "local") == {
        "graph": "G1", "status": "complete", "methods": 1
    }
    assert (tmp_path / "full" / "G1" / "gw-rr" / "status.json").is_file()


def test_preview_blocks_incomplete_comparisons_and_keeps_all_methods(tmp_path):
    preview = load_script("experiment025-summary-preview.py")
    folder = tmp_path / "analysis"
    folder.mkdir()
    data = [
        {"graph": "G1", "method": "gw-rr",
         "status": "complete_16_best_seen", "refined_starts": "16",
         "best_seen_cut": "100"},
        {"graph": "G1", "method": "control-paired",
         "status": "partially_refined", "refined_starts": "3",
         "best_seen_cut": ""},
    ]
    with (folder / "full_best_final_by_graph.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        wr = csv.DictWriter(f, fieldnames=list(data[0]))
        wr.writeheader()
        wr.writerows(data)
    (folder / "full_coverage.json").write_text(json.dumps({
        "planned_cells": 2, "completed_cells": 1,
        "planned_independent_full_refinements": 32,
        "all_complete": False,
    }))
    result = Path(preview.generate_preview(tmp_path))
    body = result.read_text()
    assert "1/2" in body and "19/32" in body
    assert "PROVISIONAL" in body
    assert "No graph has complete" in body
    assert "control-paired" in body


def test_powershell_local_mode_does_not_modify_algorithm_source():
    root = Path(__file__).resolve().parents[1]
    ps = (root / "scripts" / "experiment025-vm.ps1").read_text()
    parallel = (root / "scripts" / "experiment025-parallel.py").read_text()
    assert "[ValidateSet('git','local')][string]$PublishMode" in ps
    assert "Assert-Experiment025LocalChanges" in ps
    assert "--publish-mode" in ps and "--publish-mode" in parallel
    assert "if ($PublishMode -eq 'local')" in ps
    assert "publish_mode == \"git\"" in parallel


def test_local_parallel_driver_invokes_guarded_launcher(tmp_path, monkeypatch):
    """Bug regression: local-only scheduler must not bypass Windows I/O retries."""
    driver = load_script("experiment025-parallel.py")
    driver.STOP.clear()
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    invoked = []

    def fake_run(command, **kwargs):
        invoked.append(command)
        target = tmp_path / "full" / "G56" / "mst-l2-longest"
        target.mkdir(parents=True, exist_ok=True)
        (target / "status.json").write_text(json.dumps(dict(
            starts_completed=16, status="complete_16_best_seen",
            completed_seed_indices=list(range(16))
        )))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    monkeypatch.setattr(
        driver, "publish",
        lambda *args: (_ for _ in ()).throw(AssertionError("Unexpected Git publication"))
    )
    result = driver.run_graph("G56", "full", ("mst-l2-longest",), 16, "local")
    assert result["status"] == "complete"
    assert len(invoked) == 1
    assert invoked[0][1] == "scripts/experiment025-windows-io-guard.py"
    assert "-m" not in invoked[0]


def test_scoped_guard_retries_transient_windows_access_denied():
    """No disk writes or scientific computations; simulate WinError 5."""
    guard = load_script("experiment025-windows-io-guard.py")
    src = r"results\gset\experiment025-gw16-bestcut-quality\full\G56\mst-l2-longest\progress\0005\best.json.tmp"
    dst = src.removesuffix(".tmp")
    attempts, delays = [], []

    def simulate_replace(source, target):
        attempts.append((source, target))
        if len(attempts) < 3:
            error = PermissionError("simulated Windows file lock")
            error.winerror = 5
            raise error

    guard.retry_checkpoint_replace(
        src, dst, replace_func=simulate_replace,
        sleep_func=delays.append, attempts=4,
    )
    assert len(attempts) == 3
    assert len(delays) == 2
    assert all(x == (src, dst) for x in attempts)


def test_guard_does_not_retry_unrelated_file_or_non_windows_permission_error():
    guard = load_script("experiment025-windows-io-guard.py")
    source = "outside-archive.json.tmp"
    target = "outside-archive.json"
    attempted = []

    def always_denied(*args):
        attempted.append(args)
        error = PermissionError("simulated Windows file lock")
        error.winerror = 5
        raise error

    import pytest
    with pytest.raises(PermissionError):
        guard.retry_checkpoint_replace(
            source, target, replace_func=always_denied,
            sleep_func=lambda seconds: None,
        )
    assert len(attempted) == 1

    source = r"results\gset\experiment025-gw16-bestcut-quality\full\G56\progress\0005\best.json.tmp"
    target = source.removesuffix(".tmp")
    attempted.clear()

    def not_a_windows_error(*args):
        attempted.append(args)
        raise PermissionError("unknown permission error")

    with pytest.raises(PermissionError):
        guard.retry_checkpoint_replace(
            source, target, replace_func=not_a_windows_error,
            sleep_func=lambda seconds: None,
        )
    assert len(attempted) == 1
