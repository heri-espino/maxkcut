"""Tests for the orchestration-only Experiment025 Windows parallel driver.

Never run expensive numerical refinements in this test module.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_driver():
    path = Path(__file__).resolve().parents[1] / "scripts" / "experiment025-parallel.py"
    spec = importlib.util.spec_from_file_location("experiment025_parallel_driver", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sequential_chunks_on_one_graph_and_serial_publication(tmp_path, monkeypatch):
    driver = load_driver()
    driver.STOP.clear()
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    observed, published = [], []

    def fake_run(command, **kwargs):
        method = command[command.index("--method") + 1]
        assert command[command.index("--graph") + 1] == "G11"
        assert command[command.index("--max-new-seeds") + 1] == "4"
        target = tmp_path / "pilot" / "G11" / method
        target.mkdir(parents=True, exist_ok=True)
        old = 0
        status = target / "status.json"
        if status.exists():
            old = json.loads(status.read_text())["starts_completed"]
        current = min(16, old + 4)
        status.write_text(json.dumps({
            "starts_completed": current,
            "status": "complete_16_best_seen" if current == 16 else "partially_refined",
            "completed_seed_indices": list(range(current)),
        }))
        observed.append((method, current))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    monkeypatch.setattr(driver, "publish", lambda g, p, m, n:
                        published.append((g, p, m, n)))
    result = driver.run_graph("G11", "pilot", ("gw-rr", "control-paired"), 4)
    assert result == {"graph": "G11", "status": "complete", "methods": 2}
    assert observed == [(m, n) for m in ("gw-rr", "control-paired")
                        for n in (4, 8, 12, 16)]
    assert published == [("G11", "pilot", m, n) for m, n in observed]


def test_publisher_stages_exact_graph_and_never_force_push(tmp_path, monkeypatch):
    driver = load_driver()
    monkeypatch.setattr(driver, "ROOT", tmp_path / "experiment025")
    recorded = []

    def fake_git(*args, **kwargs):
        recorded.append(args)
        return SimpleNamespace(returncode=1 if args[:3] == (
            "diff", "--cached", "--quiet") else 0, stderr="", stdout="")

    monkeypatch.setattr(driver, "git", fake_git)
    driver.publish("G14", "full", "gw-rr", 16)
    assert recorded[0] == ("add", "-A", "--",
                           (tmp_path / "experiment025/full/G14").as_posix())
    assert recorded[-1] == ("push", "origin", "HEAD:experiment025-results")
    assert not any("force" in arg for call in recorded for arg in call)
    assert any("commit" in call for call in recorded)


def test_failed_subprocess_is_published_then_aborts(tmp_path, monkeypatch):
    driver = load_driver()
    driver.STOP.clear()
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    sent = []
    monkeypatch.setattr(driver.subprocess, "run",
                        lambda *a, **kw: SimpleNamespace(returncode=7))
    monkeypatch.setattr(driver, "publish", lambda *a: sent.append(a))
    with pytest.raises(RuntimeError, match="G1/gw-rr failed"):
        driver.run_graph("G1", "pilot", ("gw-rr", "control-paired"), 16)
    assert sent == [("G1", "pilot", "gw-rr", 0)]


def test_driver_pins_one_git_branch_and_avoids_eva_solver():
    source = (Path(__file__).resolve().parents[1] /
              "scripts" / "experiment025-parallel.py").read_text()
    assert "experiment025-results" in source
    assert "scripts/experiment025-windows-io-guard.py" in source
    assert "maxkcut.experiment025_bestcut" not in source
    assert "maxkcut.eva_cli" not in source
    assert "ThreadPoolExecutor" in source
    assert "PUBLISH_LOCK" in source


def test_worker_log_is_force_staged_only_for_active_method(tmp_path, monkeypatch):
    driver = load_driver()
    driver.ROOT = tmp_path / "result"
    logpath = driver.ROOT / "full" / "G21" / "logs" / "control-paired-parallel.log"
    logpath.parent.mkdir(parents=True, exist_ok=True)
    logpath.write_text("PermissionError: [WinError 5] Access is denied")
    calls = []

    def fake_git(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1 if args[:3] == (
            "diff", "--cached", "--quiet") else 0,
            stdout="abc123\\n" if args[:2] == ("rev-parse", "HEAD") else "")

    monkeypatch.setattr(driver, "git", fake_git)
    driver.publish("G21", "full", "control-paired", 2)
    assert ("add", "-f", "--", logpath.as_posix()) in calls
    assert ("add", "-A", "--", (driver.ROOT / "full" / "G21").as_posix()) in calls
    assert calls[-1] == ("push", "origin", "HEAD:experiment025-results")
