"""G81 VM checkpoint orchestration tests; never launch a real SDP."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner():
    source = Path(__file__).resolve().parents[1] / "scripts" / "run_g81_vm.py"
    spec = importlib.util.spec_from_file_location("g81_vm_runner_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_log_excerpt_is_bounded_and_private_paths_redacted(
    runner, monkeypatch, tmp_path,
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    file = tmp_path / "solver.log"
    file.write_text(
        "solver path: " + str(tmp_path) + "\n" +
        "  250| 1.23e-04  2.34e-04  3.45e-04\n",
        encoding="utf8",
    )
    text = runner.log_excerpt(file, 10000)
    assert "<REPO>" in text
    assert str(tmp_path) not in text
    assert "250|" in text


def test_started_checkpoint_tracks_progress_but_cannot_claim_resume(
    runner, monkeypatch, tmp_path,
):
    batch = tmp_path / "reports" / "gset" / "g81-vm-128gb"
    public = tmp_path / "results" / "gset" / "g81-128gb-vm"
    (batch / "cases").mkdir(parents=True)
    log = batch / "runs" / "G81" / "attempt-001" / "solver.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        " iter | pri res | dua res | gap |\n"
        "  250| 1.23e-04  2.34e-04  3.45e-04 -1.0e+00\n",
        encoding="utf8",
    )
    (batch / "cases" / "G81.json").write_text(json.dumps({
        "graph": "G81", "status": "started", "attempts": 1,
        "log_path": str(log),
    }), encoding="utf8")
    (batch / "batch-stdout.txt").write_text("G81 full SDP started\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "BATCH", batch)
    monkeypatch.setattr(runner, "PUBLIC", public)
    monkeypatch.setattr(runner, "linux_memory", lambda: {"MemAvailable_gib": 100})
    monkeypatch.setattr(runner, "linux_processes", lambda pid: [{"pid": pid}])
    pushed = []
    monkeypatch.setattr(runner, "push_snapshot", lambda: pushed.append(True))
    fake_process = SimpleNamespace(poll=lambda: None, pid=1234)
    runner.checkpoint(
        fake_process, state="running", exit_code=None, do_push=True,
    )
    saved = json.loads((public / "checkpoint.json").read_text())
    assert saved["case_status"] == "started"
    assert saved["scs_latest_iteration"]["iteration"] == 250
    assert saved["cannot_resume_partial_solver_iterations"] is True
    assert saved["true_scs_iteration_checkpoint"] is False
    assert saved["best_cut_if_complete"] is None
    assert pushed == [True]
    assert not (public / "compact-archives").exists()


def test_only_public_g81_folder_staged_when_pushing(runner, monkeypatch, tmp_path):
    public = tmp_path / "results" / "gset" / "g81-128gb-vm"
    public.mkdir(parents=True)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PUBLIC", public)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:3] == ["git", "branch", "--show-current"]:
            return subprocess.CompletedProcess(command, 0, stdout="main\n")
        if command[:4] == ["git", "diff", "--cached", "--quiet"]:
            return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.push_snapshot()
    assert ["git", "add", "-A", "--", "results/gset/g81-128gb-vm"] in commands
    assert ["git", "push", "origin", "HEAD:main"] in commands
    assert all("--force" not in cmd for cmd in commands)
    assert not any("reports" in " ".join(cmd) for cmd in commands)


def test_case_log_path_cannot_escape_trusted_batch(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "BATCH", tmp_path / "batch")
    with pytest.raises(ValueError, match="escapes"):
        runner.latest_solver_log({
            "graph": "G81", "log_path": str(tmp_path / "outside.log")
        })
