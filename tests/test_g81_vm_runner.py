"""G81 VM checkpoint orchestration tests; never launch a real SDP."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import maxkcut.g81_vm as _runner


@pytest.fixture
def runner():
    return _runner


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


def test_native_scs_nan_inf_diagnostics_are_preserved_without_json_failure(
    runner, tmp_path,
):
    target = tmp_path / "checkpoint.json"
    runner.save_json(target, {
        "scs_progress": {
            "primal_residual": float("nan"),
            "duality_gap": float("inf"),
        }
    })
    data = json.loads(target.read_text())
    assert data["scs_progress"]["primal_residual"] == "nan"
    assert data["scs_progress"]["duality_gap"] == "inf"


def test_indirect_g81_uses_separate_archive_and_publication_without_sdp(
    runner, monkeypatch, tmp_path,
):
    import sys

    root = tmp_path
    (root / "data").mkdir()
    (root / "data" / "G81.txt").write_text(
        "20000 40000\n", encoding="utf8",
    )
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "GRAPH", root / "data" / "G81.txt")
    monkeypatch.setattr(
        runner, "BATCH", root / "reports" / "gset" / "g81-vm-128gb",
    )
    monkeypatch.setattr(
        runner, "PUBLIC", root / "results" / "gset" / "g81-128gb-vm",
    )
    commands = []

    class NoSolveProcess:
        pid = 54321

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def no_solve(command, **options):
        commands.append(command)
        return NoSolveProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", no_solve)
    monkeypatch.setattr(sys, "argv", [
        "run_g81_vm.py", "--scs-indirect",
        "--cpu-threads", "8", "--no-push",
        "--checkpoint-seconds", "900",
    ])
    # Batch returns no completed case: runner must report unsuccessful G81.
    assert runner.main() == 2
    assert len(commands) == 1
    assert "--scs-indirect" in commands[0]
    assert "--no-compare" in commands[0]
    assert "--retry-failed" not in commands[0]
    indirect = root / "reports" / "gset" / "g81-vm-128gb-indirect"
    assert commands[0][commands[0].index("--output-dir") + 1] == str(indirect)
    public = root / "results" / "gset" / "g81-128gb-vm-indirect"
    assert (public / "checkpoint.json").exists()
    config = json.loads((public / "run-config.json").read_text(encoding="utf8"))
    assert config["scs_linear_solver_backend"] == "indirect_conjugate_gradient"
    assert "--scs-indirect" in config["run_command"]
    assert not (root / "reports" / "gset" / "g81-vm-128gb").exists()
    assert not (root / "results" / "gset" / "g81-128gb-vm").exists()
