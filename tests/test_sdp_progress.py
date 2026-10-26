"""Read-only progress monitoring tests; no real SDP computations."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from maxkcut.sdp_progress import batch_status, latest_scs_iteration


def test_parse_scs_progress_and_ignore_irrelevant_output(tmp_path):
    log = tmp_path / "solver.log"
    log.write_text(
        "------------------------------------------------------------------\n"
        " iter | pri res | dua res |   gap   |   obj   |  scale  | time (s)\n"
        "------------------------------------------------------------------\n"
        "     0| 1.00e+00  2.00e+00  3.00e+00  0.00e+00  1.00e-01  1.00e-01\n"
        "   250| 1.20e-03  3.50e-04  7.00e-04 -4.33e+02  2.30e-01  1.20e+02\n"
        "Solver still solving\n", encoding="utf8",
    )
    status = latest_scs_iteration(log)
    assert status["iteration"] == 250
    assert status["primal_residual"] == pytest.approx(1.2e-3)
    assert status["dual_residual"] == pytest.approx(3.5e-4)
    assert status["duality_gap"] == pytest.approx(7e-4)


def test_status_reads_case_records_and_progress_without_running_solver(tmp_path):
    folder = tmp_path / "full-sdp"
    cases = folder / "cases"
    cases.mkdir(parents=True)
    log = folder / "runs" / "G56" / "attempt-001" / "solver.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "  1750| 6.00e-04  8.00e-04  1.00e-02 -1.00e+02\n",
        encoding="utf8",
    )
    (cases / "G55.json").write_text(json.dumps({
        "graph": "G55", "status": "complete", "best_cut": 9512,
    }), encoding="utf8")
    (cases / "G56.json").write_text(json.dumps({
        "graph": "G56", "status": "started",
        "log_path": str(log), "settings": {"max_iters": 3000},
    }), encoding="utf8")
    (cases / "G81.json").write_text(json.dumps({
        "graph": "G81", "status": "skipped_size",
    }), encoding="utf8")
    report = batch_status(folder)
    assert report["graphs_with_case_records"] == 3
    assert report["statuses"] == {
        "complete": 1, "skipped_size": 1, "started": 1,
    }
    in_progress = report["started_cases_not_guaranteed_live"]
    assert len(in_progress) == 1
    assert in_progress[0]["graph"] == "G56"
    assert in_progress[0]["scs_progress"]["iteration"] == 1750
    assert in_progress[0]["scs_progress"]["iteration_cap"] == 3000
    assert in_progress[0]["log_age_seconds"] >= 0


def test_missing_verbose_output_does_not_invent_progress(tmp_path):
    root = tmp_path / "batch"
    folder = root / "cases"
    folder.mkdir(parents=True)
    (folder / "G56.json").write_text(json.dumps({
        "graph": "G56", "status": "started",
        "log_path": str(root / "solver.log"),
    }), encoding="utf8")
    report = batch_status(root)
    active = report["started_cases_not_guaranteed_live"][0]
    assert active["scs_progress"] is None
    assert "without --solver-progress" in active["note"]


def test_status_cli_is_read_only_even_with_unsafe_graph_size(
    monkeypatch, tmp_path, capsys,
):
    import maxkcut.sdp_batch_cli as cli

    root = tmp_path / "runs"
    (root / "cases").mkdir(parents=True)
    (root / "cases" / "G81.json").write_text(json.dumps({
        "graph": "G81", "status": "complete",
    }), encoding="utf8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Status must not invoke any numerical solver")

    monkeypatch.setattr(cli, "run_full_sdp_batch", forbidden)
    monkeypatch.setattr(cli, "inventory", forbidden)
    monkeypatch.setattr(sys, "argv", [
        "maxkcut-sdp-batch", "--status", "--output-dir", str(root),
    ])
    cli.main()
    report = json.loads(capsys.readouterr().out)
    assert report["completed_case_records"] == 1
    assert not (root / "plan.json").exists()
