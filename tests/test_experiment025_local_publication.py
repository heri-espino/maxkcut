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
