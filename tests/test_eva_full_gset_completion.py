"""Experiment021: immutable cross-experiment 71/71 coverage tests."""
import hashlib
import json

import numpy as np
import pytest

import maxkcut.eva_full_gset_completion_cli as mod


def test_full_coverage_requires_no_missing_factors():
    complete = {
        "graph_count": 71,
        "complete_graphs": 71,
        "no_EVA_factor_count": 0,
        "incomplete_candidate_count": 0,
        "runs_in_all_runs_csv": 71 * 9 * 50,
    }
    assert mod.require_full_coverage(complete) is complete
    for key, bad in (
        ("graph_count", 68),
        ("complete_graphs", 68),
        ("no_EVA_factor_count", 3),
        ("incomplete_candidate_count", 1),
        ("runs_in_all_runs_csv", 30600),
    ):
        corrupted = dict(complete, **{key: bad})
        with pytest.raises(RuntimeError, match=key):
            mod.require_full_coverage(corrupted)


def test_inventory_reuses_old_archives_and_only_adds_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "EXPECTED_GRAPHS", 4)
    monkeypatch.setattr(mod, "MISSING", ("G60", "G61"))
    data = tmp_path / "data"
    data.mkdir()
    old = tmp_path / "experiment020"
    new = tmp_path / "experiment021"
    source = tmp_path / "saved-eva"
    eva = new / "eva-numba"
    old.mkdir()
    records = []
    for graph, status in (
        ("G1", "ready"), ("G60", "no_completed_EVA_factor"),
        ("G61", "no_completed_EVA_factor"),
        ("G63", "reused_019"),
    ):
        file = data / f"{graph}.txt"
        file.write_text(f"test-{graph}", encoding="utf8")
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        records.append({
            "graph": graph, "status": status,
            "factor_source": str(source) if status != "no_completed_EVA_factor" else None,
            "factor_sha256": f"factor-{graph}" if status != "no_completed_EVA_factor" else None,
            "graph_sha256": digest if status != "no_completed_EVA_factor" else None,
            "output_dir": str(old / graph),
        })
    (old / "inventory.json").write_text(
        json.dumps({"graph_count": 4, "records": records}), encoding="utf8"
    )
    # Only G60 has a newly finished pair of factor/case files.
    for subdir, suffix in (("cases", "json"), ("factors", "npz")):
        folder = eva / subdir
        folder.mkdir(parents=True)
        (folder / f"G60.{suffix}").write_bytes(b"test")

    def factor_reader(file, directory):
        return (
            np.zeros((4, 4)), np.zeros((4, 2)), {},
            hashlib.sha256(file.read_bytes()).hexdigest(),
            f"factor-{file.stem}",
        )

    def rows_reader(record, restarts):
        assert restarts == 50
        return [object()] * (9 * restarts)

    manifest = mod.make_inventory(
        old=old, new=new, factor_dir=eva, data_dir=data,
        factor_reader=factor_reader, rows_reader=rows_reader,
    )
    status = {r["graph"]: r["status"] for r in manifest["records"]}
    assert status == {
        "G1": "ready", "G60": "ready",
        "G61": "no_completed_EVA_factor", "G63": "reused_019",
    }
    assert manifest["missing_factors"] == ["G61"]
    assert manifest["verified_ready"] == 2
    assert manifest["reused_verified_candidate"] == 1
    for r in manifest["records"]:
        if r["graph"] in ("G1", "G63"):
            assert r["output_dir"] == str(old / r["graph"])
        else:
            assert r["output_dir"] == str(new / r["graph"])
    assert (old / "inventory.json").read_text(encoding="utf8") == json.dumps(
        {"graph_count": 4, "records": records}
    )


def test_rejects_partially_published_new_factor(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "EXPECTED_GRAPHS", 1)
    monkeypatch.setattr(mod, "MISSING", ("G60",))
    old = tmp_path / "old"
    data = tmp_path / "data"
    eva = tmp_path / "new" / "eva-numba"
    old.mkdir()
    data.mkdir()
    (data / "G60.txt").write_text("test")
    (old / "inventory.json").write_text(json.dumps({
        "records": [{
            "graph": "G60", "status": "no_completed_EVA_factor",
            "output_dir": "unused",
        }],
    }))
    (eva / "cases").mkdir(parents=True)
    (eva / "cases" / "G60.json").write_text("{}")
    with pytest.raises(RuntimeError, match="partially published"):
        mod.make_inventory(
            old=old, new=tmp_path / "new", factor_dir=eva, data_dir=data,
        )
