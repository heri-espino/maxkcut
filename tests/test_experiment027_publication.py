"""Verify safe, reproducible Experiment027 raw-result publication."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "experiment027_publish", ROOT / "scripts" / "experiment027_publish.py"
)
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


def _source(tmp_path, graph="G1"):
    base = tmp_path / "raw" / "full" / graph
    (base / "angular-gain-greedy" / "progress" / "0000").mkdir(parents=True)
    (base / "gw16_common.npz").write_bytes(b"dummy\n")
    (base / "gw16_common.json").write_text('{"starts":16}\n')
    folder = base / "angular-gain-greedy" / "progress" / "0000"
    (folder / "events.jsonl").write_text('{"stage":"AngularCandidate","score":7}\n')
    (folder / "best_000001.npz").write_bytes(b"fake partition for packaging")
    return base


def _complete_seed_tree(tmp_path, graph="G1"):
    base = tmp_path / "raw" / "full" / graph
    (base).mkdir(parents=True)
    for file in ("gw16_common.npz", "gw16_common.json"):
        (base / file).write_bytes(b"fake immutable shared directions")
    for method in pub.METHODS:
        path = base / method
        path.mkdir()
        (path / "status.json").write_text(
            json.dumps({"status": "complete_16", "starts_completed": 16})
        )
        for i in range(16):
            (path / f"{i:04d}.json").write_text("{}")
            (path / f"{i:04d}.npz").write_bytes(b"small")
            journal = path / "progress" / f"{i:04d}"
            journal.mkdir(parents=True)
            (journal / "events.jsonl").write_text('{"step": 1}\n')
    return base


def test_immutable_lfs_package_is_content_addressed(tmp_path, monkeypatch):
    source = _source(tmp_path)
    outroot = tmp_path / "published"
    monkeypatch.setattr(pub, "PROJECT", tmp_path)
    archive, manifest, metadata = pub.archived_graph(source, outroot / "archives", "G1")
    assert archive.exists() and manifest.exists()
    assert archive.name.startswith("G1-")
    assert metadata["member_count"] == 4
    assert metadata["zip_sha256"] == pub.digest(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert "full/G1/angular-gain-greedy/progress/0000/events.jsonl" in z.namelist()
    same_archive, same_manifest, same_meta = pub.archived_graph(
        source, outroot / "archives", "G1")
    assert same_archive == archive
    assert same_manifest == manifest
    assert same_meta == metadata
    (source / "angular-gain-greedy" / "progress" / "0000" / "events.jsonl").write_text(
        '{"step": 99}\n'
    )
    next_archive, next_manifest, _ = pub.archived_graph(
        source, outroot / "archives", "G1")
    assert next_archive != archive
    assert archive.is_file() and manifest.is_file()
    assert next_archive.is_file() and next_manifest.is_file()


def test_archive_rejects_temporary_raw_or_symlink(tmp_path, monkeypatch):
    source = _source(tmp_path)
    monkeypatch.setattr(pub, "PROJECT", tmp_path)
    (source / "partial.tmp").write_text("being written")
    with pytest.raises(RuntimeError, match="in-flight"):
        pub.archived_graph(source, tmp_path / "published" / "archives", "G1")
    (source / "partial.tmp").unlink()
    (source / "symlink").symlink_to(source / "gw16_common.npz")
    with pytest.raises(ValueError, match="Symlink"):
        pub.archived_graph(source, tmp_path / "published" / "archives", "G1")


def test_missing_full_seed_archive_is_not_publishable(tmp_path):
    root = tmp_path / "raw"
    _complete_seed_tree(tmp_path)
    pub.validate_graph(root, "G1")
    (root / "full" / "G1" / "angular-pair-greedy" / "0003.npz").unlink()
    with pytest.raises(FileNotFoundError, match="0003.npz"):
        pub.validate_graph(root, "G1")


def test_audit_matches_exact_requested_graphs(tmp_path):
    root = tmp_path / "raw"
    audit = root / "analysis"
    audit.mkdir(parents=True)
    report = {
        "status": "independently_verified_archives", "graphs": 1,
        "completed_new_seeds": 48, "audited_method_seed_archives": 80,
    }
    (audit / "audited_comparison.json").write_text(json.dumps(report))
    for name in pub.AUDIT_FILES:
        if name != "audited_comparison.json" and name != "audited_graph_comparisons.csv":
            (audit / name).write_text("a,b\n")
    (audit / "audited_graph_comparisons.csv").write_text(
        "graph,method\n" + "".join(f"G1,{m}\n" for m in pub.METHODS)
    )
    assert pub.validate_audit(root, ["G1"]) == audit
    with pytest.raises(ValueError, match="different graph IDs"):
        pub.validate_audit(root, ["G11"])


def test_dry_run_never_invokes_git_or_creates_archives(tmp_path, monkeypatch, capsys):
    _complete_seed_tree(tmp_path)
    monkeypatch.setattr(pub, "PROJECT", tmp_path)
    monkeypatch.setattr(pub, "SOURCE", tmp_path / "raw")
    monkeypatch.setattr(pub, "PUBLISH", tmp_path / "published")
    monkeypatch.setattr(pub, "command", lambda *a, **k: pytest.fail("Git/audit called"))
    assert pub.main(["--graphs", "G1", "--dry-run"]) == 0
    assert not (tmp_path / "published").exists()
    assert '"new_method_seed_archives": 48' in capsys.readouterr().out
