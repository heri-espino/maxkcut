"""Experiment027 regression: factor-source selection must match Experiment025 exactly."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).resolve().parents[1]/"scripts"/"experiment027_from_archives.py"
spec = importlib.util.spec_from_file_location("experiment027_from_archives", MODULE)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def records(tmp_path, monkeypatch):
    original = tmp_path/"original"
    baseline = original/"results/gset/experiment025-gw16-bestcut-quality"
    data = original/"data"
    data.mkdir(parents=True)
    graph = b"3 2\n1 2 1\n2 3 1\n"
    (data/"G14.txt").write_bytes(graph)
    good = b"simulated EVA archive bytes"
    expected_factor = sha(good)
    status = baseline/"full/G14/gw-rr/status.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps(dict(status="complete_16_best_seen", starts_completed=16,
                                      graph_sha256=sha(graph), factor_sha256=expected_factor)))
    monkeypatch.setattr(subject, "FACTOR_RELATIVE_DIRS", ("older", "fallback"))
    for name, content in (("older", b"different factor"), ("fallback", good)):
        folder = original/name
        (folder/"cases").mkdir(parents=True)
        (folder/"factors").mkdir(parents=True)
        (folder/"cases/G14.json").write_text(json.dumps(dict(
            status="converged", graph_sha256=sha(graph))))
        (folder/"factors/G14.npz").write_bytes(content)
    return SimpleNamespace(original=original,baseline=baseline,data=data,
                           factor=expected_factor, status=status)


def test_resolves_by_historical_factor_sha_not_first_available(tmp_path,monkeypatch):
    r=records(tmp_path,monkeypatch)
    folder,digest=subject.select_pinned_factor("G14",r.original,r.baseline,r.data)
    assert folder==r.original/"fallback"
    assert digest==r.factor


def test_missing_factor_refuses_to_substitute(tmp_path,monkeypatch):
    r=records(tmp_path,monkeypatch)
    (r.original/"fallback/factors/G14.npz").unlink()
    with pytest.raises(FileNotFoundError,match="Do not rerun EVA"):
        subject.select_pinned_factor("G14",r.original,r.baseline,r.data)


def test_changed_graph_refuses_factor(tmp_path,monkeypatch):
    r=records(tmp_path,monkeypatch)
    (r.data/"G14.txt").write_bytes(b"changed")
    with pytest.raises(ValueError,match="graph SHA256"):
        subject.select_pinned_factor("G14",r.original,r.baseline,r.data)


def test_incomplete_baseline_refuses_paired_comparison(tmp_path,monkeypatch):
    r=records(tmp_path,monkeypatch)
    x=json.loads(r.status.read_text())
    x["starts_completed"]=1
    r.status.write_text(json.dumps(x))
    with pytest.raises(ValueError,match="not complete"):
        subject.select_pinned_factor("G14",r.original,r.baseline,r.data)


def test_preflight_prevents_partial_launch(tmp_path,monkeypatch):
    r=records(tmp_path,monkeypatch)
    with pytest.raises(FileNotFoundError,match="baseline unavailable"):
        subject.select_pinned_factor("G11",r.original,r.baseline,r.data)
