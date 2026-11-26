"""Experiment026: reference integrity and isolated toy-search resume."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("numba")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiment026_hyperplane_sweep.py"
spec = importlib.util.spec_from_file_location("experiment026_hyperplane", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_literature_reference_table_has_71_and_updated_targets():
    refs, digest = runner.load_references(Path(__file__).resolve().parents[1] /
                                           "references/gset-literature-2026-11.csv")
    assert len(refs) == 71 and len(digest) == 64
    for graph, cut in {
        "G1":11624, "G9":2054, "G28":3298, "G36":7680,
        "G40":2400, "G42":2481, "G44":6650, "G53":3850,
        "G60":14190, "G63":27047, "G81":14060
    }.items():
        assert int(refs[graph]["reference_cut"]) >= cut
    assert refs["G81"]["reference_kind"] == "conditional_proven_optimum"
    assert int(refs["G81"]["published_upper_bound"]) == 14060


def test_resume_without_repeating_or_overwriting_completed_batches(tmp_path, monkeypatch):
    rng = np.random.default_rng(87)
    n, p = 8, 4
    w = rng.integers(-3, 5, size=(n, n)).astype(float)
    w = np.triu(w, k=1)
    w = w + w.T
    np.fill_diagonal(w, 0)
    graph = sparse.csr_matrix(w)
    y = rng.normal(size=(n, p))
    y /= np.linalg.norm(y, axis=1)[:, None]
    edges = len(sparse.triu(graph, k=1).data)
    monkeypatch.setattr(runner, "select_factor", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(runner, "_factor_and_case",
                        lambda *a, **kw: (graph, y, {}, "graphsha", "factorsha"))
    params = dict(output=str(tmp_path / "results"), data_dir="data",
                  factor_dir=None, hours=0.1, seed=2026,
                  directions=3, batches_per_restart=2,
                  refine_every=0, max_batches=1)
    ref = {"n_vertices":str(n), "n_edges":str(edges),
           "reference_cut":"12"}
    one = runner.run_graph("G1", params, ref, "literature-sha")
    assert one["batches_completed"] == 1
    checkpoint = tmp_path / "results" / "G1" / "checkpoint.json"
    before = checkpoint.read_text(encoding="utf-8")
    two = runner.run_graph("G1", params, ref, "literature-sha")
    assert two["batches_completed"] == 2
    assert checkpoint.read_text(encoding="utf-8") != before
    files = sorted((tmp_path / "results" / "G1" / "traces").glob("batch_*.jsonl"))
    assert len(files) == 2
    assert all(len(p.read_text().splitlines()) == 3 for p in files)
    assert two["best_cut"] >= one["best_cut"]
    monkeypatch.setattr(runner, "_factor_and_case",
                        lambda *a, **kw: (graph, y, {}, "different", "factorsha"))
    with pytest.raises(ValueError, match="altered"):
        runner.run_graph("G1", params, ref, "literature-sha")


def test_dry_run_never_reads_eva_or_runs_optimizer(tmp_path, monkeypatch, capsys):
    import shutil
    reference_copy = tmp_path / "refs.csv"
    shutil.copy(Path(__file__).resolve().parents[1] /
                "references/gset-literature-2026-11.csv", reference_copy)
    monkeypatch.setattr(runner, "select_factor",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("EVA accessed")))
    assert runner.main(["--graphs","G1","G42","--references",str(reference_copy),
                        "--output",str(tmp_path / "nothing"),
                        "--dry-run"]) == 0
    assert not (tmp_path / "nothing").exists()
    assert '"G42": 2481' in capsys.readouterr().out
