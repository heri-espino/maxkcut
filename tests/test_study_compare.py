"""Small synthetic archive-verified full SDP versus low-rank comparison.

No CVXPY solve, GPU allocation, CUDA import, or network access required.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from maxkcut.archive import new_archive, new_run_id, save_method, update_manifest
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight
from maxkcut.study_compare import (
    UNSOLVED_THESIS_GRAPHS, compare_saved_runs, full_sdp_preflight,
    write_study_comparison,
)


def _fake_run(root: Path, filename: str, graph: str, methods: dict,
              *, run_id: str, n: int = 3, tamper: bool = False,
              snapshot: bool = True, solver: str = "SCS") -> Path:
    output = root / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    content = []
    for method, scores in methods.items():
        for rep, cut in enumerate(scores):
            item = {
                "graph": graph, "n": n, "k": 2, "method": method,
                "restart": rep, "cut_weight": cut,
                "artifact_run_id": run_id,
            }
            if filename.startswith("sdp"):
                item.update({
                    "solver_status": "optimal_inaccurate",
                    "solver": solver, "sdp_original_solve_seconds": 11.,
                    "cache_hit": False, "sdp_objective_primal": 10.,
                })
            else:
                item.update({
                    "rank": 2,
                    "original_optimizer_seconds": 2.,
                    "factor_cache_hit": False, "device": "cuda",
                })
            content.append(item)
    output.write_text(
        "".join(json.dumps(entry) + "\n" for entry in content),
        encoding="utf8",
    )
    if snapshot:
        archive = root / "artifacts" / run_id / f"{graph}-k2"
        archive.mkdir(parents=True, exist_ok=True)
        np.savez(
            archive / "graph.npz",
            n=np.asarray(n), i=np.asarray([0, 1]),
            j=np.asarray([1, 2]),
            weights=np.asarray([1., 1. if tamper else -1.]),
        )
        (archive / "manifest.json").write_text(
            json.dumps({
                "settings": {
                    "epochs": 100, "optimizer_starts": 1, "rank": 2,
                }
            }),
            encoding="utf8",
        )
    return output


def test_full_sdp_and_cuda_comparison_checks_exact_graph_identity(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "G11.txt").write_text(
        "3 2\n1 2 1\n2 3 -1\n", encoding="utf8",
    )
    sdp = _fake_run(tmp_path, "sdp-g11.jsonl", "G11", {
        "FJ": [1, 2],
        "K-Means2N": [3, 4],
        "K-MeansNM": [4, 5],
    }, run_id="sdp-run")
    cuda = _fake_run(tmp_path, "cuda-g11.jsonl", "G11", {
        "GW_torchlowrank": [0, 1],
        "K-Means2N_torchlowrank": [2, 3],
        "K-MeansNM_torchlowrank": [2, 4],
    }, run_id="cuda-run")
    rows = compare_saved_runs([sdp], cuda, data)
    assert len(rows) == 1
    item = rows[0]
    assert item["graph_identity"] == "VERIFIED_CANONICAL"
    assert item["sdp_status"] == "optimal_inaccurate"
    assert item["sdp_best_method"] == "K-MeansNM"
    assert item["sdp_best_cut"] == 5.
    assert item["sdp_GW_best"] == 2.
    assert item["cuda_best_cut"] == 4.
    assert item["sdp_minus_cuda_cut"] == 1.
    assert item["sdp_vs_cuda_optimizer_time_ratio_uncached"] == 5.5
    assert item["reference_status"] == "proven_optimum"
    assert item["cuda_rank"] == 2
    assert item["cuda_epochs"] == 100
    assert item["cuda_optimizer_starts"] == 1
    # A synthetic 3-vertex G11 must never receive paper optimum ratios.
    assert item["paper_assessment"] == "UNVERIFIED_GRAPH_IDENTITY_OR_SIZE"
    assert item["cuda_percent_of_proven_optimum"] is None
    assert item["sdp_percent_of_proven_optimum"] is None
    output = tmp_path / "comparison.csv"
    saved = write_study_comparison([sdp], cuda, data, output)
    assert saved == rows
    with output.open(newline="", encoding="utf8") as handle:
        raw = list(csv.DictReader(handle))
    assert len(raw) == 1
    assert raw[0]["sdp_best_cut"] == "5.0"


def test_graph_mismatch_or_missing_archive_disables_cut_delta(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "G12.txt").write_text(
        "3 2\n1 2 1\n2 3 -1\n", encoding="utf8",
    )
    full = _fake_run(tmp_path, "sdp.jsonl", "G12", {
        "FJ": [2], "K-Means2N": [3], "K-MeansNM": [4],
    }, run_id="sdp-run", tamper=True)
    cuda = _fake_run(tmp_path, "cuda.jsonl", "G12", {
        "GW_torchlowrank": [1],
        "K-Means2N_torchlowrank": [2],
        "K-MeansNM_torchlowrank": [3],
    }, run_id="cuda-run")
    one = compare_saved_runs([full], cuda, data)[0]
    assert one["graph_identity"] == "GRAPH_MISMATCH"
    assert one["sdp_minus_cuda_cut"] is None
    assert one["paper_assessment"] == "UNVERIFIED_GRAPH_IDENTITY_OR_SIZE"

    (tmp_path / "artifacts" / "cuda-run" / "G12-k2" / "graph.npz").unlink()
    two = compare_saved_runs([full], cuda, data)[0]
    assert two["graph_identity"] == "MISSING_ARCHIVE"
    assert two["sdp_minus_cuda_cut"] is None


def test_missing_full_sdp_is_explicit_not_run(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "G81.txt").write_text(
        "3 2\n1 2 1\n2 3 -1\n", encoding="utf8",
    )
    cuda = _fake_run(tmp_path, "cuda.jsonl", "G81", {
        "GW_torchlowrank": [0], "K-Means2N_torchlowrank": [1],
        "K-MeansNM_torchlowrank": [2],
    }, run_id="cuda-run")
    only = compare_saved_runs([], cuda, data)[0]
    assert only["sdp_status"] == "NOT_RUN"
    assert only["sdp_best_cut"] is None
    assert only["cuda_best_cut"] == 2.
    assert only["sdp_minus_cuda_cut"] is None


def test_rejects_mixed_runs_and_nonfinite_values(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "G11.txt").write_text(
        "3 2\n1 2 1\n2 3 -1\n", encoding="utf8",
    )
    sdp = _fake_run(tmp_path, "sdp.jsonl", "G11", {
        "FJ": [1], "K-Means2N": [2],
    }, run_id="sdp-run")
    cuda = _fake_run(tmp_path, "cuda.jsonl", "G11", {
        "GW_torchlowrank": [1], "K-Means2N_torchlowrank": [2],
    }, run_id="cuda-run")
    with sdp.open("a", encoding="utf8") as handle:
        handle.write(json.dumps({
            "graph": "G11", "n": 3, "k": 2, "method": "K-MeansNM",
            "cut_weight": 3, "artifact_run_id": "different-run",
        }) + "\n")
    with pytest.raises(ValueError, match="multiple runs"):
        compare_saved_runs([sdp], cuda, data)

    sdp.write_text(json.dumps({
        "graph": "G11", "n": 3, "k": 2, "method": "FJ",
        "cut_weight": float("nan"), "artifact_run_id": "sdp-run",
    }) + "\n")
    with pytest.raises(ValueError, match="non-finite"):
        compare_saved_runs([sdp], cuda, data)


def test_preflight_seven_uses_header_and_not_dense_sdp(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for graph in UNSOLVED_THESIS_GRAPHS:
        (data / f"{graph}.txt").write_text(
            "20000 40000\n", encoding="utf8"
        )
    plan = full_sdp_preflight(data, UNSOLVED_THESIS_GRAPHS)
    assert len(plan) == 7
    assert all(x["n"] == 20000 for x in plan)
    assert all(x["one_dense_float64_gram_gib"] > 2.9 for x in plan)
    assert all("VERY_LARGE" in x["message"] for x in plan)
    assert not (tmp_path / "reports").exists()


def _write_audited_cuda_run(path, graph_path, labels, *, complete=True):
    """Append one fully audited CUDA-run block, with correct archived scores."""
    graph = load_graph(graph_path)
    edges = edge_arrays(graph)
    run = new_archive(
        path.parent / "artifacts", new_run_id(), graph_path,
        graph.shape[0], edges, 2,
        {"rank": 2, "epochs": 100, "optimizer_starts": 1},
        jsonl_output=path,
    )
    samples = []
    for method in (
        "GW_torchlowrank", "K-Means2N_torchlowrank",
        "K-MeansNM_torchlowrank",
    ):
        values = [cut_weight(edges, labels)] * 2
        save_method(run, method, [labels, labels], values, k=2)
        for rep, score in enumerate(values):
            samples.append({
                "graph": graph_path.stem, "n": graph.shape[0], "k": 2,
                "method": method, "restart": rep, "cut_weight": score,
                "artifact_run_id": run["manifest"]["run_id"],
                "device": "cuda", "rank": 2,
            })
    if complete:
        update_manifest(run, status="complete")
    with path.open("a", encoding="utf8") as stream:
        for row in samples:
            stream.write(json.dumps(row) + "\\n")
    return run["manifest"]["run_id"]


def test_append_only_cuda_selects_latest_verified_run_not_maximum_cut(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    graph_path = data / "G11.txt"
    graph_path.write_text("3 2\\n1 2 1\\n2 3 -1\\n", encoding="utf8")
    cuda_path = tmp_path / "all71-cuda-baseline.jsonl"
    # First complete run has best cut 1.
    run_one = _write_audited_cuda_run(
        cuda_path, graph_path, np.array([0, 1, 1]),
    )
    # More recent complete run has lower best cut 0; do NOT cherry-pick run one.
    run_two = _write_audited_cuda_run(
        cuda_path, graph_path, np.array([0, 1, 0]),
    )
    # Most recent (third) run was interrupted and must NOT be selected.
    _write_audited_cuda_run(
        cuda_path, graph_path, np.array([0, 1, 1]), complete=False,
    )
    rows = compare_saved_runs([], cuda_path, data)
    assert len(rows) == 1
    row = rows[0]
    assert run_one != run_two
    assert row["cuda_best_cut"] == 0.
    assert row["cuda_run_id"] == run_two
    assert row["cuda_candidate_runs"] == 3
    assert row["cuda_selection_policy"] == "latest_complete"
    assert row["graph_identity"] == "VERIFIED_CANONICAL"
    assert row["sdp_status"] == "NOT_RUN"


def test_duplicate_cuda_without_any_valid_complete_archive_fails(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    graph = data / "G11.txt"
    graph.write_text("3 2\\n1 2 1\\n2 3 -1\\n", encoding="utf8")
    output = tmp_path / "cuda.jsonl"
    _write_audited_cuda_run(
        output, graph, np.array([0, 1, 1]), complete=False,
    )
    _write_audited_cuda_run(
        output, graph, np.array([0, 1, 0]), complete=False,
    )
    with pytest.raises(ValueError, match="none has a complete verified archive"):
        compare_saved_runs([], output, data)


def test_compare_only_reuses_existing_full_sdp_records_without_solvers(
    tmp_path, monkeypatch
):
    import maxkcut.sdp_batch_cli as cli
    import sys

    data = tmp_path / "data"
    data.mkdir()
    (data / "G11.txt").write_text(
        "3 2\\n1 2 1\\n2 3 -1\\n", encoding="utf8"
    )
    full = _fake_run(tmp_path, "sdp-graph.jsonl", "G11", {
        "FJ": [1], "K-Means2N": [2], "K-MeansNM": [3],
    }, run_id="full")
    cuda = _fake_run(tmp_path, "cuda-graph.jsonl", "G11", {
        "GW_torchlowrank": [1], "K-Means2N_torchlowrank": [2],
        "K-MeansNM_torchlowrank": [2],
    }, run_id="gpu")
    out = tmp_path / "batch"
    (out / "cases").mkdir(parents=True)
    (out / "cases" / "G11.json").write_text(
        json.dumps({"graph": "G11", "status": "complete",
                    "source_jsonl": str(full)}),
        encoding="utf8",
    )
    def no_solver(*args, **kwargs):
        raise AssertionError("compare-only must never run convex SDP")
    monkeypatch.setattr(cli, "run_full_sdp_batch", no_solver)
    monkeypatch.setattr(sys, "argv", [
        "maxkcut-sdp-batch", "--compare-only",
        "--gset-dir", str(data),
        "--output-dir", str(out),
        "--cuda-results", str(cuda),
    ])
    cli.main()
    assert (out / "sdp-vs-cuda.csv").is_file()
    with (out / "sdp-vs-cuda.csv").open(newline="", encoding="utf8") as stream:
        values = list(csv.DictReader(stream))
    assert len(values) == 1
    assert values[0]["sdp_best_cut"] == "3.0"
    assert values[0]["cuda_best_cut"] == "2.0"
