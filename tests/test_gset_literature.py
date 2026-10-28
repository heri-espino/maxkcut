"""Offline checks for source-labeled G-set benchmarks and EVA comparison."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from maxkcut.gset_literature import compare, load_references, write_comparison


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data/gset_literature_reference_71.csv"
EVA = (ROOT / "results/gset/literature-comparison/"
       "experiment004-eva-precision-source.csv")
SAVED = (ROOT / "results/gset/literature-comparison/"
         "experiment004-eva-precision-comparison.csv")


def test_all_71_benchmark_values_have_source_and_proof_status():
    ref = load_references(REF)
    expected = {f"G{i}" for i in range(1, 68)}
    expected |= {"G70", "G72", "G77", "G81"}
    assert set(ref) == expected
    proven = {name for name, row in ref.items()
              if row["reference_status"] == "published_proven_optimum"}
    assert proven == {
        "G11", "G48", "G49", "G57", "G62",
        "G65", "G72", "G77", "G81",
    }
    assert int(ref["G63"]["reference_cut"]) == 27047
    assert ref["G63"]["source_key"] == "Khan_Shukla_2025_G63"
    assert int(ref["G62"]["reference_cut"]) == 4872
    assert int(ref["G77"]["published_upper_bound"]) == 9940
    assert int(ref["G81"]["published_upper_bound"]) == 14060
    assert all(int(r["reference_cut"]) >= int(r["openising_snapshot_cut"])
               for r in ref.values())


def test_saved_experiment004_comparison_reconstructs_all_graphs(tmp_path):
    rows = compare(EVA, REF)
    assert len(rows) == 71
    by_graph = {r["graph"]: r for r in rows}
    assert len(by_graph) == 71
    assert sum(r["eva_status"] == "converged" for r in rows) == 61
    assert sum(r["eva_status"] == "iteration_limit" for r in rows) == 10
    assert all(r["graph_identity_evidence"] == "size_and_edge_count_match_only"
               for r in rows)
    assert by_graph["G63"]["eva_best_cut"] == 26060
    assert by_graph["G63"]["reference_cut"] == 27047
    assert by_graph["G77"]["gap_to_reference"] == 924
    assert by_graph["G81"]["gap_to_reference"] == 1316

    out = write_comparison(EVA, REF, tmp_path / "comparison.csv")
    with out.open(newline="", encoding="utf8") as stream:
        written = list(csv.DictReader(stream))
    with SAVED.open(newline="", encoding="utf8") as stream:
        committed = list(csv.DictReader(stream))
    assert len(written) == len(committed)
    for generated, original in zip(written, committed):
        assert generated["graph"] == original["graph"]
        assert float(generated["eva_best_cut"]) == float(original["eva_best_cut"])
        assert float(generated["gap_to_reference"]) == float(original["gap_to_reference"])
        assert generated["reference_status"] == original["reference_status"]
        assert generated["source_url"] == original["source_url"]


def test_reference_validation_rejects_false_optimum(tmp_path):
    path = tmp_path / "false.csv"
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "graph", "reference_vertices", "reference_edges",
            "reference_cut", "reference_status", "published_upper_bound",
            "source_key", "source_url",
        ))
        writer.writeheader()
        writer.writerow({
            "graph": "G1", "reference_vertices": "800",
            "reference_edges": "19176", "reference_cut": "11624",
            "reference_status": "published_proven_optimum",
            "published_upper_bound": "",
            "source_key": "incorrect", "source_url": "https://example.org",
        })
    with pytest.raises(ValueError, match="proof requires"):
        load_references(path)


def test_missing_graph_does_not_fake_scores(tmp_path):
    eva = tmp_path / "missing.csv"
    eva.write_text(
        "graph,vertices,edges,status,eva_sweeps,eva_FJ_best,"
        "eva_KMeans2N_best,eva_KMeansNM_best,graph_sha256\n",
        encoding="utf8",
    )
    rows = compare(eva, REF)
    assert len(rows) == 71
    assert all(r["eva_status"] == "not_run" for r in rows)
    assert all(r["eva_best_cut"] == "" for r in rows)
    assert all(r["gap_to_reference"] == "" for r in rows)
