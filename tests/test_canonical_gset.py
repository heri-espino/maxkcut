"""Canonical G-set text input, legacy input comparison and proven references.

All synthetic test graphs are tiny. No large SDP or batch solve is executed.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

from maxkcut.comparison import compare_rows, write_comparison
from maxkcut.graph_compare import compare_directories
from maxkcut.io import edge_arrays, load_graph
from maxkcut.references import TOROIDAL_TABLE_1, reference_for_graph
from maxkcut.sweep import discover_graphs, inspect_gset, run_sweep


def test_gset_txt_preserves_signed_weights_and_header_n(tmp_path):
    path = tmp_path / "G11.txt"
    path.write_text("5 3\n1 2 4\n2 3 -2\n4 2 3\n", encoding="utf8")
    a = load_graph(path)
    assert sparse.isspmatrix_csr(a)
    assert a.shape == (5, 5)
    assert a[0, 1] == 4.
    assert a[1, 2] == -2.
    assert a[1, 3] == 3.
    assert a[4].nnz == 0  # vertex 5 exists even though isolated
    assert a.nnz == 6
    i, j, weights = edge_arrays(a)
    assert len(i) == 3
    assert sorted(weights) == [-2., 3., 4.]


@pytest.mark.parametrize("rows,pattern", [
    ("3 2\n1 2 1\n", "exactly 2"),
    ("3 2\n1 2 1\n2 1 -1\n", "duplicate"),
    ("3 1\n1 4 1\n", "in 1..3"),
    ("3 1\n1 1 1\n", "self-loops"),
    ("3 1\n1 2 nan\n", "finite"),
    ("3 1\n1.1 2 1\n", "integers"),
    ("hello\n1 2 1\n", "header"),
])
def test_gset_txt_rejects_malformed_or_ambiguous_edges(tmp_path, rows, pattern):
    p = tmp_path / "G1.txt"
    p.write_text(rows, encoding="utf8")
    with pytest.raises(ValueError, match=pattern):
        load_graph(p)


def test_txt_discovery_preferred_over_mat_but_legacy_selectable(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "G81.txt").write_text("3 1\n1 2 1\n")
    (d / "G2.txt").write_text("3 1\n1 3 -1\n")
    savemat(d / "G2.mat", {"A": sparse.csr_matrix([
        [0, 9, 0], [9, 0, 0], [0, 0, 0],
    ])})
    assert [p.name for p in discover_graphs(d)] == ["G2.txt", "G81.txt"]
    assert discover_graphs(d, ["G2"])[0].suffix == ".txt"
    assert discover_graphs(d, ["G2.mat"])[0].suffix == ".mat"
    assert inspect_gset(d)[0]["weight_min"] == -1.
    with pytest.raises(ValueError, match="Repeated"):
        discover_graphs(d, ["G2.txt", "G2.mat"])


def test_paper_17_entries_and_exact_reference_statuses():
    assert len(TOROIDAL_TABLE_1) == 17
    assert reference_for_graph("G11")["reference_status"] == "proven_optimum"
    assert reference_for_graph("G11")["published_cut"] == 564
    assert reference_for_graph("G81")["published_upper_bound"] == 14060
    assert reference_for_graph("G62")["previous_known_cut"] == 4870
    assert reference_for_graph("G62")["published_cut"] == 4872
    assert reference_for_graph("G50")["reference_status"] == "best_known_not_proven"
    assert reference_for_graph("G63") is None


def test_comparison_keeps_proven_and_best_known_distinct():
    data = [
        {"graph": "G11", "n": 800, "status": "complete",
         "best_cut": 546, "mean_cut": 530, "completed_restarts": 5},
        {"graph": "G12", "n": 800, "status": "complete",
         "best_cut": 550, "mean_cut": 540, "completed_restarts": 5},
        {"graph": "G62", "n": 7000, "status": "complete",
         "best_cut": 4872, "mean_cut": 4800, "completed_restarts": 5},
        {"graph": "G50", "n": 3000, "status": "complete",
         "best_cut": 5880, "mean_cut": 5800, "completed_restarts": 5},
        {"graph": "G81", "n": 200, "status": "complete",
         "best_cut": 14060, "mean_cut": 14000, "completed_restarts": 5},
        {"graph": "G63", "n": 7000, "status": "complete",
         "best_cut": 100, "mean_cut": 90, "completed_restarts": 5},
    ]
    entries = compare_rows(data)
    assert entries[0]["reference_status"] == "proven_optimum"
    assert entries[0]["gap_to_paper_cut"] == 18
    assert entries[0]["percent_of_optimum_if_verified"] == pytest.approx(
        100 * 546 / 564
    )
    assert entries[1]["percent_of_optimum_if_verified"] is None
    assert entries[2]["gap_to_paper_cut"] == 0
    assert entries[3]["reference_status"] == "best_known_not_proven"
    assert entries[4]["assessment"] == "GRAPH_SIZE_MISMATCH_check_identity"
    assert entries[4]["percent_of_optimum_if_verified"] is None
    assert entries[5]["assessment"] == "no_reference_in_paper_table_1"


def test_graph_directory_audit_checks_weighted_edge_identity(tmp_path):
    data = tmp_path / "data"
    old = tmp_path / "G-Set"
    data.mkdir()
    old.mkdir()
    (data / "G11.txt").write_text(
        "3 2\n1 2 1\n2 3 -1\n", encoding="utf8"
    )
    savemat(old / "G11.mat", {"A": sparse.csr_matrix([
        [0, 1, 0], [1, 0, -1], [0, -1, 0],
    ])})
    original = compare_directories(data, old, tmp_path / "audit.csv")
    assert original[0]["status"] == "identical_graph"
    assert original[0]["max_absolute_difference"] == 0
    (data / "G11.txt").write_text(
        "3 2\n1 2 1\n2 3 1\n", encoding="utf8"
    )
    changed = compare_directories(data, old, tmp_path / "audit.csv")
    assert changed[0]["status"] == "different_weights_or_edges"
    assert changed[0]["max_absolute_difference"] == 2


def test_txt_baseline_produces_saved_comparison_without_sdp(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "G11.txt").write_text(
        "4 3\n1 2 1\n2 3 -1\n3 4 1\n", encoding="utf8"
    )
    output = tmp_path / "output"
    result = run_sweep(
        data, output, restarts=2, max_sweeps=4, seconds_per_graph=10,
    )
    assert result["completed"] == 1
    assert (output / "comparison.csv").is_file()
    with (output / "comparison.csv").open(newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["assessment"] == "GRAPH_SIZE_MISMATCH_check_identity"
    assert rows[0]["paper_cut"] == "564"
    assert write_comparison(output / "summary.csv").is_file()
