"""Compare canonical G-set .txt graph weights to legacy MATLAB graph inputs.

Graph names alone do not establish input identity. This module checks the
entire normalized undirected weighted adjacency, not file bytes or just n/m.
"""
from __future__ import annotations

import csv
from pathlib import Path
from uuid import uuid4

from .io import load_graph
from .sweep import discover_graphs


def compare_directories(canonical_dir: Path, legacy_dir: Path,
                        report_path: Path) -> list[dict]:
    """Write one row per canonical graph; never solve an optimization problem."""
    canonical_files = discover_graphs(canonical_dir)
    legacy_files = {
        item.stem: item for item in discover_graphs(legacy_dir)
    }
    records = []
    for source in canonical_files:
        row = {
            "graph": source.stem,
            "canonical_file": str(source),
            "legacy_file": None,
            "n_canonical": None,
            "n_legacy": None,
            "same_weighted_graph": False,
            "different_entries": None,
            "max_absolute_difference": None,
            "status": None,
            "error": None,
        }
        other = legacy_files.get(source.stem)
        if other is None:
            row["status"] = "legacy_missing"
            records.append(row)
            continue
        row["legacy_file"] = str(other)
        try:
            a = load_graph(source)
            b = load_graph(other)
            row["n_canonical"] = a.shape[0]
            row["n_legacy"] = b.shape[0]
            if a.shape != b.shape:
                row["status"] = "different_dimensions"
            else:
                difference = (a - b).tocsr()
                difference.eliminate_zeros()
                row["different_entries"] = int(difference.nnz)
                row["max_absolute_difference"] = (
                    float(max(abs(difference.data))) if difference.nnz else 0.
                )
                row["same_weighted_graph"] = difference.nnz == 0
                row["status"] = (
                    "identical_graph" if difference.nnz == 0
                    else "different_weights_or_edges"
                )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        records.append(row)

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "graph", "canonical_file", "legacy_file", "n_canonical",
        "n_legacy", "same_weighted_graph", "different_entries",
        "max_absolute_difference", "status", "error",
    ]
    temporary = report_path.with_name(
        report_path.name + "." + uuid4().hex + ".tmp"
    )
    try:
        with temporary.open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(report_path)
    finally:
        temporary.unlink(missing_ok=True)
    return records
