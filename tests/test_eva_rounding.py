"""Tests: reuse frozen EVA factors for append-only rounding extensions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maxkcut.archive import verify_archive
from maxkcut.eva_cli import run_one
from maxkcut.eva_rounding_cli import (
    _round_one, _write_collection_summary, round_saved_factor,
)
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight
from maxkcut.eva import load_eva_factor


def _completed_small_eva(tmp_path):
    graph_path = tmp_path / "G11.txt"
    graph_path.write_text(
        "5 7\n1 2 2\n1 3 -1\n1 4 1\n2 3 2\n"
        "2 5 1\n3 4 1\n4 5 3\n", encoding="utf8",
    )
    source = tmp_path / "results" / "eva"
    args = SimpleNamespace(
        max_n=20, output_dir=source, rank=None,
        tolerance=1e-6, max_sweeps=20, seed=2026,
        checkpoint_every=2, rounding_seed=2026,
        restarts=2, cpu_threads=1, convex_dir=None,
        push=False, push_branch=None, push_every=0,
    )
    run_one(graph_path, args)
    return graph_path, source, tmp_path / "results" / "extensions"


def test_add_80_style_roundings_without_reoptimizing_or_overwriting(tmp_path):
    path, source, dest = _completed_small_eva(tmp_path)
    first_factor = (source / "factors" / "G11.npz").read_bytes()
    base_archive = json.loads(
        (source / "cases" / "G11.json").read_text()
    )["archive_path"]
    methods = ["FJ_EVA", "K-Means2N_EVA", "K-MeansNM_EVA"]
    report = round_saved_factor(
        path, source=source, dest=dest,
        methods=methods, first=2, restarts=3,
    )
    assert all(report["methods"][m]["count"] == 5 for m in methods)
    assert Path(base_archive).is_dir()
    assert (source / "factors" / "G11.npz").read_bytes() == first_factor
    snapshots = list((dest / "factor-snapshots").glob("*.npz"))
    assert len(snapshots) == 1
    assert snapshots[0].read_bytes() == first_factor
    segments = list((dest / "segments").rglob("r00002-00005.json"))
    assert len(segments) == 3
    for file in segments:
        info = json.loads(file.read_text())
        verified = verify_archive(Path(info["archive_path"]))
        assert verified["status"] == "complete"
        assert verified["methods"][info["method"]]["verified_partitions"] == 3

    # The same command verifies old archives instead of regenerating them.
    again = round_saved_factor(
        path, source=source, dest=dest,
        methods=methods, first=2, restarts=3,
    )
    assert again["methods"] == report["methods"]
    assert len(list((dest / "archives").glob("*"))) == 3

    # A later command extends 5 -> 7, leaving previous 2 and 3 intact.
    more = round_saved_factor(
        path, source=source, dest=dest,
        methods=methods, first=5, restarts=2,
    )
    assert all(more["methods"][m]["count"] == 7 for m in methods)
    assert len(list((dest / "archives").glob("*"))) == 6
    with pytest.raises(ValueError, match="already exist"):
        round_saved_factor(
            path, source=source, dest=dest,
            methods=methods, first=4, restarts=4,
        )


def test_new_flip_algorithms_never_worsen_same_seed_roundings(tmp_path):
    path, source, dest = _completed_small_eva(tmp_path)
    graph = load_graph(path)
    edges = edge_arrays(graph)
    vectors, _ = load_eva_factor(source / "factors" / "G11.npz")
    for restart in range(4):
        for initial, improved in (
            ("FJ_EVA", "FJ-Flip_EVA"),
            ("K-MeansNM_EVA", "K-MeansNM-Flip_EVA"),
        ):
            labels1, _ = _round_one(
                vectors, graph, edges, initial, 11, restart, 2026
            )
            labels2, _ = _round_one(
                vectors, graph, edges, improved, 11, restart, 2026
            )
            assert cut_weight(edges, labels2) >= cut_weight(edges, labels1)

    report = round_saved_factor(
        path, source=source, dest=dest,
        methods=["FJ-Flip_EVA", "K-MeansNM-Flip_EVA"],
        first=0, restarts=4,
    )
    assert report["methods"]["FJ-Flip_EVA"]["count"] == 4
    assert report["methods"]["K-MeansNM-Flip_EVA"]["count"] == 4
    assert report["methods"]["FJ_EVA"]["count"] == 2


def test_extension_summary_and_factor_identity_validation(tmp_path):
    path, source, dest = _completed_small_eva(tmp_path)
    round_saved_factor(
        path, source=source, dest=dest,
        methods=["FJ_EVA"], first=2, restarts=3,
    )
    _write_collection_summary(dest)
    with (dest / "summary.csv").open(newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    row = next(r for r in rows if r["method"] == "FJ_EVA")
    assert row["graph"] == "G11"
    assert row["rounding_count"] == "5"
    with pytest.raises(ValueError, match="different|mismatch"):
        round_saved_factor(
            path, source=source, dest=dest,
            methods=["FJ_EVA"], first=2, restarts=3,
            seed=999,
        )
