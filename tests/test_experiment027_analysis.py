"""Small deterministic archived-data tests for read-only Experiment027 analysis."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "experiment027_analyze", ROOT / "scripts" / "experiment027_analyze.py"
)
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)

from maxkcut.experiment023_methods import write_case, sha
from maxkcut.experiment025_journal import IterationJournal
from maxkcut.io import edge_arrays
from maxkcut.objective import cut_weight


def _graph():
    a = np.array([[0, 4, -1, 2],
                  [4, 0, 3, -2],
                  [-1, 3, 0, 5],
                  [2, -2, 5, 0]], dtype=float)
    return sparse.csr_matrix(a)


def _fixture(tmp_path):
    graph = _graph()
    edges = edge_arrays(graph)
    labels = np.array([0, 1, 0, 1], dtype=np.int8)
    score = float(cut_weight(edges, labels))
    folder = tmp_path / "full" / "G1"
    method = "angular-margin-greedy"
    key = "fingerprint_fixture"
    sha_data = "graph", "factor", "directions"
    progress = folder / method / "progress" / "0000"
    journal = IterationJournal(progress, edges, graph.shape[0], key)
    journal.observe(dict(stage="GW", iteration=0, current_cut=score), labels)
    info = dict(index=0, fingerprint=key, graph_sha256=sha_data[0],
                factor_sha256=sha_data[1],
                shared_direction_sha256=sha_data[2],
                best_seen_cut=score, raw_cut=score, progress_events=1,
                seed_seconds=1.0,
                angular_summary=dict(paths_evaluated=1))
    stages = dict(Round=score, Flip=score, KL=score, Final=score)
    write_case(folder, method, 0, labels, stages, info,
               {name: labels.copy() for name in stages})
    expected = dict(fingerprint=key, graph_sha256=sha_data[0],
                    factor_sha256=sha_data[1],
                    direction_sha=sha_data[2],
                    direction_key="shared_direction_sha256")
    return graph, edges, labels, folder, method, expected


def test_independent_archive_rescore_and_tamper_detected(tmp_path):
    graph, edges, labels, folder, method, expected = _fixture(tmp_path)
    row = analyze.audit_seed(folder, method, 0, graph, edges, expected)
    assert row["best"] == row["raw"] == cut_weight(edges, labels)
    assert row["events"] == 1
    assert row["paths"] == 1
    meta = folder / method / "0000.json"
    saved = json.loads(meta.read_text(encoding="utf8"))
    saved["stages"]["Final"] += 1
    meta.write_text(json.dumps(saved), encoding="utf8")
    with pytest.raises(ValueError, match="Final"):
        analyze.audit_seed(folder, method, 0, graph, edges, expected)


def test_detects_corrupted_strict_best_snapshot(tmp_path):
    graph, edges, _, folder, method, expected = _fixture(tmp_path)
    best = folder / method / "progress" / "0000" / "best_000001.npz"
    best.write_bytes(b"bad")
    with pytest.raises(ValueError, match="strict-best snapshot"):
        analyze.audit_seed(folder, method, 0, graph, edges, expected)


def test_normal_trajectory_uses_original_signed_cut(tmp_path):
    graph = _graph()
    edges = edge_arrays(graph)
    y = np.array([[1.0, 0.0], [0.0, 1.0],
                  [-1.0, 0.0], [0.0, -1.0]])
    r = np.array([1.0, 0.0])
    score = float(cut_weight(edges, (y @ r > 0).astype(np.int8)))
    folder = tmp_path / "full" / "G1"
    path = folder / "angular-gain-greedy" / "progress" / "0000"
    path.mkdir(parents=True)
    file = path / "accepted_normals.npz"
    np.savez_compressed(file, normals=r[np.newaxis, :],
                        cuts=np.array([score]), iterations=np.array([0]))
    meta = dict(accepted_normals_sha256=sha(file), raw_cut=score)
    assert analyze.audit_normals(folder, "angular-gain-greedy", 0, y, edges, meta) == 1
    meta["raw_cut"] = score + 1.0
    with pytest.raises(ValueError, match="round"):
        analyze.audit_normals(folder, "angular-gain-greedy", 0, y, edges, meta)


def test_graph_clustered_summaries_and_paired_seeds():
    rows = []
    for g in ("G1", "G2"):
        for method, score in (("gw-rr", 10), ("hyperplane-sweep", 11),
                              ("angular-margin-greedy", 12),
                              ("angular-gain-greedy", 13),
                              ("angular-pair-greedy", 11)):
            for seed in range(16):
                rows.append(dict(graph=g, method=method, seed=seed,
                                 best=float(score), raw=float(score - 1),
                                 seconds=1.0, paths=2))
    bygraph, leaderboard, paired = analyze.summarize(rows, ["G1", "G2"], bootstraps=100)
    assert len(bygraph) == 6
    assert len(paired) == 2 * 3 * 16 * 2
    result = next(x for x in leaderboard if x["method"] == "angular-gain-greedy")
    assert result["wins_vs_hyperplane"] == 2
    assert result["mean_relative_gw_pct"] == pytest.approx(30)
    assert result["bootstrap95_low"] == result["bootstrap95_high"] == 30
