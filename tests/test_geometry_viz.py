"""Visualization tests use tiny SAVED artifacts; no numerical SDP/EVA solver."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from maxkcut.archive import new_archive, save_method, update_manifest
from maxkcut.eva import _identity, _prepared_graph
from maxkcut.geometry_viz import (
    archived_roundings, discover_eva_cases, load_convex_embedding,
    load_eva_artifacts, load_sweep_log, pairwise_cosines,
    project_vectors, replay_kmeans, sample_vertices, vertex_metrics,
)
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight


@pytest.fixture
def archive_fixture(tmp_path):
    root = tmp_path
    source = Path("results/gset/experiment002-eva")
    graph_path = root / "data" / "G22.txt"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text(
        "5 7\n1 2 2\n1 3 -1\n1 4 1\n2 3 2\n"
        "2 5 1\n3 4 1\n4 5 3\n", encoding="utf8"
    )
    graph = load_graph(graph_path)
    rng = np.random.default_rng(1234)
    vectors = rng.normal(size=(5, 4))
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    graph_sha = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    factor_path = root / source / "factors" / "G22.npz"
    factor_path.parent.mkdir(parents=True)
    metadata = {"identity": _identity(_prepared_graph(graph), 4, 2026, 0.001),
                "sweeps": 7, "converged": True, "objective": 2.0}
    np.savez_compressed(factor_path, vectors=vectors,
                        metadata=np.array(json.dumps(metadata)))
    edges = edge_arrays(graph)
    labels = [
        np.array([0, 1, 1, 0, 1], dtype=np.uint8),
        np.array([1, 0, 1, 0, 1], dtype=np.uint8),
    ]
    scores = [cut_weight(edges, l) for l in labels]
    archive = new_archive(root / source / "archives", "synthetic0001",
                          graph_path, len(vectors), edges, 2,
                          {"algorithm": "test"}, jsonl_output=root / "tests.jsonl")
    save_method(archive, "K-MeansNM_EVA", labels, scores, k=2)
    update_manifest(archive, status="complete")
    case = {
        "graph": "G22", "status": "converged",
        "graph_sha256": graph_sha, "factor_sweeps": 7,
        "factor_rank": 4, "archive_path": str(archive["path"]),
        "provenance": {"seed": 2026, "tolerance": 0.001,
                       "rounding_seed": 2026},
    }
    case_path = root / source / "cases" / "G22.json"
    case_path.parent.mkdir(parents=True)
    case_path.write_text(json.dumps(case))
    log_path = root / source / "logs" / "G22-sweeps.jsonl"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        '{"graph":"G22","sweep":1,"objective":1.2,"max_angle":0.4}\n'
        '{"graph":"G22","sweep":7,"objective":2.0,"max_angle":0.0005}\n'
    )
    return root, source, graph, vectors, case, factor_path, labels, scores


def test_read_existing_factor_and_archived_partitions(archive_fixture):
    root, source, graph, original_y, case, path, labels, scores = archive_fixture
    assert discover_eva_cases(root, source) == ["G22"]
    loaded_graph, y, saved, factor_hash = load_eva_artifacts(
        "G22", root, source
    )
    np.testing.assert_array_equal(y, original_y)
    assert loaded_graph.shape == (5, 5)
    assert factor_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    result = archived_roundings("G22", saved, factor_hash, root, source)
    assert [r[0] for r in result["K-MeansNM_EVA"]] == [0, 1]
    for (_, score, actual), expected, stored_score in zip(
        result["K-MeansNM_EVA"], labels, scores
    ):
        np.testing.assert_array_equal(actual, expected)
        assert score == stored_score


def test_graph_hash_mismatch_is_rejected(archive_fixture):
    root, source, *_ = archive_fixture
    case_path = root / source / "cases" / "G22.json"
    record = json.loads(case_path.read_text())
    record["graph_sha256"] = "not-the-original"
    case_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Graph input"):
        load_eva_artifacts("G22", root, source)


def test_convex_cache_visualizes_without_solver(archive_fixture):
    root, source, graph, y, case, *_ = archive_fixture
    path = root / "reports/gset/sdp_cache/G22-k2-synthetic.npz"
    path.parent.mkdir(parents=True)
    x = y @ y.T
    np.savez(path, gram=x, metadata=np.array(json.dumps({
        "identity": {"graph_sha256": case["graph_sha256"],
                     "solver": "SCS"}, "solver_info": {}
    })))
    g, coordinates, metadata = load_convex_embedding("G22", root)
    assert coordinates.shape == (5, 3)
    assert metadata["solver"] == "SCS"
    assert not np.isnan(coordinates).any()
    with pytest.raises(ValueError, match="disabled"):
        load_convex_embedding("G22", root, max_n=4)


def test_projection_sampling_spectrum_and_metrics(archive_fixture):
    pytest.importorskip("sklearn")
    root, source, graph, y, *_ = archive_fixture
    indices = sample_vertices(len(y), limit=4, seed=13)
    assert len(set(indices)) == 4
    coords, info = project_vectors(y, "PCA", selected=indices)
    assert coords.shape == (4, 3)
    assert 0 <= sum(info["explained_variance_ratio"]) <= 1.0000001
    assert len(pairwise_cosines(y, number=20)) <= 20
    assert (vertex_metrics(graph)["Degree"] >= 0).all()
    with pytest.raises(ValueError, match="Unknown"):
        project_vectors(y, "not-a-projection")


def test_rounding_replay_keeps_factor_unchanged(archive_fixture):
    root, source, graph, y, *_ = archive_fixture
    before = y.copy()
    frames = replay_kmeans(y, graph, "G22", restart=2, seed=2026)
    assert frames
    assert frames[0]["step"] == 1
    for frame in frames:
        assert frame["labels"].shape == (5,)
        assert frame["best_cut"] >= frame["cut"]
    assert frames[-1]["best_cut"] >= frames[0]["cut"]
    np.testing.assert_array_equal(y, before)
    assert load_sweep_log("G22", root, source)[-1]["objective"] == 2.0


def test_extension_uses_exact_factor_fingerprint(archive_fixture):
    root, source, graph, y, case, factor_file, labels, scores = archive_fixture
    factor_sha = hashlib.sha256(factor_file.read_bytes()).hexdigest()
    edges = edge_arrays(graph)
    folder = root / source / "rounding-extensions"
    archive = new_archive(folder / "archives", "synthetic0002",
                          root / "data/G22.txt", len(y), edges, 2,
                          {"source": "test"}, jsonl_output=folder / "trial.jsonl")
    save_method(archive, "FJ_EVA", labels[:1], scores[:1], k=2)
    update_manifest(archive, status="complete")
    segment = folder / "segments" / f"G22-{factor_sha[:16]}" / "FJ_EVA" / "r00020-00021.json"
    segment.parent.mkdir(parents=True)
    segment.write_text(json.dumps({
        "graph_sha256": case["graph_sha256"],
        "factor_sha256": factor_sha,
        "method": "FJ_EVA", "start_restart": 20,
        "stop_restart_exclusive": 21,
        "archive_path": str(archive["path"]),
    }))
    result = archived_roundings("G22", case, factor_sha, root, source)
    assert result["FJ_EVA"][0][0] == 20
    no_match = archived_roundings("G22", case, "other-factor", root, source)
    assert "FJ_EVA" not in no_match
