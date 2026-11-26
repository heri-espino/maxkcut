"""Self-contained, quick protocol checks: no G-set or VM access required."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiment028_fullrank_pilot.py"
spec = importlib.util.spec_from_file_location("experiment028_fullrank_pilot", SCRIPT)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_seed_stream_pairing_and_rank_definition():
    assert list(mod.GRAPHS) == ["G1", "G6", "G11", "G14", "G18"]
    for seed in mod.SEEDS:
        a = np.random.default_rng(mod.seed_sequence(seed, "G1", "K-MeansNM_EVA")).standard_normal(25)
        b = np.random.default_rng(mod.seed_sequence(seed, "G1", "K-MeansNM_EVA")).standard_normal(25)
        c = np.random.default_rng(mod.seed_sequence(seed, "G1", "K-Means2N_EVA")).standard_normal(25)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)
    assert int(np.ceil(np.sqrt(2*(800+1)))) == 41


def test_windows_linux_line_ending_identity(tmp_path: Path):
    p = tmp_path / "graph.txt"
    p.write_bytes(b"3 2\n1 2 1\n2 3 -1\n")
    hashes = mod.graph_byte_hashes(p)
    assert hashes["lf"] == hashes["raw"]
    assert hashes["crlf"] == hashlib.sha256(b"3 2\r\n1 2 1\r\n2 3 -1\r\n").hexdigest()
    assert hashes["crlf"] != hashes["lf"]


def test_synthetic_factor_and_cluster_protocol(tmp_path: Path):
    graphdir = tmp_path / "graphs"
    graphdir.mkdir()
    # Nontrivial signed 7-vertex graph with an odd cycle and extra chords.
    edges = [(1,2,1),(2,3,-2),(3,4,2),(4,5,1),(5,6,-1),(6,7,2),
             (7,1,1),(1,4,-1),(2,5,2),(3,6,1),(4,7,-2)]
    data = graphdir / "G1.txt"
    data.write_text("7 11\n" + "\n".join(f"{a} {b} {w}" for a,b,w in edges) + "\n")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    features = tmp_path / "features.csv"
    with features.open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=["graph","graph_sha256","n","family","weight_type"])
        writer.writeheader()
        writer.writerow({"graph":"G1","graph_sha256":digest,"n":7,"family":"synthetic","weight_type":"signed"})
    out = tmp_path / "results"
    args=["--graphs","G1","--seeds","2026","2027","--graph-dir",str(graphdir),
          "--features",str(features),"--output",str(out),"--max-sweeps","20",
          "--tolerance","1e-3","--checkpoint-every","5","--allow-synthetic"]
    mod.main(args)
    summary=json.loads((out / "summary.json").read_text())
    assert summary["complete"] and summary["observed_rows"]==12
    with (out / "all_runs.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==12
    assert {int(row["p"]) for row in rows}=={4,7}  # ceil(sqrt(16))=4, versus n=7
    for row in rows:
        assert np.isfinite(float(row["cut_weight"]))
        assert row["factor_sha256"]
    with (out / "paired_differences.csv").open(newline="") as f:
        pairs=list(csv.DictReader(f))
    assert len(pairs)==6
    for r in pairs:
        assert abs(float(r["delta_full_minus_default"]) -
                   (float(r["full_cut"]) - float(r["default_cut"]))) < 1e-12
    # Exact same call must resume without re-solving or changing archived cases.
    before=[p.stat().st_mtime_ns for p in sorted(out.rglob("case.json"))]
    mod.main(args)
    after=[p.stat().st_mtime_ns for p in sorted(out.rglob("case.json"))]
    assert before==after
    mod.main(args+["--check-only"])


def test_protected_command_arguments():
    with pytest.raises(SystemExit):
        mod.main(["--graphs", "G1", "--seeds", "2026"])
    with pytest.raises(SystemExit):
        mod.main(["--graphs", "G1", "--seeds", "2026", "2026", "--allow-synthetic"])