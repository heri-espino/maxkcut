"""Synthetic and non-numerical regression tests for Experiment029 extension."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _script(name, filename):
    path=Path(__file__).resolve().parents[1]/"scripts"/filename
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

@pytest.fixture
def modules():
    return (_script("experiment028", "experiment028_fullrank_pilot.py"),
            _script("experiment029", "experiment029_kmeansnm_100seeds.py"))


def make_synthetic(tmp_path):
    graph_dir=tmp_path/"data";graph_dir.mkdir()
    # Seven vertices, signed, asymmetric topology; n=7 -> default p=4.
    edges=[(1,2,2),(1,3,-1),(2,3,1),(2,5,-2),(3,4,3),
           (4,5,1),(5,6,-1),(6,7,3),(7,1,2),(4,7,-1)]
    body="7 10\n"+"".join(f"{i} {j} {w}\n" for i,j,w in edges)
    path=graph_dir/"G1.txt";path.write_text(body)
    h=hashlib.sha256(body.encode()).hexdigest()
    features=tmp_path/"features.csv"
    features.write_text("graph,graph_sha256,n,family,weight_type\nG1,"+h+",7,synthetic,signed\n")
    return graph_dir,features


def test_reuse_old_factors_and_only_compute_missing_seeds(tmp_path,modules):
    old,new=modules
    g,f=make_synthetic(tmp_path)
    metadata=old.metadata_from_csv(f)["G1"]
    graph=old.load_graph(g/"G1.txt");edges=old.edge_arrays(graph)
    baseline=tmp_path/"experiment028"
    for seed in (2026,2027):
        for mode in ("default","full"):
            old.one_case("G1",graph,edges,metadata,mode,seed,baseline,
                         max_sweeps=80,tolerance=1e-3,checkpoint_every=8)
    base_hashes={p:p.read_bytes() for p in baseline.rglob("factor.npz")}
    out=tmp_path/"experiment029"
    args=["--graphs","G1","--old-seeds","2026","2027",
          "--new-seeds","2028","2029","--allow-synthetic",
          "--graph-dir",str(g),"--features",str(f),
          "--prior",str(baseline),"--output",str(out),"--workers","1",
          "--max-sweeps","80","--tolerance","0.001","--checkpoint-every","8"]
    new.main(args)
    assert all(p.read_bytes()==v for p,v in base_hashes.items())
    with (out/"all_runs.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==8
    assert {r["source_experiment"] for r in rows}=={"experiment028","experiment029"}
    assert {int(r["p"]) for r in rows}=={4,7}
    assert {r["graph"] for r in rows}=={"G1"}
    assert len(list((out/"runs").glob("*/**/case.json")))==4
    with (out/"paired_differences.csv").open(newline="") as f:
        paired=list(csv.DictReader(f))
    assert len(paired)==4
    for row in paired:
        assert abs(float(row["delta_full_minus_default"]) -
                (float(row["full_cut"])-float(row["default_cut"])))<1e-10
    # Check mode must audit all eight exact original signed hard cuts.
    new.main(args+["--check-only"])
    # Resume must not regenerate previous case files or change results.
    before=[p.stat().st_mtime_ns for p in out.rglob("case.json")]
    new.main(args)
    after=[p.stat().st_mtime_ns for p in out.rglob("case.json")]
    assert before==after


def test_changed_protocol_rejected(modules):
    _,mod=modules
    with pytest.raises(SystemExit):
        mod.main(["--graphs","G1","--old-seeds","2026","--new-seeds","2031"])
    assert mod.OLD_SEEDS==(2026,2027,2028,2029,2030)
    assert len(mod.NEW_SEEDS)==95
    assert mod.ALL_SEEDS[0]==2026 and mod.ALL_SEEDS[-1]==2125
    assert len(mod.ALL_SEEDS)==100
    assert mod.METHOD=="K-MeansNM_EVA"


def test_rank_seed_fingerprint_and_100seed_target(modules):
    _,mod=modules
    assert len(mod.GRAPHS)==5
    assert 5*2*len(mod.ALL_SEEDS)==1000
    assert 5*2*len(mod.NEW_SEEDS)==950
    assert 5*2*len(mod.OLD_SEEDS)==50
    ss=mod.rng_for(2031,"G6")
    assert np.array_equal(ss.standard_normal(4),mod.rng_for(2031,"G6").standard_normal(4))
    assert not np.array_equal(mod.rng_for(2031,"G6").standard_normal(4),mod.rng_for(2031,"G14").standard_normal(4))