"""Experiment020 all G-set available-EVA inventory and paired statistics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from maxkcut.eva_kmeans_initializers import INIT2
from maxkcut.eva_kmeans_init_all_cli import (
    _bootstrap_mean_ci, _read_long_rows, aggregate, inventory,
)


def _summary(path,graph,restarts=3,shift=0):
    path.parent.mkdir(parents=True,exist_ok=True)
    cols=[
        "graph","initializer","restart","raw_cut",
        "flip_cut","kl_cut","final_cut","improvement",
        "gap_to_target","extra_start_cuts",
        "lloyd_iterations","ils_cycles","seconds",
        "archive_path","case_id",
    ]
    with path.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader()
        for init_id,init in enumerate(INIT2):
            for restart in range(restarts):
                raw=60+restart+shift
                # Pair-correct score shifts are easy to check exactly.
                gain=init_id%3
                writer.writerow({
                    "graph":graph,"initializer":init,
                    "restart":restart,"raw_cut":raw,
                    "flip_cut":raw+1,"kl_cut":raw+2,
                    "final_cut":raw+3+gain,
                    "improvement":3+gain,
                    "gap_to_target":100-(raw+3+gain),
                    "extra_start_cuts":16 if init=="gw-screen16" else 0,
                    "lloyd_iterations":4,"ils_cycles":5,
                    "seconds":1.5,
                    "archive_path":f"archives/{graph}/{init}/{restart}",
                    "case_id":f"id-{graph}-{init}-{restart}",
                })


def test_inventory_distinguishes_completed_from_missing_factors(
    tmp_path,monkeypatch,
):
    data=tmp_path/"data"
    data.mkdir()
    for name in ("G1","G2","G3","G63"):
        (data/f"{name}.txt").write_text("placeholder",encoding="utf8")
    a=tmp_path/"older"
    b=tmp_path/"numba"
    for base,name in ((a,"G1"),(b,"G2"),(b,"G63")):
        for item,ext in (("cases","json"),("factors","npz")):
            folder=base/item
            folder.mkdir(parents=True,exist_ok=True)
            (folder/f"{name}.{ext}").write_text("sample",encoding="utf8")
    def fake_check(path,source):
        n=int(path.stem[1:])
        return np.zeros((4,4)),np.zeros((4,2)),{},(
            f"graph-sha-{n}"),f"factor-sha-{n}"
    monkeypatch.setattr(
        "maxkcut.eva_kmeans_init_all_cli._factor_and_case",fake_check
    )
    doc=inventory(
        data_dir=data,out=tmp_path/"out",
        sources=(b,a),reused=tmp_path/"previous-G63",
    )
    assert doc["graph_count"]==4
    assert doc["verified_ready"]==2
    assert doc["reused_verified_candidate"]==1
    assert doc["unavailable_count"]==1
    status={r["graph"]:r["status"] for r in doc["records"]}
    assert status=={
        "G1":"ready","G2":"ready","G3":"no_completed_EVA_factor",
        "G63":"reused_019",
    }
    assert json.loads(
        (tmp_path/"out"/"inventory.json").read_text(encoding="utf8")
    )["verified_ready"]==2


def test_bootstrap_seed_reproducible():
    x=np.array([-2.,1.,7.,0.,3.])
    assert _bootstrap_mean_ci(x,seed=101,samples=200)==(
        _bootstrap_mean_ci(x,seed=101,samples=200)
    )
    assert _bootstrap_mean_ci(np.array([2.,2.,2.]))==[2.,2.]


def test_exact_paired_rows_long_format_and_missing_detection(tmp_path):
    a=tmp_path/"G1"/"summary.csv"
    _summary(a,"G1",restarts=3)
    rec={"graph":"G1","output_dir":str(a.parent),
         "factor_source":"source1"}
    rows=_read_long_rows(rec,restarts=3)
    assert len(rows)==len(INIT2)*3
    assert rows[0]["rounding_seed"]==2026
    assert all(x["search_seed"]==2026 for x in rows)
    assert all(x["rounding_seed_stream"]==(
        0 if x["initializer"]=="paired" else 1
    ) for x in rows)
    assert _read_long_rows(rec,restarts=4) is None


def test_aggregate_complete_statistics_and_immutable_coverage(tmp_path):
    out=tmp_path/"results"
    record=lambda name,status,source:(dict(
        graph=name,status=status,factor_source=source,
        output_dir=str(
            out/name if name!="G63" else out/"reference019"
        ),
    ))
    manifest={
        "graph_count":3,
        "records":[
            record("G1","ready","experiment002"),
            record("G2","no_completed_EVA_factor",None),
            record("G63","reused_019","experiment003"),
        ],
    }
    out.mkdir(parents=True)
    (out/"inventory.json").write_text(
        json.dumps(manifest),encoding="utf8",
    )
    _summary(out/"G1"/"summary.csv","G1",restarts=3)
    _summary(out/"reference019"/"summary.csv","G63",restarts=3,
             shift=20)
    report=aggregate(
        out=out,restarts=3,bootstrap_samples=100,
        require_complete=True,
    )
    assert report["complete_graphs"]==2
    assert report["runs_in_all_runs_csv"]==2*len(INIT2)*3
    assert report["no_EVA_factor_count"]==1
    assert report["reused_existing_graphs"]==1
    stats=out/"statistics"
    with (stats/"paired_vs_original_kmeansnm.csv").open(
        newline="",encoding="utf8",
    ) as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==2*len(INIT2)
    one=next(x for x in rows
             if x["graph"]=="G1" and x["initializer"]=="sample-pair")
    assert float(one["mean_delta"])==0.
    assert int(one["ties"])==3
    high=next(x for x in rows
              if x["graph"]=="G1" and x["initializer"]=="independent")
    assert float(high["mean_delta"])==1.
    assert int(high["wins"])==3
    assert (stats/"per_instance_methods.csv").exists()
    assert (stats/"all_runs.csv").exists()
    assert (stats/"coverage.csv").exists()


def test_incomplete_graph_never_masquerades_as_complete(tmp_path):
    out=tmp_path/"r"
    out.mkdir()
    manifest={
        "graph_count":1,
        "records":[{"graph":"G1","status":"ready",
                    "factor_source":"original",
                    "output_dir":str(out/"G1")}],
    }
    (out/"inventory.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        aggregate(out=out,restarts=3,require_complete=True)
    report=aggregate(out=out,restarts=3)
    assert report["incomplete_candidate_count"]==1
    assert report["runs_in_all_runs_csv"]==0
