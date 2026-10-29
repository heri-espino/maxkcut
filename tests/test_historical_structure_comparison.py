"""Synthetic tests for strict and stage-aware retrospective historical comparisons."""
import csv
from pathlib import Path

import pytest
from maxkcut.historical_structure_comparison import (build_raw, aggregate_raw,
    build_g63, read_csv, must_g63, SOURCES, execute, RAW)


def test_62_raw_graphs_and_categories():
    def row(g):
        return dict(graph=g,status="converged",eva_FJ_best="10",
                    eva_KMeans2N_best="12",eva_KMeansNM_best="13")
    d2=[row(f"G{i}") for i in range(1,60)]
    d3=[row("G77"),row("G81")]
    d8=[dict(graph="G63", method=m, rounding_count="20",best_cut=str(v))
        for m,v in (("FJ_EVA",10),("K-Means2N_EVA",12),("K-MeansNM_EVA",13))]
    rows=build_raw(d2,d3,d8)
    assert len(rows)==186
    assert len([r for r in rows if r["method"]=="KMeansNM_EVA" and r["shared_best"]])==62
    agg=aggregate_raw(rows)
    assert {r["family"]:r["graphs"] for r in agg if r["method"]=="GW/FJ_EVA"} == {
        "random":27,"planar":23,"toroidal":12}
    assert all(r["gain_vs_GW"]==3 for r in rows if r["method"]=="KMeansNM_EVA")


def test_62_raw_rejects_missing_and_duplicate():
    d2=[dict(graph=f"G{i}",status="converged",eva_FJ_best="1",
             eva_KMeans2N_best="2",eva_KMeansNM_best="3") for i in range(1,60)]
    d3=[dict(graph="G77",status="converged",eva_FJ_best="1",
             eva_KMeans2N_best="2",eva_KMeansNM_best="3"),
        dict(graph="G81",status="converged",eva_FJ_best="1",
             eva_KMeans2N_best="2",eva_KMeansNM_best="3")]
    d8=[dict(graph="G63",method=m,rounding_count="20",best_cut="1") for m in
        ("FJ_EVA","K-Means2N_EVA","K-MeansNM_EVA")]
    with pytest.raises(ValueError,match="Expected 59"):
        build_raw(d2[:-1],d3,d8)
    with pytest.raises(ValueError,match="duplicate"):
        build_raw(d2,d3+[d3[0]],d8)
    with pytest.raises(ValueError,match="missing one"):
        build_raw(d2,d3,d8[:2])


def test_g63_soft_em_retains_elite_and_not_improvement():
    def d(graph="G63",**kw):return dict(graph=graph,**{k:str(v) for k,v in kw.items()})
    data={
        "008":[d(method="FJ_EVA",best_cut=10)],
        "009":[d(metric="l2",q=1,final_cut=3,best_coarse_cut=1,longest_edge_baseline=1)],
        "012":[d(stage="CutAwareKMeansKL",cut=11)],
        "013":[d(family="SavedKMeansFlip+KL+ILS",config_id="a",source_cut=10,round_cut=9,final_cut=12)],
        "016":[d(initialization="elite",method="em-cem",source_cut=26917,em_cut=26808,final_cut=26917,config_id="a")],
        "017":[d(k=4,init="paired",post_KL_cut=12,raw_cut=11)],
        "018":[d(init="paired",post_KL_cut=12,raw_cut=10)],
        "019":[d(initializer="paired",final_cut=12,raw_cut=10)],
    }
    rows=build_g63(data)
    em=next(r for r in rows if r["method"]=="EM-elite-em-cem")
    assert em["best_cut"]==26917 and em["max_per_record_gain_from_source"]==0
    assert em["positive_improvements_on_own_source"]==0
    assert em["retained_elite_source"]==1
    mst=next(r for r in rows if "MST-single-edge" in r["method"])
    assert mst["best_cut"]==1


def test_missing_summary_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        execute(tmp_path,tmp_path/"new")
    assert not (tmp_path/"new").exists()


def test_reader_header_and_g63_scope(tmp_path):
    p=tmp_path/"empty.csv"
    p.write_text("",encoding="utf8")
    with pytest.raises(ValueError,match="Empty CSV"):
        read_csv(p)
    with pytest.raises(ValueError,match="G63-only"):
        must_g63([dict(graph="G11")],"016")