"""Experiment019: signed fixed-EVA 9-initializer Flip+KL+ILS regression."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("numba")

from maxkcut.archive import load_method, verify_archive
from maxkcut.eva_cli import run_one
from maxkcut.eva_kmeans_initializers import INIT2
from maxkcut.eva_kmeans_init_full_cli import (
    run_graph, STAGES,
)
from maxkcut.eva_pipeline_grid import DEFAULTS, evaluate_pipeline
from maxkcut.eva_rounding_cli import _factor_and_case
from maxkcut.eva_multigroup_kmeans import optimize_multigroup_kmeans
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight


def _saved_eva(tmp):
    n=12
    edges=[]
    for i in range(1,n+1):
        for j in range(i+1,n+1):
            if (i+2*j)%3==0:
                w=(i*j)%7-3
                if w==0:
                    w=1
                edges.append((i,j,w))
    path=tmp/"G11.txt"
    path.write_text(
        f"{n} {len(edges)}\n"+
        "\n".join(f"{i} {j} {w}" for i,j,w in edges)
        +"\n",encoding="utf8",
    )
    source=tmp/"eva"
    args=SimpleNamespace(
        max_n=20,output_dir=source,rank=None,
        tolerance=1e-6,max_sweeps=12,seed=2026,
        checkpoint_every=2,rounding_seed=2026,
        restarts=2,cpu_threads=1,convex_dir=None,
        push=False,push_branch=None,push_every=0,
    )
    run_one(path,args)
    return path,source


def test_paired_full_pipeline_reproduces_original_013_stages(tmp_path):
    path,source=_saved_eva(tmp_path)
    graph,y,_,_,_=_factor_and_case(path,source)
    edges=edge_arrays(graph)
    restart=4
    seed=2026
    raw_candidates,_=optimize_multigroup_kmeans(
        y,graph,
        np.random.default_rng(np.random.SeedSequence(
            [seed,11,2,restart,0],
        )),
        k=2,init="paired",max_iter=100,top_merges=1,
    )
    raw=raw_candidates[0]["labels"]
    cfg={
        **DEFAULTS,
        "rounder":"saved-kmeans-flip",
        "flip_sweeps":3,"kl_passes":4,
        "ils_cycles":3,"ils_kl_passes":2,
        "kick_sizes":[1,2,3,4],
        "temperature":12.,
    }
    expected,expected_score,_,_=evaluate_pipeline(
        graph,y,saved_labels=raw,
        saved_score=float(cut_weight(edges,raw)),
        restart=restart,graph_number=11,
        rounding_seed=seed,search_seed=seed,
        config=cfg,target=10000,
    )
    dest=tmp_path/"full"
    records,_=run_graph(
        path,factor_dir=source,dest=dest,
        methods=["paired"],restarts=1,start_restart=restart,
        chunk_size=1,workers=1,
        kmeans_iters=100,flip_sweeps=3,kl_passes=4,
        ils_cycles=3,ils_kl_passes=2,
        kick_sizes=(1,2,3,4),target=10000.,
    )
    record=records[0]
    for stage in STAGES:
        labels,scores=load_method(
            Path(record["archive_path"])/f"Exp019-{stage}_EVA.npz"
        )
        np.testing.assert_array_equal(labels[0],expected[stage])
        assert scores[0]==pytest.approx(expected_score[stage])
    assert record["trials"][0]["rounding"]["init"]=="paired"


def test_full_9_method_parallel_signed_archives_resume(tmp_path):
    path,source=_saved_eva(tmp_path)
    frozen=source/"factors"/"G11.npz"
    digest=hashlib.sha256(frozen.read_bytes()).hexdigest()
    dest=tmp_path/"experiment019"
    opts=dict(
        factor_dir=source,dest=dest,methods=INIT2,
        restarts=2,chunk_size=1,workers=2,
        kmeans_iters=8,flip_sweeps=3,
        kl_passes=3,ils_cycles=3,ils_kl_passes=2,
        kick_sizes=(1,2,3),target=10000.,
    )
    done,ranks=run_graph(path,**opts)
    assert len(done)==len(INIT2)*2
    assert len(ranks)==len(INIT2)
    assert all(r["restarts"]==2 for r in ranks)
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==digest
    edges=edge_arrays(load_graph(path))
    assert len(list((dest/"cases").glob("G11-init-full-*.json")))==len(done)
    for record in done:
        report=verify_archive(record["archive_path"])
        assert report["status"]=="complete"
        trial=record["trials"][0]
        values=[]
        for stage in STAGES:
            labels,scores=load_method(
                Path(record["archive_path"])/f"Exp019-{stage}_EVA.npz"
            )
            assert len(labels)==1
            assert cut_weight(edges,labels[0])==scores[0]
            assert scores[0]==trial["scores"][stage]
            values.append(scores[0])
        assert values==sorted(values)
        assert trial["ils"]["cycles_completed"]==3
    again,second_ranks=run_graph(path,**opts)
    assert {x["id"]:x["best_final"] for x in done}=={
        x["id"]:x["best_final"] for x in again
    }
    assert ranks==second_ranks
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==digest
    assert (dest/"summary.csv").is_file()
    assert (dest/"leaderboard.json").is_file()


def test_settings_fingerprints_preserve_distinct_budgets(tmp_path):
    path,source=_saved_eva(tmp_path)
    dest=tmp_path/"full"
    common=dict(
        factor_dir=source,dest=dest,
        methods=("paired",),restarts=1,
        chunk_size=1,workers=1,kmeans_iters=6,
        flip_sweeps=2,kl_passes=2,ils_kl_passes=1,
        kick_sizes=(1,2,3),target=10000.,
    )
    a,_=run_graph(path,ils_cycles=2,**common)
    b,_=run_graph(path,ils_cycles=3,**common)
    assert a[0]["id"]!=b[0]["id"]
    assert len(list((dest/"cases").glob("G11-init-full-*.json")))==2


def test_invalid_initializers_and_budgets(tmp_path):
    path,source=_saved_eva(tmp_path)
    with pytest.raises(ValueError):
        run_graph(
            path,factor_dir=source,dest=tmp_path/"out",
            methods=["wrong"],restarts=1,
        )
    with pytest.raises(ValueError):
        run_graph(
            path,factor_dir=source,dest=tmp_path/"out",
            methods=["paired"],restarts=1,ils_cycles=0,
        )
