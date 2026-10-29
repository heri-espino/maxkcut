"""Experiment016: Bernoulli EM, CEM, Gibbs and KL signed Max-Cut."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("numba")

from maxkcut.archive import (
    new_archive, save_method, update_manifest, verify_archive,
    load_method, new_run_id,
)
from maxkcut.eva_cli import run_one
from maxkcut.eva_rounding_cli import _factor_and_case
from maxkcut.eva_soft_em_bernoulli import (
    _free_energy, aligned, cem, gibbs_anneal, soft_em,
)
from maxkcut.eva_soft_em_bernoulli_cli import (
    run_graph, load_elites, STAGES,
)
from maxkcut.io import edge_arrays, load_graph
from maxkcut.objective import cut_weight


def signed_graph(n=12,seed=11):
    rng=np.random.default_rng(seed)
    a=rng.integers(-4,5,size=(n,n)).astype(float)
    a[np.tril_indices(n)]=0
    return sparse.csr_matrix(a+a.T)


def factor(n=12,p=5,seed=8):
    y=np.random.default_rng(seed).normal(size=(n,p))
    return y/np.linalg.norm(y,axis=1)[:,None]


@pytest.mark.parametrize("mode",["mean","elite"])
def test_soft_em_probabilities_and_fixed_temperature_free_energy(mode):
    g=signed_graph()
    y=factor()
    anchor=np.random.default_rng(32).integers(0,2,size=12,dtype=np.int8)
    original=y.copy()
    p,labels,report=soft_em(
        g,y,np.random.default_rng(2026),
        mode=mode,anchor=anchor if mode=="elite" else None,
        kappa=1.,delta=.35,temp_start=3.5,temp_end=.5,
        stages=4,em_iters=8,damping=.4,
    )
    np.testing.assert_array_equal(original,y)
    assert p.shape==(len(y),)
    assert np.isfinite(p).all()
    assert np.all((p>0)&(p<1))
    np.testing.assert_array_equal(labels,(p>=.5).astype(np.int8))
    assert report["hard_cut"]==cut_weight(edge_arrays(g),labels)
    by_stage={}
    for entry in report["trace"]:
        by_stage.setdefault(entry["stage"],[]).append(
            entry["free_energy"]
        )
    for values in by_stage.values():
        assert all(
            next_value>=prior-1e-6
            for prior,next_value in zip(values,values[1:])
        )


def test_soft_em_reproducible_and_no_division_by_zero_for_mean_zero():
    g=signed_graph(8,seed=7)
    y=factor(4,4,seed=5)
    y=np.vstack((y,-y))
    assert np.linalg.norm(y.mean(axis=0))<1e-12
    x=soft_em(
        g,y,np.random.default_rng(11),
        stages=2,em_iters=4,
    )
    z=soft_em(
        g,y,np.random.default_rng(11),
        stages=2,em_iters=4,
    )
    np.testing.assert_array_equal(x[0],z[0])
    np.testing.assert_array_equal(x[1],z[1])


def test_both_sides_of_same_cut_are_equivalent_for_CEM():
    reference=np.array([0,1,0,1,1,0,0,1],dtype=np.int8)
    np.testing.assert_array_equal(
        aligned(reference,1-reference),reference,
    )
    graph=signed_graph(8,seed=17)
    probs=np.full(8,.5)
    source_score=cut_weight(edge_arrays(graph),reference)
    first,part,report=cem(
        graph,probs,np.random.default_rng(1),
        anchor=reference,generations=5,population=20,
    )
    assert report["best_cut"]>=source_score
    assert report["best_cut"]==cut_weight(edge_arrays(graph),part)
    assert np.all((first>0)&(first<1))
    assert all(b["best_cut"]>=a["best_cut"]
               for a,b in zip(report["trace"],report["trace"][1:]))


def test_cem_and_gibbs_signed_replay_and_best_incumbent():
    graph=signed_graph(18,seed=91)
    edges=edge_arrays(graph)
    seed=np.random.default_rng(9).integers(0,2,size=18,dtype=np.int8)
    p=np.where(seed,.96,.04)
    def run_cem():
        return cem(
            graph,p,np.random.default_rng(19),anchor=seed,
            generations=6,population=18,elite_fraction=.2,
            injection=.02,
        )
    a=run_cem()
    b=run_cem()
    np.testing.assert_array_equal(a[0],b[0])
    np.testing.assert_array_equal(a[1],b[1])
    assert a[2]["best_cut"]>=cut_weight(edges,seed)
    def run_gibbs():
        return gibbs_anneal(
            graph,a[1],np.random.default_rng(20),
            sweeps=15,temp_start=6.,temp_end=.15,
        )
    ga=run_gibbs()
    gb=run_gibbs()
    np.testing.assert_array_equal(ga[0],gb[0])
    assert ga[1]["best_cut"]>=a[2]["best_cut"]
    assert ga[1]["best_cut"]==cut_weight(edges,ga[0])


def _source(tmp):
    path=tmp/"G11.txt"
    path.write_text(
        "7 10\n1 2 2\n1 3 -1\n1 4 1\n2 3 2\n"
        "2 5 1\n3 4 -2\n4 5 3\n4 6 2\n5 7 -1\n6 7 2\n",
        encoding="utf8",
    )
    factor_dir=tmp/"original-EVA"
    opts=SimpleNamespace(
        max_n=20,output_dir=factor_dir,rank=None,
        tolerance=1e-6,max_sweeps=12,seed=2026,
        checkpoint_every=2,rounding_seed=2026,
        restarts=2,cpu_threads=1,convex_dir=None,
        push=False,push_branch=None,push_every=0,
    )
    run_one(path,opts)
    graph,y,_,graph_sha,factor_sha=_factor_and_case(path,factor_dir)
    elites_dir=tmp/"experiment013"
    labels=np.array([0,1,0,1,0,1,0],dtype=np.int8)
    score=float(cut_weight(edge_arrays(graph),labels))
    archive=new_archive(
        elites_dir/"archives",new_run_id(),path,len(labels),
        edge_arrays(graph),2,{"synthetic":True},
        jsonl_output=elites_dir/"trials"/"G11.jsonl",
    )
    save_method(archive,"Grid-Final_EVA",[labels],[score],k=2)
    update_manifest(archive,status="complete")
    case={
        "status":"complete",
        "settings":{
            "graph_sha256":graph_sha,
            "factor_sha256":factor_sha,
        },
        "archive_path":str(archive["path"]),
        "trials":[{"restart":5,"scores":{"Final":score}}],
    }
    case_path=elites_dir/"cases"/"G11-grid-synthetic.json"
    case_path.parent.mkdir(parents=True,exist_ok=True)
    case_path.write_text(json.dumps(case),encoding="utf8")
    return path,factor_dir,elites_dir,labels,score


def test_verified_elite_source_rejects_bad_graph_and_missing_archive(tmp_path):
    path,source,elites,expected,score=_source(tmp_path)
    _,_,_,_,sha=_factor_and_case(path,source)
    found=load_elites(path,sha,elites,limit=2)
    assert len(found)==1
    np.testing.assert_array_equal(found[0]["labels"],expected)
    assert found[0]["score"]==score
    with pytest.raises(FileNotFoundError):
        load_elites(path,"0"*64,elites,limit=2)


def test_parallel_four_method_two_seed_inits_archive_and_resume(tmp_path):
    path,source,elite_dir,elite_labels,elite_cut=_source(tmp_path)
    source_factor=source/"factors"/"G11.npz"
    before=hashlib.sha256(source_factor.read_bytes()).hexdigest()
    dest=tmp_path/"experiment016"
    opts=dict(
        factor_dir=source,dest=dest,elite_dir=elite_dir,
        methods=("em-kl","em-cem","em-gibbs","em-cem-kl"),
        initializations=("mean","elite"),
        episodes=2,chunk_size=1,workers=2,
        em_stages=2,em_iters=3,cem_generations=3,
        population=10,gibbs_sweeps=3,kl_passes=3,
        seed=14,target=1000.,
    )
    fresh=run_graph(path,**opts)
    assert len(fresh)==16
    assert hashlib.sha256(source_factor.read_bytes()).hexdigest()==before
    cases=list((dest/"cases").glob("G11-em-*.json"))
    assert len(cases)==16
    edges=edge_arrays(load_graph(path))
    for case_file in cases:
        record=json.loads(case_file.read_text(encoding="utf8"))
        report=verify_archive(record["archive_path"])
        assert report["status"]=="complete"
        for stage in STAGES:
            name=f"Exp016-{stage}"
            assert report["methods"][name]["verified_partitions"]==1
            parts,scores=load_method(
                Path(record["archive_path"])/f"{name}.npz"
            )
            assert cut_weight(edges,parts[0])==scores[0]
        if record["initialization"]=="elite":
            assert record["trials"][0]["scores"]["Source"]==elite_cut
    before_archives=len(list((dest/"archives").iterdir()))
    second=run_graph(path,**opts)
    assert {
        rec["id"]:rec["best_final"] for rec in fresh
    }=={
        rec["id"]:rec["best_final"] for rec in second
    }
    assert len(list((dest/"archives").iterdir()))==before_archives
    assert (dest/"summary.csv").is_file()
