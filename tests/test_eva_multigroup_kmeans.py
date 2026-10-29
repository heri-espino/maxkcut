"""Experiment017 k-group KMeansNM + exact signed group coarsening tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("numba")

from maxkcut.archive import load_method,verify_archive
from maxkcut.eva_cli import run_one
from maxkcut.eva_rounding_cli import _factor_and_case
from maxkcut.eva_multigroup_kmeans import (
    enumerate_group_merges,optimize_multigroup_kmeans,
)
from maxkcut.eva_multigroup_kmeans_cli import (
    run_graph,STAGES,EXTRA_STAGES,
)
from maxkcut.io import edge_arrays,load_graph
from maxkcut.objective import cut_weight
from maxkcut.rounding import round_spherical_kmeans


def _graph(n=24,seed=7):
    rng=np.random.default_rng(seed)
    a=rng.integers(-4,7,size=(n,n)).astype(float)
    a[np.tril_indices(n)]=0
    return sparse.csr_matrix(a+a.T)


def _vectors(n=24,p=7,seed=17):
    y=np.random.default_rng(seed).normal(size=(n,p))
    return y/np.linalg.norm(y,axis=1)[:,None]


@pytest.mark.parametrize("k",[2,4,6,8,10])
def test_enumeration_exhaustive_signed_group_merging(k):
    graph=_graph(25,k)
    edges=edge_arrays(graph)
    rng=np.random.default_rng(k)
    groups=rng.integers(k,size=25)
    solutions=enumerate_group_merges(
        edges,groups,k,top=1 << (k-1),
    )
    assert len(solutions)==1 << (k-1)
    brute=[]
    for mask in range(1 << (k-1)):
        binary=np.array([
            (mask >> (int(group)-1)) & 1 if group else 0
            for group in groups
        ],dtype=np.int8)
        brute.append((cut_weight(edges,binary),mask))
    brute.sort(key=lambda x:(-x[0],x[1]))
    assert [(x[0],x[1]) for x in solutions]==brute
    for score,mask,labels in solutions:
        assert score==cut_weight(edges,labels)
        assert labels.dtype==np.int8
    assert solutions[0][0]>=0.


def test_negative_signed_graph_prefers_trivial_merge():
    graph=-np.ones((7,7),dtype=float)
    np.fill_diagonal(graph,0)
    g=sparse.csr_matrix(graph)
    groups=np.arange(7,dtype=np.int16)%4
    solutions=enumerate_group_merges(edge_arrays(g),groups,4)
    assert solutions[0][0]==0
    assert solutions[0][1]==0
    assert np.all(solutions[0][2]==0)


@pytest.mark.parametrize("seed",[2,17,42])
def test_two_centroid_antipodal_exact_original_kmeansnm(seed):
    """Seed schedule and Lloyd loop match existing rounding.py k=2."""
    graph=_graph(28,seed)
    edges=edge_arrays(graph)
    y=_vectors(28,6,seed+1)
    rng0=np.random.default_rng(np.random.SeedSequence([
        seed,11,2,4,0,
    ]))
    rng1=np.random.default_rng(np.random.SeedSequence([
        seed,11,2,4,0,
    ]))
    ref,iterations=round_spherical_kmeans(
        y,2,rng0,antipodal=True,edges=edges,
        keep_best=True,max_iter=100,
    )
    candidates,diag=optimize_multigroup_kmeans(
        y,graph,rng1,k=2,init="paired",
        max_iter=100,top_merges=1,
    )
    assert diag["lloyd_iterations"]==iterations
    assert diag["nontrivial_group_assignments_per_iteration"]==1
    assert candidates[0]["score"]==pytest.approx(
        cut_weight(edges,ref),
    )
    np.testing.assert_array_equal(
        candidates[0]["labels"],ref,
    )


@pytest.mark.parametrize("k",[4,6,8,10])
def test_multigroup_raw_candidate_scores_and_deterministic_replay(k):
    graph=_graph(28,5)
    y=_vectors(28,9,6)
    untouched=y.copy()
    def solve():
        return optimize_multigroup_kmeans(
            y,graph,np.random.default_rng(1234),
            k=k,init="paired",max_iter=15,top_merges=4,
        )
    first,info=solve()
    second,again=solve()
    np.testing.assert_array_equal(y,untouched)
    assert info["best_raw_cut"]==again["best_raw_cut"]
    assert len(first)<=4
    assert all(
        a["score"]>=b["score"]
        for a,b in zip(first,first[1:])
    )
    assert len(set(
        np.packbits(
            x["labels"] if x["labels"][0]==0
            else 1-x["labels"],
            bitorder="little",
        ).tobytes() for x in first
    ))==len(first)
    for a,b in zip(first,second):
        np.testing.assert_array_equal(a["labels"],b["labels"])
        np.testing.assert_array_equal(a["groups"],b["groups"])
        predicted=np.array([
            ((a["mask"]>>(int(c)-1))&1) if c else 0
            for c in a["groups"]
        ],dtype=np.int8)
        np.testing.assert_array_equal(a["labels"],predicted)
        assert a["score"]==cut_weight(edge_arrays(graph),a["labels"])


def test_reject_odd_antipodal_k_and_invalid_groups():
    g=_graph()
    y=_vectors()
    with pytest.raises(ValueError):
        optimize_multigroup_kmeans(
            y,g,np.random.default_rng(1),k=5,init="paired",
        )
    with pytest.raises(ValueError):
        enumerate_group_merges(
            edge_arrays(g),np.full(24,9),k=4,
        )


def _eva_source(tmp_path):
    n=12
    edges=[]
    for i in range(1,n+1):
        for j in range(i+1,n+1):
            if ((i+2*j)%3)==0:
                w=(i*j)%7-3
                if w==0:
                    w=2
                edges.append((i,j,w))
    path=tmp_path/"G11.txt"
    path.write_text(
        f"{n} {len(edges)}\n"+"\n".join(
            f"{i} {j} {w}" for i,j,w in edges
        )+"\n",encoding="utf8",
    )
    factor=tmp_path/"eva"
    args=SimpleNamespace(
        max_n=20,output_dir=factor,rank=None,
        tolerance=1e-6,max_sweeps=12,seed=2026,
        checkpoint_every=2,rounding_seed=2026,
        restarts=2,cpu_threads=1,convex_dir=None,
        push=False,push_branch=None,push_every=0,
    )
    run_one(path,args)
    return path,factor


def test_parallel_group_coarsening_all_k_archive_and_resume(tmp_path):
    path,factor=_eva_source(tmp_path)
    original=(factor/"factors"/"G11.npz")
    sha=hashlib.sha256(original.read_bytes()).hexdigest()
    dest=tmp_path/"multi-results"
    options=dict(
        factor_dir=factor,dest=dest,
        ks=(2,4,6,8,10),
        inits=("paired","independent"),
        restarts=3,chunk_size=2,workers=2,
        max_iter=7,top_merges=3,kl_passes=3,
        rounding_seed=2026,kl_seed=2026,target=1000.,
    )
    first,ranks=run_graph(path,**options)
    assert len(first)==20
    assert len(ranks)==10
    assert hashlib.sha256(original.read_bytes()).hexdigest()==sha
    assert len(list((dest/"cases").glob("G11-multi-*.json")))==20
    edges=edge_arrays(load_graph(path))
    for file in (dest/"cases").glob("G11-multi-*.json"):
        case=json.loads(file.read_text(encoding="utf8"))
        assert verify_archive(case["archive_path"])["status"]=="complete"
        path_archive=Path(case["archive_path"])
        group_file=Path(case["groups_path"])
        with np.load(group_file,allow_pickle=False) as data:
            groups=data["groups"].copy()
            masks=data["masks"].copy()
            restarts=data["restarts"].copy()
        candidate_raw,candidate_score=load_method(
            path_archive/"Exp017-CandidatesRaw_EVA.npz",
        )
        assert groups.shape[1]==len(candidate_score)
        for index,labels in enumerate(candidate_raw):
            predicted=np.array([
                ((int(masks[index])>>(int(g)-1))&1) if g else 0
                for g in groups[:,index]
            ],dtype=np.int8)
            np.testing.assert_array_equal(labels,predicted)
            assert cut_weight(edges,labels)==candidate_score[index]
        for stage in STAGES+EXTRA_STAGES:
            labels,scores=load_method(
                path_archive/f"Exp017-{stage}_EVA.npz",
            )
            for labels_one,score in zip(labels,scores):
                assert cut_weight(edges,labels_one)==score
        for trial in case["trials"]:
            s=trial["scores"]
            assert s["AfterKL"]>=s["Raw"]
            assert s["AfterKL"]>=s["BeforeKL"]
    archived_before=len(list((dest/"archives").iterdir()))
    identical,ranks_same=run_graph(path,**options)
    assert {
        record["id"]:record["best_KL"] for record in first
    }=={
        record["id"]:record["best_KL"] for record in identical
    }
    assert ranks==ranks_same
    assert len(list((dest/"archives").iterdir()))==archived_before
    assert (dest/"summary.csv").is_file()
    assert (dest/"leaderboard.json").is_file()
