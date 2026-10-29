"""Experiment018: diversified KMeansNM initialization tests."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("numba")

from maxkcut.eva_cli import run_one
from maxkcut.eva_kmeans_initializers import (
    INIT2,centers_for_k2,
)
from maxkcut.eva_multigroup_kmeans import optimize_multigroup_kmeans
from maxkcut.eva_multigroup_kmeans_cli import run_graph
from maxkcut.io import edge_arrays
from maxkcut.objective import cut_weight
from maxkcut.rounding import round_spherical_kmeans


def _inputs(n=30,d=8,seed=4):
    rng=np.random.default_rng(seed)
    vectors=rng.standard_normal((n,d))
    vectors/=np.linalg.norm(vectors,axis=1)[:,None]
    data=rng.integers(-5,7,size=(n,n)).astype(float)
    data[np.tril_indices(n)]=0
    graph=sparse.csr_matrix(data+data.T)
    return vectors,graph


@pytest.mark.parametrize("strategy",INIT2)
def test_each_initialization_is_reproducible_finite_and_unit(strategy):
    y,graph=_inputs()
    unchanged=y.copy()
    edges=edge_arrays(graph)
    run=lambda:centers_for_k2(
        y,np.random.default_rng(2026),
        init=strategy,edges=edges,
    )
    a,meta=run()
    b,_=run()
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(y,unchanged)
    assert a.shape==(2,y.shape[1])
    assert np.all(np.isfinite(a))
    np.testing.assert_allclose(np.linalg.norm(a,axis=1),1.,atol=1e-12)
    assert meta["screened_hard_cuts"]==(16 if strategy=="gw-screen16" else 0)
    if strategy.endswith("antipodal") or strategy=="paired":
        np.testing.assert_allclose(a[0],-a[1],atol=1e-12)
    if strategy=="mean-perturb":
        assert np.dot(a[0],a[1])>0.
    if strategy=="farthest-pair":
        indices=meta["sample_vertices"]
        cosine=y@y[indices[0]]
        cosine[indices[0]]=np.inf
        assert indices[1]==np.argmin(cosine)


def test_exact_same_seed_legacy_kmeansnm_control():
    y,graph=_inputs(27,7,seed=9)
    edges=edge_arrays(graph)
    rng0=np.random.default_rng(np.random.SeedSequence([2026,63,2,3,0]))
    rng1=np.random.default_rng(np.random.SeedSequence([2026,63,2,3,0]))
    expected,iterations=round_spherical_kmeans(
        y,2,rng0,max_iter=100,antipodal=True,
        edges=edges,keep_best=True,
    )
    result,diag=optimize_multigroup_kmeans(
        y,graph,rng1,k=2,init="paired",
        max_iter=100,top_merges=1,
    )
    np.testing.assert_array_equal(result[0]["labels"],expected)
    assert result[0]["score"]==cut_weight(edges,expected)
    assert diag["lloyd_iterations"]==iterations


def test_nine_methods_preserve_binary_signed_cut_and_provenance():
    y,graph=_inputs(35,8,seed=20)
    edges=edge_arrays(graph)
    seen={}
    for strategy in INIT2:
        results,diag=optimize_multigroup_kmeans(
            y,graph,np.random.default_rng(1234),
            k=2,init=strategy,max_iter=6,top_merges=2,
        )
        assert diag["init"]==strategy
        assert diag["initialization_details"]["screened_hard_cuts"]==(
            16 if strategy=="gw-screen16" else 0
        )
        for row in results:
            assert row["score"]==cut_weight(edges,row["labels"])
            assert np.isin(row["labels"],[0,1]).all()
        seen[strategy]=results[0]["score"]
    assert len(seen)==len(INIT2)


def _saved_eva(tmp_path):
    n=12
    tuples=[]
    for i in range(1,n+1):
        for j in range(i+1,n+1):
            if (i+j)%3==0:
                w=(i*j)%7-3
                tuples.append((i,j,w if w else 1))
    path=tmp_path/"G11.txt"
    path.write_text(
        f"{n} {len(tuples)}\n"+
        "\n".join(f"{i} {j} {w}" for i,j,w in tuples)+"\n",
        encoding="utf8",
    )
    factor_dir=tmp_path/"eva"
    args=SimpleNamespace(
        max_n=20,output_dir=factor_dir,rank=None,
        tolerance=1e-6,max_sweeps=12,seed=2026,
        checkpoint_every=2,rounding_seed=2026,
        restarts=2,cpu_threads=1,convex_dir=None,
        push=False,push_branch=None,push_every=0,
    )
    run_one(path,args)
    return path,factor_dir


def test_self_hosted_parallel_init_grid_verified_resume(tmp_path):
    path,factor_dir=_saved_eva(tmp_path)
    source=factor_dir/"factors"/"G11.npz"
    original_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    output=tmp_path/"exp018"
    args=dict(
        factor_dir=factor_dir,dest=output,
        ks=(2,),inits=INIT2,
        restarts=2,chunk_size=1,workers=2,
        max_iter=8,top_merges=2,
        kl_passes=3,rounding_seed=2026,
        kl_seed=2026,target=1000.,
    )
    cases,ranks=run_graph(path,**args)
    assert len(cases)==2*len(INIT2)
    assert len(ranks)==len(INIT2)
    assert len(list((output/"cases").glob("G11-multi-*.json")))==len(cases)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original_sha
    cases2,ranks2=run_graph(path,**args)
    assert {x["id"]:x["best_KL"] for x in cases}=={
        x["id"]:x["best_KL"] for x in cases2
    }
    assert ranks==ranks2
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original_sha
    assert (output/"summary.csv").is_file()
    assert (output/"leaderboard.json").is_file()


def test_invalid_high_k_modes_are_rejected():
    y,g=_inputs(20,6)
    with pytest.raises(ValueError):
        optimize_multigroup_kmeans(
            y,g,np.random.default_rng(0),
            k=4,init="pca-antipodal",
        )
    with pytest.raises(ValueError):
        centers_for_k2(
            y,np.random.default_rng(0),
            init="gw-screen16",edges=None,
        )
