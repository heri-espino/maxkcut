"""Experiment027: real signed-cut, spherical geometry and resume invariants."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("numba")

SCRIPT = Path(__file__).resolve().parents[1]/"scripts"/"experiment027_angular.py"
spec = importlib.util.spec_from_file_location("experiment027_angular", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(n=9, p=4):
    rng=np.random.default_rng(27)
    upper=rng.integers(-3,6,size=(n,n)).astype(float)
    upper=np.triu(upper,1)
    graph=sparse.csr_matrix(upper+upper.T)
    y=rng.standard_normal((n,p))
    y/=np.linalg.norm(y,axis=1)[:,None]
    r=np.random.default_rng(41).standard_normal(p)
    r/=np.linalg.norm(r)
    return graph,y,r


def test_angular_margin_is_exact_boundary_distance():
    r=np.array([1.,0.,0.])
    y=np.array([[.5,np.sqrt(3)/2,0.],[0.,1.,0.],[1.,0.,0.]])
    margins=module.angular_margins(y,r)
    np.testing.assert_allclose(margins,[np.pi/6,0.,np.pi/2],atol=1e-12)
    t=module._tangent_to_vertex(y[0],r)
    boundary=np.cos(margins[0])*r+np.sin(margins[0])*t
    assert abs(float(y[0]@boundary))<1e-12
    assert module._tangent_to_vertex(y[2],r) is None


def test_pair_target_satisfies_both_crossing_constraints():
    r=np.array([1.,0.,0.,0.])
    v=np.array([.6,.8,0.,0.]); w=np.array([.6,0.,.8,0.])
    tangent=module._tangent_to_pair(v,w,r)
    assert tangent is not None
    a=np.stack([v,w])
    # The intended pair target is encountered along the chosen sweep plane.
    root=np.linalg.svd(a)[2][-1]
    dot=min(1.,max(-1.,float(r @ tangent)))
    assert abs(dot)<1e-12
    t=np.linalg.pinv(a @ a.T)
    projected=r-a.T@(t@(a@r))
    target=projected/np.linalg.norm(projected)
    theta=np.arccos(float(r@target))
    normal=np.cos(theta)*r+np.sin(theta)*tangent
    assert np.max(np.abs(a@normal))<1e-10


@pytest.mark.parametrize("method", module.METHODS)
def test_greedy_is_monotone_deterministic_and_keeps_eva_immutable(method):
    graph,y,r=fixture()
    saved=y.copy()
    from maxkcut.io import edge_arrays
    from maxkcut.objective import cut_weight
    initial=float(cut_weight(edge_arrays(graph),(y@r>0).astype(np.int8)))
    events=[]
    result=module.greedy_angular(graph,y,r,method,rounds=3,candidates=3,
                                 on_iteration=lambda info,labels:events.append((info,labels.copy())))
    normal,labels,info=result
    same=module.greedy_angular(graph,y,r,method,rounds=3,candidates=3)
    np.testing.assert_array_equal(y,saved)
    np.testing.assert_array_equal(normal,same[0])
    np.testing.assert_array_equal(labels,same[1])
    np.testing.assert_array_equal(labels,(y@normal>0).astype(np.int8))
    assert float(cut_weight(edge_arrays(graph),labels))==info["optimized_raw_cut"]
    assert info["optimized_raw_cut"]>=initial-1e-8
    assert info["paths_evaluated"]<=9
    assert any(event["stage"]=="AngularCandidate" for event,_ in events)
    assert len(info["accepted_normals"])==1+info["accepted_iterations"]
    assert all(b["cut"]>=a["cut"] for a,b in zip(
        info["accepted_normals"],info["accepted_normals"][1:]))


def test_saved_seed_is_independently_verified_and_skipped_on_rerun(tmp_path,monkeypatch):
    graph,y,_=fixture(n=8,p=4)
    monkeypatch.setattr(module,"_factor_and_case",lambda *a:(graph,y,None,"graph-sha","factor-sha"))
    args=SimpleNamespace(root=tmp_path/"new",baseline_root=tmp_path/"old",
                         data_dir=tmp_path,factor_dir=tmp_path,
                         rounds=1,candidates=2,seed=0)
    initial=module.run_one("G1","angular-margin-greedy","pilot",args)
    assert initial["starts_completed"]==1
    folder=args.root/"pilot"/"G1"/"angular-margin-greedy"
    assert (folder/"0000.npz").is_file()
    journal=folder/"progress"/"0000"
    before=(journal/"events.jsonl").read_bytes()
    normalhash=module.sha(journal/"accepted_normals.npz")
    previous=module.run_one("G1","angular-margin-greedy","pilot",args)
    assert previous["starts_completed"]==1
    assert before==(journal/"events.jsonl").read_bytes()
    assert normalhash==module.sha(journal/"accepted_normals.npz")
    assert json.loads((folder/"0000.json").read_text())["raw_cut"]<=initial["best_final"]


def test_partial_journal_is_not_overwritten(tmp_path,monkeypatch):
    graph,y,_=fixture(n=8,p=4)
    monkeypatch.setattr(module,"_factor_and_case",lambda *a:(graph,y,None,"g","f"))
    args=SimpleNamespace(root=tmp_path/"new",baseline_root=tmp_path/"old",
                         data_dir=tmp_path,factor_dir=tmp_path,
                         rounds=1,candidates=2,seed=0)
    folder=args.root/"pilot"/"G1"/"angular-gain-greedy"/"progress"/"0000"
    folder.mkdir(parents=True)
    (folder/"partial.txt").write_text("retain")
    with pytest.raises(RuntimeError,match="Incomplete seed journal"):
        module.run_one("G1","angular-gain-greedy","pilot",args)
    assert (folder/"partial.txt").read_text()=="retain"
