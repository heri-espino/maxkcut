"""Experiment024 invariants: shared GW directions, meaningful seeds, retention.

Tests need no EVA solver and do not modify historical experimental files.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from maxkcut import experiment024_paired_gw16 as e
from maxkcut.io import edge_arrays


def tiny_problem():
    y=np.asarray([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],dtype=float)
    graph=sparse.csr_matrix(np.asarray([
        [0.,1.,-1.,2.],[1.,0.,3.,0.],[-1.,3.,0.,2.],[2.,0.,2.,0.]
    ]))
    return graph,y,edge_arrays(graph)


def test_exactly_16_shared_unit_normals():
    g,y,edges=tiny_problem()
    a=e.shared_directions(y,11)
    b=e.shared_directions(y,11)
    c=e.shared_directions(y,12)
    assert a.shape==(16,2)
    assert np.array_equal(a,b)
    assert not np.array_equal(a,c)
    np.testing.assert_allclose(np.linalg.norm(a,axis=1),1.,atol=1e-14)
    assert len({a[i].tobytes() for i in range(16)})==16
    assert all(e.gw_partition(y,z).shape==(4,) for z in a)
    with pytest.raises(ValueError,match="16"):
        e.shared_directions(y,11,starts=15)


def test_unit_direction_archive_identifies_factors(tmp_path):
    graph,y,edges=tiny_problem()
    normals,parts,meta=e.directions_archive(tmp_path/"G11",y,"G11","graph-a","factor-a")
    again=e.directions_archive(tmp_path/"G11",y,"G11","graph-a","factor-a")
    assert np.array_equal(normals,again[0])
    assert np.array_equal(parts,again[1])
    with pytest.raises(ValueError,match="changed"):
        e.directions_archive(tmp_path/"G11",y,"G11","graph-a","factor-b")


def test_gw_seeded_kmeans_never_loses_its_initial_cut():
    graph,y,edges=tiny_problem()
    for r in e.shared_directions(y,11):
        gw=e.gw_partition(y,r)
        best,details=e.seeded_lloyd(y,edges,r,max_iter=8)
        assert e._score(edges,best)>=e._score(edges,gw)-1e-8
        assert details["retained_original_gw"]


def test_gw_and_threshold_share_exact_direction(tmp_path):
    graph,y,edges=tiny_problem()
    r=e.shared_directions(y,11)[0]
    gw=e.gw_partition(y,r)
    cfg=e.BUDGETS["pilot"]
    for method in ("gw-rr","gw-threshold","control-paired"):
        p,score,info=e.generate_candidate(
            graph,y,edges,method,r,gw,0,cfg,tmp_path/"G11","hash-g","hash-f")
        assert p.shape==(4,)
        assert np.isin(p,[0,1]).all()
        assert score>=e._score(edges,gw)-1e-8
        assert info["shared_gw_direction"] is True


def test_seeded_group_and_cutaware_receive_shared_vector(monkeypatch,tmp_path):
    graph,y,edges=tiny_problem()
    r=e.shared_directions(y,11)[0]
    gw=e.gw_partition(y,r)
    received={}
    def fake_cutaware(*args,**kwargs):
        received["cutaware"]=kwargs["initial_direction"].copy()
        return gw.copy(),{}
    def fake_groups(*args,**kwargs):
        received["group"]=kwargs["initial_direction"].copy()
        return [{"labels":gw.copy()}],{}
    monkeypatch.setattr(e,"improve_kmeans_direction",fake_cutaware)
    monkeypatch.setattr(e,"optimize_multigroup_kmeans",fake_groups)
    for method in ("cutaware-kmeans","multigroup-k4"):
        p,score,info=e.generate_candidate(
            graph,y,edges,method,r,gw,0,e.BUDGETS["pilot"],
            tmp_path/"G11","graph","factor")
        assert np.array_equal(p,gw)
    np.testing.assert_array_equal(received["cutaware"],r)
    np.testing.assert_array_equal(received["group"],r)


def test_method_list_and_expected_coverage():
    assert len(e.GRAPH_IDS)==71
    assert len(e.METHODS)==18
    assert e.METHODS[0]=="gw-rr"
    assert e.BUDGETS["pilot"]["starts"]==16
    assert e.BUDGETS["full"]["starts"]==16
    assert all(method in e.METHODS for method in e.MST)
