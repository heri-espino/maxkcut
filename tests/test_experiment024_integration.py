"""Fast cross-family smoke on a synthetic original signed graph.

Exercising all eighteen initializers catches API mismatches before a long
self-hosted experiment. No EVA optimize/SDP/real G-set data are needed.
"""
import numpy as np
import pytest
from scipy import sparse

from maxkcut import experiment024_paired_gw16 as e
from maxkcut.io import edge_arrays


@pytest.mark.parametrize("method", e.METHODS)
def test_every_method_receives_same_gw_seed_and_preserves_incumbent(tmp_path, method):
    rng=np.random.default_rng(24)
    n,p=40,6
    y=rng.standard_normal((n,p))
    y/=np.linalg.norm(y,axis=1)[:,None]
    mat=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if rng.random()<0.18:
                mat[i,j]=1. if rng.random()<0.65 else -1.
                mat[j,i]=mat[i,j]
    graph=sparse.csr_matrix(mat)
    graph.sort_indices()
    edges=edge_arrays(graph)
    r=e.shared_directions(y,11)[0]
    gw=e.gw_partition(y,r)
    conf=e.BUDGETS["pilot"].copy()
    conf["direction_trials"]=1
    conf["line_directions"]=1
    conf["em_iters"]=1
    conf["cem_generations"]=1
    conf["gibbs_sweeps"]=1
    conf["kmeans_iters"]=4
    cand,raw,info=e.generate_candidate(
        graph,y,edges,method,r,gw,0,conf,tmp_path/"G11","synthetic-g","synthetic-f"
    )
    assert cand.shape==(n,),method
    assert set(np.unique(cand))<={0,1}
    assert abs(raw-e._score(edges,cand))<1e-8
    assert raw>=e._score(edges,gw)-1e-8
    assert info["shared_gw_direction"]
    if method in e.MST:
        assert info["mst_is_gw_anchored_hybrid"]
        assert info["independent_mst_random_draws"]==0
