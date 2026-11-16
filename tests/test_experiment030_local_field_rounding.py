"""Synthetic, deterministic checks for Experiment030; never access G-set LFS."""
import numpy as np
from scipy import sparse
from maxkcut.rounding import round_spherical_kmeans
from maxkcut.objective import cut_weight
from experiment030_local_field_rounding import derive_fields, cluster_with_trace, METHODS
from experiment029_kmeansnm_100seeds import rng_for


def test_unit_embedding_field_can_have_arbitrary_norm_and_zero():
    y=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[1.,0.]])
    w=np.zeros((5,5));w[0,1]=w[1,0]=3.;w[0,2]=w[2,0]=-0.2
    a=sparse.csr_matrix(w)
    representations,diag=derive_fields(a,y)
    assert np.allclose(np.linalg.norm(y,axis=1),1)
    assert diag['max_field_norm']>1 and diag['min_field_norm']==0
    assert diag['zero_field_count']==2
    for method in METHODS:
        assert representations[method].shape==y.shape
        assert np.all(np.isfinite(representations[method]))
    assert np.all(np.linalg.norm(representations['unit_minus_h'],axis=1)>0.999999)
    assert np.allclose(representations['coherence_y'][3],y[3])  # isolated fallback
    assert np.allclose(representations['abs_degree_y'][3],y[3]*1e-12)


def test_exact_fixed_point_h_is_negative_y_scaled():
    signs=np.array([1.,-1.,1.,-1.])
    y=np.column_stack([signs,np.zeros(4)])
    w=np.zeros((4,4))
    for i in range(4):
        for j in range(i+1,4):
            w[i,j]=w[j,i]=-signs[i]*signs[j]*(1+0.2*i+0.1*j)
    a=sparse.csr_matrix(w)
    v,d=derive_fields(a,y)
    assert np.allclose(v['unit_minus_h'],y,atol=1e-13)
    assert np.allclose(v['raw_minus_h'],v['field_strength_y'],atol=1e-13)
    assert d['min_cosine_to_minus_h']>0.99999999999
    assert np.allclose(v['coherence_y'],y,atol=1e-12)


def test_traced_clustering_exactly_matches_frozen_rounder():
    n=15
    rng=np.random.default_rng(130)
    y=rng.normal(size=(n,5))
    y/=np.linalg.norm(y,axis=1)[:,None]
    a=sparse.random(n,n,density=.4,random_state=3,format='csr')
    a=a+a.T
    a.data=np.where(a.data>.5,a.data,-a.data)
    edge_coo=sparse.triu(a,1).tocoo()
    edges=(edge_coo.row,edge_coo.col,edge_coo.data)
    reps,_=derive_fields(a,y)
    for method in METHODS:
        vectors=reps[method]
        for seed in (2026,2033,2050):
            labels,count,history=cluster_with_trace(vectors,edges,seed,'G1')
            original,its=round_spherical_kmeans(vectors,2,rng_for(seed,'G1'),
                antipodal=True,edges=edges,keep_best=True)
            assert np.array_equal(labels,original)
            assert count==its==len(history)
            assert abs(history[-1]['best_so_far']-cut_weight(edges,labels))<1e-9
            assert all(z['best_so_far']>=history[k-1]['best_so_far'] for k,z in enumerate(history) if k)


def test_positive_row_scaling_does_not_change_first_antipodal_assignment():
    rng=np.random.default_rng(2)
    y=rng.normal(size=(12,6));y/=np.linalg.norm(y,axis=1)[:,None]
    weights=np.exp(rng.normal(size=12))
    direction=np.random.default_rng(2026).normal(size=6)
    lhs=(y@direction>=0)
    rhs=((weights[:,None]*y)@direction>=0)
    assert np.array_equal(lhs,rhs)