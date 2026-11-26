"""Unit tests for exact geometry and signed GW controls; no EVA solve."""
import numpy as np
import pytest
from scipy import sparse
from experiment031_signed_gw_geometry import (
    rng_normals, signed_fields, edge_angles, edge_summary,
    signed_cut_batch, score_decomposition, DIRECTIONS,
)


def matrix(n, triples):
    ii=[];jj=[];vv=[]
    for i,j,w in triples:
        ii.extend([i,j]);jj.extend([j,i]);vv.extend([w,w])
    return sparse.csr_matrix((vv,(ii,jj)),shape=(n,n),dtype=float)


def test_signed_weights_favor_correct_angle_sign_and_expected_gw():
    y=np.asarray([[1.,0.],[-1.,0.],[1.,0.]])
    e=(np.asarray([0,0]),np.asarray([1,2]),np.asarray([2.,-3.]))
    dot,prob=edge_angles(y,e)
    assert np.allclose(dot,[-1,1])
    assert np.allclose(prob,[1,0])
    s=edge_summary(y,e)
    assert s['pos_angle_deg']==pytest.approx(180)
    assert s['neg_angle_deg']==pytest.approx(0)
    assert s['expected_positive_cut']==pytest.approx(2)
    assert s['expected_negative_cut']==pytest.approx(0)
    assert s['expected_signed_cut']==pytest.approx(2)
    assert s['pos_unfavorable_frac']==0
    assert s['neg_unfavorable_frac']==0
    labels=np.asarray([[1,1],[0,0],[1,0]],dtype=bool)
    assert np.allclose(signed_cut_batch(e,labels),[2,-1])
    assert score_decomposition(e,labels[:,0])==(2.,0.)
    assert score_decomposition(e,labels[:,1])==(2.,-3.)


def test_field_sign_and_unit_norm_not_affected_by_edge_sign():
    y=np.asarray([[1.,0.],[1.,0.],[0.,1.]])
    a=matrix(3,[(0,1,-2.),(0,2,3.)])
    hp,hn,h,d,degree,cos,nonzero=signed_fields(a,y)
    assert np.allclose(hp[0],[0,3]);assert np.allclose(hn[0],[-2,0])
    assert np.allclose(h[0],[-2,3]); assert d[0]==pytest.approx(np.sqrt(13))
    assert degree[0]==5
    assert nonzero.all()
    assert np.max(np.abs(cos))<=1+1e-12


def test_gw_fixed_point_h_projection_same_partition():
    # Single positive edge: EVA coordinate-fixed vectors are opposite.
    y=np.asarray([[1.,0.],[-1.,0.]])
    a=matrix(2,[(0,1,3.)])
    hp,hn,h,d,degree,cos,nz=signed_fields(a,y)
    assert np.allclose(h,-3*y)
    assert np.allclose(cos,[1,1])
    r=np.random.default_rng(44).normal(size=(2,64))
    assert np.array_equal(y@r>=0,(-h)@r>=0)
    assert np.allclose(hn,0)


def test_negative_weights_cooperate_with_shared_side():
    # Single negative edge: both EVA embeddings are aligned.
    y=np.asarray([[0.,1.],[0.,1.]])
    a=matrix(2,[(0,1,-5.)])
    hp,hn,h,d,degree,cos,nz=signed_fields(a,y)
    assert np.allclose(h,-5*y)
    assert np.allclose(cos,[1,1])
    assert np.allclose(hp,0)
    r=np.random.default_rng(65).normal(size=(2,101))
    assert np.array_equal(y@r>=0,(-h)@r>=0)


def test_zero_field_and_rng_reproducibility():
    a=matrix(3,[(0,1,1.),(0,2,1.)])
    y=np.asarray([[0.,1.],[1.,0.],[-1.,0.]])
    _,_,h,d,degree,cos,nz=signed_fields(a,y)
    assert d[0]==0 and not nz[0]
    u=rng_normals('G11','default',2032,41)
    assert u.shape==(41,DIRECTIONS)
    assert np.array_equal(u,rng_normals('G11','default',2032,41))
    assert not np.array_equal(u,rng_normals('G11','full',2032,41))


def test_random_hyperplane_angle_law_empirically():
    y=np.asarray([[1.,0.],[0.,1.]])
    r=np.random.default_rng(93).normal(size=(2,10000))
    cut=np.mean((y@r>=0)[0]!=(y@r>=0)[1])
    assert abs(cut-0.5)<0.02