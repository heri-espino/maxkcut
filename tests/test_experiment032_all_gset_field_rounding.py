"""Numerical and source-rounder regression checks for Experiment032."""
import numpy as np
from scipy import sparse
from maxkcut.io import edge_arrays
from maxkcut.rounding import round_spherical_kmeans, round_fj
from maxkcut.objective import cut_weight
from experiment032_all_gset_field_rounding import (
    SEED, METHODS, signed_cut, normals, cluster_trace,
    prepare_representations, pack, bootstrap_mean,
)


def example_graph(n=29, p=5):
    rng=np.random.default_rng(42)
    y=rng.normal(size=(n,p));y/=np.linalg.norm(y,axis=1)[:,None]
    i,j=np.triu_indices(n,1)
    w=rng.choice([-3.,-1.,1.,2.],size=len(i))
    a=sparse.coo_matrix((w,(i,j)),shape=(n,n)).tocsr();a=a+a.T
    return y,a,edge_arrays(a)


def test_random_gw_direction_matches_source_rounder_and_kmeans_start():
    y,a,edges=example_graph()
    for restart in (0,1,4,14):
        ctr,_=normals(31,restart,y.shape[1])
        rng=np.random.default_rng(np.random.SeedSequence([SEED,31,2,restart,0]))
        official=round_fj(y,2,rng)
        expected=(y@ctr[0]>=0).astype(np.uint8)
        assert np.array_equal(official,expected)
        initial_labels=np.argmax(y@ctr.T,axis=1)
        assert np.array_equal(initial_labels,1-expected)


def test_source_kmeans_rounder_exact_trajectory_and_best_cut():
    y,a,edges=example_graph()
    reps,_=prepare_representations(a,y)
    for restart in (0,1,2,7,19):
        ctr,_=normals(31,restart,y.shape[1])
        for v in reps.values():
            _,after_init_rng=normals(31,restart,y.shape[1])
            labels,steps,history=cluster_trace(v,edges,ctr,after_init_rng)
            source_rng=np.random.default_rng(np.random.SeedSequence([SEED,31,2,restart,0]))
            official,official_steps=round_spherical_kmeans(v,2,source_rng,
                antipodal=True,keep_best=True,edges=edges)
            assert np.array_equal(labels,official)
            assert steps==official_steps==len(history)
            assert all(history[j]['best_so_far']<=history[j+1]['best_so_far']
                       for j in range(len(history)-1))
            assert np.isclose(signed_cut(edges,labels),history[-1]['best_so_far'])
            assert signed_cut(edges,labels)>=history[0]['cut_weight']


def test_signed_edges_make_field_reverse_when_bond_negative():
    # For a negative-only two-node graph, EVA aligns y1 and y2.
    y=np.array([[1.,0.],[1.,0.]])
    a=sparse.csr_matrix(np.array([[0.,-4.],[-4.,0.]]))
    reps,info=prepare_representations(a,y)
    assert info['zero_fields']==0
    assert np.allclose(reps['minusH'],4.*y)
    assert np.isclose(info['field_alignment_min_cos'],1.)
    ctr,_=normals(1,0,2)
    assert np.array_equal(reps['Y']@ctr[0]>=0,reps['minusH']@ctr[0]>=0)


def test_positive_bond_prefers_antipodal_y():
    y=np.array([[1.,0.],[-1.,0.]])
    a=sparse.csr_matrix(np.array([[0.,5.],[5.,0.]]))
    reps,info=prepare_representations(a,y)
    assert np.allclose(reps['minusH'],5*y)
    assert info['zero_fields']==0


def test_zero_fields_fallback_and_signed_cut_rescore():
    y=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    a=sparse.csr_matrix(np.array([[0.,1.,0.,1.],[1.,0.,0.,0.],[0.,0.,0.,0.],[1.,0.,0.,0.]]))
    reps,info=prepare_representations(a,y)
    assert info['zero_fields']==2  # first row neighbors cancel, third isolated
    assert np.array_equal(reps['minusH'][0],y[0])
    assert np.array_equal(reps['minusH'][2],y[2])
    edges=edge_arrays(a)
    lab=np.array([0,1,0,0],dtype=np.uint8)
    assert signed_cut(edges,lab)==float(cut_weight(edges,lab))
    assert np.array_equal(np.unpackbits(pack(lab))[:len(lab)],lab)


def test_bootstrap_and_number_of_arms():
    assert len(METHODS)==4
    d=np.array([-1.,-1.,2.,2.])
    assert bootstrap_mean(d,'G11','GW_minusH',500)==bootstrap_mean(d,'G11','GW_minusH',500)
    ci=bootstrap_mean(d,'G11','KM_minusH',500)
    assert ci[0]<=np.mean(d)<=ci[1]