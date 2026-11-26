"""Publication-only cross-platform audit of the immutable E036 computation.
Original verify() copied from computation commit 007ebcc326797e60edb791f1ed34a53f85a20505; only centroid
angle comparison changes to cosine space, avoiding arccos endpoint ill-conditioning.
The original identity(), driver hash, archive hashes and cut checks remain unchanged.
"""
import json
from pathlib import Path
import numpy as np
import experiment033_raw_hyperplane_sweep as e33
from experiment036_inertia_guided import ROOT, ARMS, STEPS, identity, geometry


def verify_geometry(actual, archived):
    # Ordinary/spherical inertia and balance retain the original tolerance.
    np.testing.assert_allclose(actual[[0,1,3]], archived[[0,1,3]], atol=1e-9, rtol=0, equal_nan=True)
    a,b=actual[2],archived[2]
    if np.isnan(a) or np.isnan(b):
        assert np.isnan(a) and np.isnan(b), "Undefined centroid angle mismatch"
    else:
        assert 0 <= a <= np.pi and 0 <= b <= np.pi, "Invalid centroid angle"
        np.testing.assert_allclose(np.cos(a),np.cos(b),atol=1e-12,rtol=0)


def verify(g,phase, graph,y,rec, baseline):
    folder=ROOT/phase/"graphs"/g
    meta=json.loads((folder/"manifest.json").read_text())
    expected=identity(g,rec)
    if any(meta.get(k)!=v for k,v in expected.items()) or meta["archive_sha256"]!=e33.sha(folder/"trajectory.npz"):
        raise ValueError("Checkpoint mismatch: "+g)
    raw=np.loadtxt(Path(rec["graph_path"].replace("\\","/")),skiprows=1,ndmin=2)
    i,j=raw[:,:2].astype(int).T-1; w=raw[:,2]
    data={}
    with np.load(folder/"trajectory.npz",allow_pickle=False) as f:
        for s in range(e33.STARTS):
            for a,arm in enumerate(ARMS):
                k=f"{s:02d}_{a}"
                d,nn,xx,ev=[f[k+"_"+z] for z in ("trace","normals","labels","events")]
                assert len(d)==len(nn)==len(xx) and 2<=len(d)<=STEPS+1
                if arm!="spherical-kmeans": assert len(d)==STEPS+1
                np.testing.assert_allclose(d[:,0],np.arange(len(d)))
                np.testing.assert_allclose(d[:,2],np.maximum.accumulate(d[:,1]),atol=1e-7,rtol=0)
                np.testing.assert_allclose(d[0,1],baseline["cuts"][s,0,0],atol=1e-7,rtol=0)
                assert np.all(np.isfinite(d[:,[1,2,3,4,5,6]])) and np.all(d[:,3]>=0)
                np.testing.assert_allclose(np.linalg.norm(nn,axis=1),1,atol=1e-9,rtol=0)
                for t in range(len(d)):
                    x=np.unpackbits(xx[t])[:len(y)]
                    q=y@nn[t]
                    assert not np.any(np.abs(q[(q>0)!=x])>1e-10)
                    np.testing.assert_allclose(w[x[i]!=x[j]].sum(),d[t,1],atol=1e-7,rtol=0)
                    verify_geometry(geometry(y,x)[0],d[t,5:9])
                if arm in ("centroid-sweep","antipodal-sweep","hybrid-sweep"):
                    assert np.all(d[1:,1]>=d[:-1,2]-1e-7)
                data[s,arm]=(d,nn,xx,ev)
    return data


def test():
    x=np.array([.021852,.021972,np.pi,.5])
    y=x.copy();y[2]=3.1415926325163688
    verify_geometry(x,y)
    for index,delta in [(0,1e-5),(2,-.01),(3,.01)]:
        z=x.copy();z[index]+=delta
        try: verify_geometry(x,z)
        except AssertionError: pass
        else: raise AssertionError("Invalid geometry accepted")
    print("Endpoint rounding regression and invalid geometry checks passed")

if __name__=="__main__":test()
