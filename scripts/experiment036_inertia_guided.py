"""E036: fixed-factor geometry-guided rounding. No SDP solves or label refinement."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
import experiment033_raw_hyperplane_sweep as e33

ROOT = Path("results/gset/experiment036-inertia-guided-sweep")
ARMS = ("spherical-kmeans", "centroid-sweep", "antipodal-sweep", "inertia-sweep", "hybrid-sweep")
STEPS = 100
EPS = 1e-8

def unit(v):
    norm = np.linalg.norm(v)
    return v/norm if norm > 1e-13 else None

def geometry(y, x):
    a = y[x==1].sum(axis=0)
    b = y[x==0].sum(axis=0)
    na, nb = np.count_nonzero(x), len(y)-np.count_nonzero(x)
    aa, bb = a@a, b@b
    eu = np.einsum("ij,ij->",y,y) - (aa/na if na else 0) - (bb/nb if nb else 0)
    sph = np.einsum("ij,ij->",y,y)+len(y)-2*(np.sqrt(aa)+np.sqrt(bb))
    angle = np.arccos(np.clip(a@b/np.sqrt(aa*bb),-1,1)) if aa*bb>0 else np.nan
    return np.array([eu/len(y),sph/len(y),angle,min(na,nb)/len(y)]),a,b

def direction(a,b,mode):
    if mode=="antipodal":
        return unit(a-b)
    ca,cb=unit(a),unit(b)
    return unit(ca-cb) if ca is not None and cb is not None else None

def tangent(r,q,z):
    v = None if q is None else unit(q-(q@r)*r)
    if v is not None:
        return v,False
    v=unit(z-(z@r)*r)
    if v is None:
        z=np.eye(len(r))[np.argmin(np.abs(r))]
        v=unit(z-(z@r)*r)
    return v,True

def arc_geometry(y,r,v, edges=None):
    """Enumerate open event intervals, with identical angles processed together.
    Return interval representative angles, spherical/Euclidean inertia, and
    optionally signed cuts using an independent difference-array edge update.
    """
    v=unit(v-(v@r)*r)
    a,b=y@r,y@v
    x=(b<0).astype(np.uint8)
    fixed=b==0
    x[fixed]=(a[fixed]>0)
    active=np.flatnonzero(~fixed)
    roots=np.arctan(-a[active]/b[active])
    order=np.argsort(roots,kind="stable")
    ids,roots=active[order],roots[order]
    ends=np.r_[0,np.flatnonzero(np.diff(roots)!=0)+1,len(ids)] if len(ids) else np.array([0])
    bounds=np.r_[-np.pi/2, roots[ends[:-1]], np.pi/2] if len(ids) else np.array([-np.pi/2,np.pi/2])
    angles=(bounds[:-1]+bounds[1:])/2
    # Float64 accumulation; no n-by-n Gram matrix.
    sign=1-2*x[ids].astype(float)
    s0=y[x==1].sum(axis=0)
    sums=np.vstack([s0,s0+np.cumsum(y[ids]*sign[:,None],axis=0)])[ends]
    sizes=(np.r_[x.sum(dtype=np.int64),x.sum(dtype=np.int64)+np.cumsum(sign)])[ends]
    other=y.sum(axis=0)-sums
    aa=np.einsum("ij,ij->i",sums,sums)
    bb=np.einsum("ij,ij->i",other,other)
    total=np.einsum("ij,ij->",y,y)
    eu=total-aa/np.maximum(sizes,1)-bb/np.maximum(len(y)-sizes,1)
    sph=total+len(y)-2*(np.sqrt(aa)+np.sqrt(bb))
    cuts=None
    if edges is not None:
        # Independent signed cut for each group. An edge changes its crossing
        # status at the earlier endpoint event, then changes back at the later.
        i,j,w=edges
        event=np.full(len(y),len(angles),dtype=int)
        event[ids]=np.searchsorted(ends[1:],np.arange(len(ids)),side="right")+1
        lo,hi=np.minimum(event[i],event[j]),np.maximum(event[i],event[j])
        delta=w*(1-2*(x[i]!=x[j]).astype(float))
        diff=np.bincount(lo,weights=delta,minlength=len(angles)+1)-np.bincount(hi,weights=delta,minlength=len(angles)+1)
        cuts=w[x[i]!=x[j]].sum()+np.cumsum(diff)[:len(angles)]
    return angles,eu/len(y),sph/len(y),cuts

def run_arm(graph,y,r,zs,arm, raw_edges=None, diagnostic=False):
    edges=e33.edge_arrays(graph)
    r=r.copy(); x=(y@r>0).astype(np.uint8)
    c=e33.score(edges,x); best=c
    rows=[]; normals=[]; labels=[]; events=[]
    prev_failed=False
    for t in range(len(zs)+1):
        state_x=x.copy(); state_r=r.copy()
        geom,a,b=geometry(y,x)
        clock=time.perf_counter()
        source="initial"; fallback=False; valid=True; intervals=1; theta=0.
        rr,xx=r.copy(),x.copy()
        if t:
            mode="antipodal" if arm=="antipodal-sweep" else "centroid"
            if arm=="hybrid-sweep":
                mode=("centroid","antipodal","random")[(t-1)%3]
            q=direction(a,b,mode) if mode!="random" else None
            if arm=="spherical-kmeans":
                source="centroid"
                if q is not None:
                    rr=q;xx=(y@rr>0).astype(np.uint8)
            else:
                # Deterministic guided-only searches would repeat an unchanged
                # arc after rejection; use the paired Gaussian on the next draw.
                source="random" if mode=="random" or prev_failed else mode
                v,fallback=tangent(r,None if source=="random" else q,zs[t-1])
                if source=="random": fallback=False
                if arm=="inertia-sweep":
                    angles,eu,sph,cut_profile=arc_geometry(y,r,v,raw_edges if diagnostic and t<=5 else None)
                    intervals=len(angles); k=int(np.argmin(sph));theta=float(angles[k])
                    rr=unit(np.cos(theta)*r+np.sin(theta)*v);xx=(y@rr>0).astype(np.uint8)
                    actual=geometry(y,xx)[0][1]
                    valid=abs(actual-sph[k])<1e-8
                    if not valid or actual>geom[1]+EPS:
                        rr,xx=r.copy(),x.copy();theta=0.
                else:
                    rr,xx,info=e33.optimize_semicircle(graph,y,r,v)
                    xx=xx.astype(np.uint8)
                    intervals=info["distinct_events"]+1;theta=info["theta"];valid=info["numerically_valid"]
                if diagnostic and t<=5:
                    angles,eu,sph,cut_profile=arc_geometry(y,r,v,raw_edges)
                    events.append(np.column_stack([np.full(len(angles),t),angles,eu,sph,cut_profile]))
        cc=e33.score(edges,xx)
        elapsed=time.perf_counter()-clock if t else 0.
        gg=geometry(y,xx)[0]
        if arm in ("inertia-sweep","spherical-kmeans"):
            accepted=gg[1]<geom[1]-EPS
        else:
            accepted=cc>c+EPS
        if accepted:
            r,x,c=rr.copy(),xx.copy(),cc
        prev_failed=not accepted
        oldbest=best;best=max(best,cc)
        turnover=np.mean(xx!=state_x)
        rows.append([t,cc,best,elapsed,intervals,gg[0],gg[1],gg[2],gg[3],
                     min(turnover,1-turnover),np.arccos(np.clip(abs(rr@state_r),0,1)),
                     theta,float(accepted),float(best>oldbest+EPS),float(valid),
                     {"initial":0,"centroid":1,"antipodal":2,"random":3}[source],float(fallback)])
        normals.append(rr.copy()); labels.append(np.packbits(xx))
        # Lloyd has converged; no fake repeated sweeps or timing.
        if t and arm=="spherical-kmeans" and not accepted:
            break
    return np.array(rows),np.array(normals),np.array(labels),np.concatenate(events) if events else np.empty((0,5))

COLUMNS=("iteration","cut","best","seconds","intervals","euclidean_inertia","spherical_inertia",
         "centroid_angle","balance","turnover","projective_angle","theta","state_accepted",
         "cut_record","valid","direction_code","fallback")
def identity(g,rec):
    return dict(graph=g,graph_sha256=rec["graph_sha256"],factor_sha256=rec["factor_sha256"],
                steps=STEPS,starts=e33.STARTS,arms=list(ARMS),driver=e33.source_sha(__file__),
                sweep=e33.source_sha("src/maxkcut/eva_direct_rounding.py"))

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
                    np.testing.assert_allclose(geometry(y,x)[0],d[t,5:9],atol=1e-9,rtol=0,equal_nan=True)
                if arm in ("centroid-sweep","antipodal-sweep","hybrid-sweep"):
                    assert np.all(d[1:,1]>=d[:-1,2]-1e-7)
                data[s,arm]=(d,nn,xx,ev)
    return data

def compute(job):
    g,phase=job
    with threadpool_limits(limits=1):
        graph,y,rec=e33.inputs(g)
        baseline,_=e33.verify(g,"full",e33.STARTS,e33.STEPS)
        folder=ROOT/phase/"graphs"/g
        if not (folder/"manifest.json").exists():
            e33.warmup()
            raw=np.loadtxt(Path(rec["graph_path"].replace("\\","/")),skiprows=1,ndmin=2)
            i,j=raw[:,:2].astype(int).T-1; w=raw[:,2]
            archive={}
            for s in range(e33.STARTS):
                start=baseline["normals"][s,0,0]
                zs=np.random.default_rng(np.random.SeedSequence([e33.SEED,33,int(g[1:]),s])).standard_normal((STEPS,y.shape[1]))
                for a,arm in enumerate(ARMS):
                    data=run_arm(graph,y,start,zs,arm,(i,j,w),g in e33.PILOT and s==0)
                    for key,val in zip(("trace","normals","labels","events"),data):
                        archive[f"{s:02d}_{a}_{key}"]=val
            folder.mkdir(parents=True,exist_ok=True)
            tmp=folder/"trajectory.tmp.npz"
            np.savez_compressed(tmp,**archive)
            os.replace(tmp,folder/"trajectory.npz")
            meta=identity(g,rec);meta["archive_sha256"]=e33.sha(folder/"trajectory.npz")
            tmp=folder/"manifest.tmp";tmp.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf8",newline="\n")
            os.replace(tmp,folder/"manifest.json")
        verify(g,phase,graph,y,rec,baseline)
        return g

def test():
    from scipy import sparse
    rng=np.random.default_rng(36)
    y=rng.normal(size=(17,5)); y/=np.linalg.norm(y,axis=1)[:,None]
    a=np.triu(rng.integers(-3,6,size=(17,17)),1);graph=sparse.csr_matrix(a+a.T,dtype=float)
    i,j=np.triu_indices(len(y),1);w=a[i,j]
    r=e33.unit(rng.normal(size=5)); v,_=tangent(r,None,rng.normal(size=5))
    angles,eu,sph,cuts=arc_geometry(y,r,v,(i,j,w))
    for k,t in enumerate(angles):
        x=(y@(np.cos(t)*r+np.sin(t)*v)>0).astype(np.uint8)
        np.testing.assert_allclose(geometry(y,x)[0][:2],[eu[k],sph[k]],atol=1e-10)
        np.testing.assert_allclose(w[x[i]!=x[j]].sum(),cuts[k],atol=1e-8)
    e33.warmup()
    for arm in ARMS:
        d,nn,xx,ev=run_arm(graph,y,r,rng.normal(size=(6,5)),arm,(i,j,w),True)
        assert np.all(np.diff(d[:,2])>=0)
        for k,x in enumerate(xx):
            x=np.unpackbits(x)[:len(y)]
            np.testing.assert_allclose(w[x[i]!=x[j]].sum(),d[k,1],atol=1e-7)
    print("Synthetic signed graph and exact event geometry checks passed")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--action",choices=["test","compute"],default="compute")
    ap.add_argument("--phase",choices=["pilot","full"],default="full")
    ap.add_argument("--workers",type=int,default=8)
    args=ap.parse_args()
    if args.action=="test": test();return
    graphs=e33.PILOT if args.phase=="pilot" else e33.GRAPHS
    # Pilot checkpoints are written to full so the full run reuses them.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(compute,(g,"full")) for g in graphs]):
            print("Verified",f.result(),flush=True)
if __name__=="__main__":main()
