"""Experiment037: paired best-known normalized GW, exact random sweep, original KMeansNM.
Frozen EVA, independent graph jobs, signed-edge audit, durable VM archive.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
import experiment033_raw_hyperplane_sweep as e33
from maxkcut.rounding import round_spherical_kmeans

GRAPHS=e33.GRAPHS
METHODS=("gw","random-sweep","kmeansnm")
STARTS,STEPS,SEED=30,100,2026
REFERENCE=Path("documents/presentations/data/experiment025_gset_reference_snapshot_2026-11-05.csv")
CERTIFIED={"G11":564,"G48":6000,"G49":6000,"G57":3494,"G62":4872,"G65":5562,"G72":7008,"G77":9940,"G81":14060}
METRICS=("euclidean_inertia","spherical_inertia","centroid_angle","balance","turnover","projective_angle")
def dump(p,obj):
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf8",newline="\n")
def sha(p):return e33.sha(p)
def references():
    with REFERENCE.open(newline="",encoding="utf8") as f:
        rows=list(csv.DictReader(f))
    assert {x["graph"] for x in rows}==set(GRAPHS) and len(rows)==71
    ref={}
    for row in rows:
        g=row["graph"];b=float(row["reference_cut"])
        if g in CERTIFIED:
            assert g=="G62" or b==CERTIFIED[g]
            b=float(CERTIFIED[g])
        assert b>0
        ref[g]=dict(graph=g,reference=b,family=row["family"],basis=("Glass-Feder 2026 Table 1; certified for canonical graph" if g in CERTIFIED else row["reference_basis"]),certified=g in CERTIFIED)
    return ref
def rng_for(g,s):
    return np.random.default_rng(np.random.SeedSequence([SEED,37,int(g[1:]),s]))
def km_trace(y,edges,rng,steps):
    # Exact source rounder: antipodal initialization, normalized independent
    # centers, assignment-stability stop, best original signed cut retained.
    centers=rng.standard_normal((2,y.shape[1]))
    centers[1]=-centers[0];centers/=np.linalg.norm(centers,axis=1)[:,None]
    cs=[];ls=[];ct=[];ts=[];previous=None
    for t in range(steps): # historical max_iter=100 assignments, t=0 initial
        clock=time.perf_counter()
        x=np.argmax(y@centers.T,axis=1).astype(np.uint8)
        c=e33.score(edges,x)
        cs.append(c);ls.append(np.packbits(x));ct.append(centers.copy())
        stable=previous is not None and np.array_equal(x,previous)
        if not stable:
            previous=x.copy()
            for k in range(2):
                members=y[x==k]
                raw=members.sum(axis=0); norm=np.linalg.norm(raw)
                centers[k]=raw/norm if len(members) and norm>1e-12 else y[rng.integers(len(y))]
        ts.append(time.perf_counter()-clock)
        if stable:break
    return np.array(cs),np.array(ls),np.array(ct),np.array(ts)
def run_start(g,s,graph,y,steps=STEPS):
    # Use the second antipodal center as GW normal so the initial label
    # convention matches argmax (ties go to class zero).
    rng=rng_for(g,s);initial=rng.standard_normal((2,y.shape[1]))
    r=-e33.unit(initial[0])
    proposal=np.random.default_rng(np.random.SeedSequence([SEED,37,int(g[1:]),s,1])).standard_normal((steps,y.shape[1]))
    pair=e33.trial(graph,y,r,proposal)
    edges=e33.edge_arrays(graph)
    kc,kl,kcent,kt=km_trace(y,edges,rng_for(g,s),steps)
    n=len(kc);assert abs(kc[0]-pair["cuts"][0,0])<=1e-7
    if s<2:
        ref,it=round_spherical_kmeans(y,2,rng_for(g,s),max_iter=steps,antipodal=True,edges=edges,keep_best=True)
        assert it==n and np.array_equal(ref,np.unpackbits(kl[np.argmax(kc)])[:len(y)])
    pad=lambda a:np.concatenate((a,np.repeat(a[-1:],steps+1-n,axis=0)),axis=0)
    cuts=np.vstack((pair["cuts"],pad(kc)))
    labels=np.concatenate((pair["labels"],pad(kl)[None]),axis=0)
    times=np.vstack((pair["seconds"],np.r_[kt,np.zeros(steps+1-n)]))
    intervals=np.vstack((pair["intervals"],np.r_[np.ones(n,dtype=int),np.zeros(steps+1-n,dtype=int)]))
    active=np.ones((3,steps+1),bool);active[2,n:]=False
    valid=np.ones_like(active);valid[1]=pair["valid"]
    diff=kcent[:,1]-kcent[:,0];norm=np.linalg.norm(diff,axis=1)
    kn=np.divide(diff,norm[:,None],out=np.zeros_like(diff),where=norm[:,None]>1e-13)
    normals=np.concatenate((pair["normals"],pad(kn)[None]),axis=0)
    return dict(cuts=cuts,best=np.maximum.accumulate(cuts,axis=1),labels=labels,seconds=times,intervals=intervals,active=active,valid=valid,normals=normals,km_centers=pad(kcent),sweep_angles=pair["angles"])
def audit(g,y,rec,d,ref,steps=STEPS,starts=STARTS):
    raw=np.loadtxt(Path(rec["graph_path"].replace("\\","/")),skiprows=1,ndmin=2)
    i,j=raw[:,:2].astype(int).T-1;w=raw[:,2]
    assert d["cuts"].shape==(starts,3,steps+1)
    assert np.isfinite(d["cuts"]).all() and np.isfinite(d["seconds"]).all() and np.all(d["seconds"]>=0)
    assert np.all(d["cuts"][:,:,0]==d["cuts"][:,0:1,0])
    np.testing.assert_allclose(d["best"],np.maximum.accumulate(d["cuts"],axis=2),atol=1e-7,rtol=0)
    assert np.all(np.diff(d["best"],axis=2)>=-1e-7)
    assert np.all(np.diff(d["cuts"][:,1],axis=1)>=-1e-7)
    assert d["active"][:,:2].all() and d["active"][:,2,0].all()
    assert not np.any(np.diff(d["active"][:,2].astype(int),axis=1)>0)
    if ref["certified"]:assert d["best"].max()<=ref["reference"]+1e-7,(g,"certified reference exceeded")
    geom=np.full((starts,3,steps+1,len(METRICS)),np.nan)
    total=float(np.einsum("ij,ij->",y,y))
    count=0
    for s in range(starts):
        for m in range(3):
            old=None;oldr=None
            for t in np.flatnonzero(d["active"][s,m]):
                x=np.unpackbits(d["labels"][s,m,t])[:len(y)]
                c=float(w[x[i]!=x[j]].sum())
                assert abs(c-d["cuts"][s,m,t])<=1e-7,(g,s,m,t,"signed cut")
                r=d["normals"][s,m,t]
                if m<2:
                    assert abs(np.linalg.norm(r)-1)<1e-9
                    q=y@r;assert not np.any(np.abs(q[(q>0)!=x])>1e-10)
                else:
                    centers=d["km_centers"][s,t]
                    q=y@(centers[1]-centers[0]);assert not np.any(np.abs(q[(q>0)!=x])>1e-10)
                a=y[x==1].sum(axis=0);b=y[x==0].sum(axis=0);na=int(x.sum());nb=len(y)-na
                aa=float(a@a);bb=float(b@b)
                ang=np.arccos(np.clip(a@b/np.sqrt(aa*bb),-1,1)) if aa*bb>0 else np.nan
                h=0 if old is None else np.mean(x!=old)
                rot=0 if oldr is None else (np.arccos(np.clip(abs(r@oldr),0,1)) if np.linalg.norm(r)>0 and np.linalg.norm(oldr)>0 else np.nan)
                geom[s,m,t]=[(total-aa/max(na,1)-bb/max(nb,1))/len(y),(total+len(y)-2*(np.sqrt(aa)+np.sqrt(bb)))/len(y),ang,min(na,nb)/len(y),min(h,1-h),rot]
                old=x;oldr=r;count+=1
            n=int(d["active"][s,m].sum())
            assert np.all(d["cuts"][s,m,n:]==d["cuts"][s,m,n-1])
            assert np.all(d["labels"][s,m,n:]==d["labels"][s,m,n-1])
    return geom,count
def identity(g,rec,ref):
    return dict(graph=g,graph_sha256=rec["graph_sha256"],factor_sha256=rec["factor_sha256"],factor_source=rec["factor_source"],reference=ref,starts=STARTS,steps=STEPS,seed=SEED,methods=list(METHODS),sources={p:e33.source_sha(p) for p in ("scripts/experiment037_best_known.py","scripts/experiment033_raw_hyperplane_sweep.py","src/maxkcut/eva_direct_rounding.py","src/maxkcut/rounding.py",str(REFERENCE))})
def compute(job):
    g,root=job;root=Path(root);ref=references()[g]
    with threadpool_limits(limits=1):
        graph,y,rec=e33.inputs(g);expected=identity(g,rec,ref)
        folder=root/"graphs"/g;folder.mkdir(parents=True,exist_ok=True)
        archive=folder/"trajectory.npz";manifest=folder/"manifest.json"
        if manifest.exists():
            m=json.loads(manifest.read_text())
            assert all(m[k]==v for k,v in expected.items()) and sha(archive)==m["archive_sha256"] and m["verified"]
            print("Reused audited",g,flush=True);return g
        if archive.exists():raise RuntimeError("Uncommitted checkpoint; inspect before overwriting: "+str(archive))
        trials=[run_start(g,s,graph,y) for s in range(STARTS)]
        d={k:np.stack([a[k] for a in trials]) for k in trials[0]}
        geom,count=audit(g,y,rec,d,ref);d["geometry"]=geom
        tmp=folder/"trajectory.tmp.npz";np.savez_compressed(tmp,**d);os.replace(tmp,archive)
        expected.update(verified=True,audited_records=count,archive_sha256=sha(archive))
        tmp=folder/"manifest.tmp";dump(tmp,expected);os.replace(tmp,manifest)
        print("Audited",g,count,flush=True);return g
def aggregate(root,out,workers):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);refs=references()
    best=[];candidate=[];seconds=[];intervals=[];active=[];valid=[];geo=[];manifests=[];files=[]
    for g in GRAPHS:
        p=root/"graphs"/g;m=json.loads((p/"manifest.json").read_text())
        assert m["verified"] and m["reference"]==refs[g] and sha(p/"trajectory.npz")==m["archive_sha256"]
        manifests.append(m)
        for name in ("trajectory.npz","manifest.json"):
            f=p/name;files.append(dict(path=f.relative_to(root).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
        with np.load(p/"trajectory.npz",allow_pickle=False) as f:
            for field,arr in [("best",best),("cuts",candidate),("seconds",seconds),("intervals",intervals),("active",active),("valid",valid),("geometry",geo)]:arr.append(f[field])
    b,c,sec,iv,act,val,geom=map(np.stack,(best,candidate,seconds,intervals,active,valid,geo))
    ref=np.array([refs[g]["reference"] for g in GRAPHS]);q=100*b/ref[:,None,None,None]
    assert q.shape==(71,30,3,101)
    records=np.concatenate((np.zeros_like(act[:,:,:,:1]),np.diff(b,axis=3)>1e-8),axis=3)&act
    active_count=act.sum(axis=(0,1))
    # Active-only means; NaN when no clustering trajectory remains.
    def active_mean(x):
        n=np.sum(np.isfinite(x)&act,axis=(0,1));return np.divide(np.nansum(np.where(act,x,np.nan),axis=(0,1)),n,out=np.full(n.shape,np.nan),where=n>0)
    diag=np.stack([active_mean(geom[...,k]) for k in range(len(METRICS))]+[active_mean(records.astype(float))])
    # Compact dashboard data: incumbent tensor; other raw candidate arrays stay on VM.
    np.savez_compressed(out/"dashboard_curves.npz",quality=q,reference=ref,diagnostics=diag,active_count=active_count)
    final=q[:,:,:,-1];gm=final.mean(axis=1);base=gm[:,1]
    boot=np.random.default_rng(3701).integers(0,71,(10000,71))
    rows=[]
    for k,m in enumerate(METHODS):
        delta=gm[:,k]-base;ci=np.quantile(delta[boot].mean(axis=1),[.025,.975])
        rows.append(dict(method=m,mean=float(gm[:,k].mean()),median=float(np.median(final[:,:,k])),initial_mean=float(q[:,:,k,0].mean()),effect_pp=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),wins=int((delta>1e-9).sum()),ties=int((abs(delta)<=1e-9).sum()),losses=int((delta<-1e-9).sum()),seconds=float(sec[:,:,k].sum(axis=2).mean()),records=float(records[:,:,k].sum(axis=2).mean()),attempts=float(act[:,:,k,1:].sum(axis=2).mean()),intervals=float(iv[:,:,k,1:].sum(axis=2).mean()),invalid=int((~val[:,:,k]&act[:,:,k]).sum()),above_reference=int((final[:,:,k]>100+1e-9).sum())))
    with (out/"per_graph.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.writer(f,lineterminator="\n");writer.writerow(["graph","reference","basis","certified","family"]+[m+"_"+x for m in METHODS for x in ("mean","median","best")])
        for i,g in enumerate(GRAPHS):writer.writerow([g,ref[i],refs[g]["basis"],refs[g]["certified"],refs[g]["family"]]+[v for k in range(3) for v in (gm[i,k],np.median(final[i,:,k]),final[i,:,k].max())])
    # Candidate-level scatter, deterministic systematic subsample; labels/normal data excluded.
    sample=[]
    for k in range(3):
        xx=geom[:,:,k,:,1];yy=100*c[:,:,k]/ref[:,None,None]
        x=(xx[:,:,0,None]-xx)[act[:,:,k]];yval=(yy-yy[:,:,0,None])[act[:,:,k]]
        sample.append(np.column_stack((x[::20],yval[::20])))
    np.savez_compressed(out/"density_sample.npz",**{m:x for m,x in zip(METHODS,sample)})
    summary=dict(experiment="037",complete=True,graphs=71,starts=30,steps=100,methods=rows,audited_records=sum(m["audited_records"] for m in manifests),new_eva_solves=0,certified_references=9,reference_source=str(REFERENCE),reference_source_sha256=e33.source_sha(REFERENCE),reference_override={"G62":4872},dashboard_pages=96,workers=workers)
    dump(out/"summary.json",summary)
    storage=dict(experiment="037",archive_root=str(root.resolve()),runner=os.environ.get("RUNNER_NAME","local"),machine=os.environ.get("COMPUTERNAME",os.environ.get("HOSTNAME","unknown")),run_id=os.environ.get("GITHUB_RUN_ID","local"),commit=os.environ.get("GITHUB_SHA","local"),files=files,total_bytes=sum(f["bytes"] for f in files),audited_records=summary["audited_records"],complete=True,graph_manifests=manifests)
    dump(root/"storage_manifest.json",storage);dump(out/"storage_manifest.json",storage)
    print("COMPLETE",summary,flush=True)
def test():
    from scipy import sparse
    e33.warmup()
    rng=np.random.default_rng(37);y=rng.normal(size=(19,5));y/=np.linalg.norm(y,axis=1)[:,None]
    a=np.triu(rng.integers(-3,6,size=(19,19)),1);g=sparse.csr_matrix(a+a.T,dtype=float)
    for s in range(3):
        d=run_start("G1",s,g,y,10)
        assert d["best"].shape==(3,11) and np.all(np.diff(d["best"],axis=1)>=0)
        assert np.all(d["cuts"][:,0]==d["cuts"][0,0])
        for m in range(3):
            for t in np.flatnonzero(d["active"][m]):
                x=np.unpackbits(d["labels"][m,t])[:len(y)]
                i,j=np.triu_indices(len(y),1)
                assert abs(a[i,j][x[i]!=x[j]].sum()-d["cuts"][m,t])<1e-7
        e=run_start("G1",s,g,y,10)
        for k in ("cuts","best","labels","active"):np.testing.assert_array_equal(d[k],e[k])
    assert references()["G62"]["reference"]==4872
    print("PASS signed cuts, common initial cut, original KMeansNM equivalence, reproducibility and reference table")
def main():
    p=argparse.ArgumentParser();p.add_argument("--action",choices=["test","compute","aggregate"],default="compute");p.add_argument("--workers",type=int,default=16);p.add_argument("--archive",type=Path);p.add_argument("--output",type=Path);a=p.parse_args()
    if a.action=="test":test();return
    if not a.archive or not a.output:raise ValueError("Specify durable archive and compact output")
    if a.workers<1:raise ValueError("workers must be positive")
    n=min(a.workers,os.cpu_count() or 1,len(GRAPHS));print("Requested",a.workers,"effective",n,flush=True)
    if a.action=="compute":
        a.archive.mkdir(parents=True,exist_ok=True);e33.warmup()
        with ProcessPoolExecutor(max_workers=n) as pool:
            fs=[pool.submit(compute,(g,str(a.archive))) for g in GRAPHS]
            for f in as_completed(fs):f.result()
    aggregate(a.archive,a.output,n)
if __name__=="__main__":
    with threadpool_limits(limits=1):main()
