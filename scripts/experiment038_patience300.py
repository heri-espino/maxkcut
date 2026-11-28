"""E038: continue E037 audited trajectories without replaying evaluated cuts."""
from __future__ import annotations
import argparse,csv,json,os,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
import experiment037_best_known as old
import experiment033_raw_hyperplane_sweep as e33

PATIENCES=(5,10,20,50,100,150,200,250,300)
P=300
CAP=5000
EPS=1e-8
METHODS=old.METHODS
SOURCE=Path("results/gset/experiment037-best-known-dashboards/full/data/storage_manifest.json")
ROOT=Path("results/gset/experiment038-patience300/full")

def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest() if hasattr(hashlib,"file_digest") else hashlib.sha256(f.read()).hexdigest()

def write_json(p,d):
    tmp=p.with_suffix(".tmp");old.dump(tmp,d);os.replace(tmp,p)

def write_npz(p,d):
    tmp=p.with_suffix(".tmp.npz");np.savez_compressed(tmp,**d);os.replace(tmp,p)

def policy(cuts,p,stable=False):
    """First hit of p consecutive nonrecords; natural stop or cap otherwise."""
    b=float(cuts[0]);last=0
    for t,c in enumerate(cuts[1:],1):
        if c>b+EPS:b=float(c);last=t
        if t-last>=p:return t,b,last,"patience"
    t=len(cuts)-1
    return t,b,last,"stable" if stable else "cap"

def update_centers(y,x,rng):
    centers=[]
    for k in range(2):
        members=y[x==k];raw=members.sum(axis=0);norm=np.linalg.norm(raw)
        centers.append(raw/norm if len(members) and norm>1e-12 else y[rng.integers(len(y))])
    return np.asarray(centers)

def extend(g,s,graph,y,d,m,cap=CAP,patience=P):
    # Prefix is immutable; only new candidates are evaluated. Each method
    # consumes the same Gaussian proposal stream indexed by absolute attempt.
    active=int(d["active"][s,m].sum())
    c=d["cuts"][s,m,:active].copy()
    b=float(c.max());last=int(np.flatnonzero(np.r_[True,np.diff(np.maximum.accumulate(c))>EPS])[-1])
    edges=e33.edge_arrays(graph)
    nr=y.shape[1];n=len(y)
    rng=np.random.default_rng(np.random.SeedSequence([old.SEED,37,int(g[1:]),s,1]))
    if m<2:
        rng.standard_normal((active-1,nr)) # advance, never score a reused draw
        r=d["normals"][s,m,last].copy()
        stable=False
    else:
        # Replay only the centroid updates and RNG consumption, not cut
        # evaluations, to reconstruct the exact next Lloyd state if needed.
        rng=old.rng_for(g,s)
        centers=rng.standard_normal((2,nr));centers[1]=-centers[0];centers/=np.linalg.norm(centers,axis=1)[:,None]
        previous=None;stable=False
        for t in range(active):
            x=np.unpackbits(d["labels"][s,m,t])[:n]
            np.testing.assert_allclose(centers,d["km_centers"][s,t],rtol=0,atol=1e-12)
            stable=previous is not None and np.array_equal(x,previous)
            if stable:
                assert t==active-1;break
            previous=x.copy();centers=update_centers(y,x,rng)
    rows={k:[] for k in ("cuts","labels","normals","centers","seconds","intervals","valid","angle","predicted")}
    for t in range(active,cap+1):
        if stable or t-1-last>=patience:break
        clock=time.perf_counter();valid=True;theta=0.;intervals=1
        cc=np.zeros((2,nr))
        if m==0:
            z=rng.standard_normal(nr);rr=e33.unit(z);x=(y@rr>0).astype(np.uint8);cut=e33.score(edges,x);predicted=cut
        elif m==1:
            z=rng.standard_normal(nr);u=z-(z@r)*r
            if np.linalg.norm(u)<=1e-13:
                u=np.eye(nr)[np.argmin(np.abs(r))];u-=(u@r)*r
            rr,x,info=e33.optimize_semicircle(graph,y,r,e33.unit(u))
            cut=e33.score(edges,x);valid=bool(info["numerically_valid"]);theta=float(info["theta"]);intervals=int(info["distinct_events"]+1);predicted=float(info["predicted_arc_max"])
            assert abs(cut-info["realized_arc_cut"])<=1e-7 and cut>=b-1e-7
        else:
            cc=centers.copy();x=np.argmax(y@centers.T,axis=1).astype(np.uint8)
            cut=e33.score(edges,x);predicted=cut;v=centers[1]-centers[0];rr=v/np.linalg.norm(v) if np.linalg.norm(v)>1e-13 else np.zeros(nr)
            stable=previous is not None and np.array_equal(x,previous)
            if not stable:previous=x.copy();centers=update_centers(y,x,rng)
        elapsed=time.perf_counter()-clock
        if cut>b+EPS:
            b=cut;last=t
            if m==1:r=rr.copy()
        for k,v in zip(rows,(cut,np.packbits(x),rr,cc,elapsed,intervals,valid,theta,predicted)):rows[k].append(v)
    shapes={"labels":((n+7)//8,),"normals":(nr,),"centers":(2,nr)}
    types={"labels":np.uint8,"intervals":np.int64,"valid":bool}
    out={k:np.asarray(v,dtype=types.get(k,float)).reshape((len(v),)+shapes.get(k,())) for k,v in rows.items()}
    out["prefix_count"]=np.asarray(active);out["stable"]=np.asarray(stable)
    allcuts=np.r_[c,out["cuts"]]
    stop,_,_,reason=policy(allcuts,patience,stable)
    assert stop==len(allcuts)-1 and (reason!="cap" or stop==cap)
    return out

def audit_extension(d,m,y,i,j,w,ref):
    assert len(d["cuts"])==len(d["labels"])==len(d["normals"])
    assert np.isfinite(d["cuts"]).all() and np.isfinite(d["seconds"]).all() and np.all(d["seconds"]>=0)
    if ref["certified"] and len(d["cuts"]):assert d["cuts"].max()<=ref["reference"]+1e-7
    for t,(packed,r,c) in enumerate(zip(d["labels"],d["normals"],d["cuts"])):
        x=np.unpackbits(packed)[:len(y)]
        assert abs(float(w[x[i]!=x[j]].sum())-c)<=1e-7
        if m<2:
            assert abs(np.linalg.norm(r)-1)<1e-9
            z=y@r
        else:
            centers=d["centers"][t];z=y@(centers[1]-centers[0])
        assert not np.any(np.abs(z[(z>0)!=x])>1e-10)

def compute(job):
    g,source,root,cap=job;source=Path(source);root=Path(root)
    with threadpool_limits(limits=1):
        graph,y,rec=e33.inputs(g);ref=old.references()[g]
        manifest=json.loads(SOURCE.read_text(encoding="utf8"))
        files={x["path"]:x["sha256"] for x in manifest["files"]}
        folder=source/"graphs"/g
        for name in ("trajectory.npz","manifest.json"):
            assert digest(folder/name)==files[f"graphs/{g}/{name}"],(g,name,"E037 SHA mismatch")
        prior=json.loads((folder/"manifest.json").read_text(encoding="utf8"))
        assert prior["verified"] and prior["reference"]==ref and prior["factor_sha256"]==rec["factor_sha256"] and prior["graph_sha256"]==rec["graph_sha256"]
        with np.load(folder/"trajectory.npz",allow_pickle=False) as f:d={k:f[k] for k in f.files}
        assert d["cuts"].shape==(30,3,101)
        raw=np.loadtxt(Path(rec["graph_path"].replace("\\","/")),skiprows=1,ndmin=2)
        i,j=raw[:,:2].astype(int).T-1;w=raw[:,2]
        dest=root/"graphs"/g;dest.mkdir(parents=True,exist_ok=True)
        identity=dict(graph=g,prefix_sha=prior["archive_sha256"],factor_sha=rec["factor_sha256"],graph_sha=rec["graph_sha256"],cap=cap,patience=P,driver_sha=e33.source_sha(__file__))
        records=[];curves=[];times=[];hazard=np.zeros((3,P+1,2),np.int64)
        new_count=0;archives=[]
        for s in range(30):
            f=dest/f"start-{s:02d}.npz";meta=f.with_suffix(".json")
            if meta.exists():
                mdata=json.loads(meta.read_text());assert mdata["identity"]==identity and digest(f)==mdata["sha256"]
                with np.load(f,allow_pickle=False) as z:packed={k:z[k] for k in z.files}
            else:
                packed={}
                for m in range(3):
                    z=extend(g,s,graph,y,d,m,cap)
                    audit_extension(z,m,y,i,j,w,ref)
                    packed.update({f"{m}_{k}":v for k,v in z.items()})
                write_npz(f,packed);write_json(meta,dict(identity=identity,sha256=digest(f)))
            archives.append(dict(path=f.relative_to(root).as_posix(),sha256=digest(f),bytes=f.stat().st_size,manifest_sha256=digest(meta)))
            qs=[];ts=[]
            for m in range(3):
                z={k[2:]:v for k,v in packed.items() if k.startswith(f"{m}_")}
                a=int(z["prefix_count"]);new_count+=len(z["cuts"])
                c=np.r_[d["cuts"][s,m,:a],z["cuts"]];b=np.maximum.accumulate(c)
                sec=np.cumsum(np.r_[d["seconds"][s,m,:a],z["seconds"]])
                effort=np.cumsum(np.r_[d["intervals"][s,m,:a],z["intervals"]])
                stable=bool(z["stable"])
                stop,_,_,reason=policy(c,P,stable);assert stop==len(c)-1 and (reason!="cap" or stop==cap)
                streak=0
                for k in range(1,len(c)):
                    h=min(P,streak);hit=int(b[k]>b[k-1]+EPS)
                    hazard[m,h,0]+=1;hazard[m,h,1]+=hit;streak=0 if hit else streak+1
                for p in PATIENCES:
                    t,v,last,why=policy(c,p,stable)
                    records.append(dict(graph=g,start=s+1,method=METHODS[m],patience=p,stop=t,last_record=last,reason=why,quality=100*v/ref["reference"],cut=v,initial=100*c[0]/ref["reference"],seconds=sec[t],intervals=int(effort[t]),demo_quality=100*d["best"][s,m,-1]/ref["reference"]))
                qs.append(np.pad(100*b/ref["reference"],(0,cap+1-len(b)),mode="edge"))
                ts.append(np.pad(sec,(0,cap+1-len(sec)),mode="edge"))
            curves.append(qs);times.append(ts)
        q=np.asarray(curves);tm=np.asarray(times)
        # Per-graph compact data only; raw extensions remain in the VM archive.
        write_npz(dest/"compact.npz",dict(mean=q.mean(axis=0),median=np.median(q,axis=0),low=np.quantile(q,.25,axis=0),high=np.quantile(q,.75,axis=0),mean_seconds=tm.mean(axis=0),hazard=hazard))
        write_json(dest/"policies.json",records)
        write_json(dest/"complete.json",dict(identity=identity,new_count=new_count,archives=archives,compact_sha=digest(dest/"compact.npz"),policies_sha=digest(dest/"policies.json")))
        print(g,"new audited cuts",new_count,flush=True)
        return g

def aggregate(root,out,cap):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    records=[];meta=[];curves={}
    for g in old.GRAPHS:
        p=root/"graphs"/g;m=json.loads((p/"complete.json").read_text())
        assert digest(p/"compact.npz")==m["compact_sha"] and digest(p/"policies.json")==m["policies_sha"]
        meta.append(m);records+=json.loads((p/"policies.json").read_text())
        with np.load(p/"compact.npz") as z:
            for k in z.files:curves.setdefault(k,[]).append(z[k])
    assert len(records)==71*30*3*len(PATIENCES)
    def csvout(name,rows):
        with (out/name).open("w",newline="",encoding="utf8") as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");wr.writeheader();wr.writerows(rows)
    csvout("per_start_policy.csv",records)
    grouped={}
    for row in records:grouped.setdefault((row["graph"],row["method"],row["patience"]),[]).append(row)
    pg=[]
    for (g,m,p),rr in grouped.items():
        assert len(rr)==30
        pg.append(dict(graph=g,method=m,patience=p,mean_quality=float(np.mean([x["quality"] for x in rr])),median_quality=float(np.median([x["quality"] for x in rr])),mean_attempts=float(np.mean([x["stop"] for x in rr])),mean_seconds=float(np.mean([x["seconds"] for x in rr])),mean_intervals=float(np.mean([x["intervals"] for x in rr])),mean_gain_from_demo=float(np.mean([x["quality"]-x["demo_quality"] for x in rr])),capped=sum(x["reason"]=="cap" for x in rr),stable=sum(x["reason"]=="stable" for x in rr)))
    csvout("per_graph_policy.csv",pg)
    rng=np.random.default_rng(3801);boot=rng.integers(0,71,(10000,71))
    summaries=[]
    for m in METHODS:
        prev=None
        for p in PATIENCES:
            rows=[x for x in pg if x["method"]==m and x["patience"]==p]
            assert len(rows)==71
            v=np.array([x["mean_quality"] for x in rows]);delta=np.zeros(71) if prev is None else v-prev
            ci=np.quantile(delta[boot].mean(axis=1),[.025,.975])
            summaries.append(dict(method=m,patience=p,mean_quality=float(v.mean()),marginal_pp=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),mean_attempts=float(np.mean([x["mean_attempts"] for x in rows])),mean_seconds=float(np.mean([x["mean_seconds"] for x in rows])),mean_intervals=float(np.mean([x["mean_intervals"] for x in rows])),mean_gain_from_demo=float(np.mean([x["mean_gain_from_demo"] for x in rows])),capped=sum(x["capped"] for x in rows),stable=sum(x["stable"] for x in rows)))
            prev=v
    csvout("patience_summary.csv",summaries)
    comparisons=[]
    for p in PATIENCES:
        v={m:np.array([x["mean_quality"] for x in pg if x["method"]==m and x["patience"]==p]) for m in METHODS}
        for m1,m2 in ((METHODS[1],METHODS[0]),(METHODS[2],METHODS[0]),(METHODS[1],METHODS[2])):
            delta=v[m1]-v[m2];ci=np.quantile(delta[boot].mean(axis=1),[.025,.975])
            comparisons.append(dict(patience=p,method=m1,baseline=m2,effect_pp=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),wins=int((delta>EPS).sum()),ties=int((abs(delta)<=EPS).sum()),losses=int((delta<-EPS).sum())))
    csvout("paired_comparisons.csv",comparisons)
    arr={k:np.stack(v) for k,v in curves.items()}
    write_npz(out/"curves.npz",arr)
    old.dump(out/"storage_manifest.json",dict(experiment="038",source_manifest_sha256=digest(SOURCE),source_run=46065620135,root=str(root.resolve()),graphs=meta))
    old.dump(out/"summary.json",dict(experiment="038",graphs=71,starts=30,patiences=PATIENCES,cap=cap,new_audited_records=sum(m["new_count"] for m in meta),new_eva_solves=0,complete=True,methods=summaries,comparisons=comparisons))
    print("COMPLETE",json.dumps(summaries),flush=True)

def test():
    assert policy([1,2,2,2,3],2)==(3,2.,1,"patience")
    assert policy([1,1,2,2,2],2)==(4,2.,2,"patience")
    assert policy([1,2,2],5,True)==(2,2.,1,"stable")
    assert policy([1,2,2],5)==(2,2.,1,"cap")
    from scipy import sparse
    e33.warmup()
    rng=np.random.default_rng(380);y=rng.normal(size=(19,5));y/=np.linalg.norm(y,axis=1)[:,None]
    a=np.triu(rng.integers(-3,6,size=(19,19)),1);graph=sparse.csr_matrix(a+a.T,dtype=float)
    short=old.run_start("G1",0,graph,y,10);long=old.run_start("G1",0,graph,y,30)
    d={k:v[None] for k,v in short.items()}
    i,j=np.triu_indices(len(y),1)
    for m in range(3):
        z=extend("G1",0,graph,y,d,m,29 if m==2 else 30,300)
        audit_extension(z,m,y,i,j,a[i,j],{"certified":False})
        n=int(z["prefix_count"]);end=int(long["active"][m].sum())
        np.testing.assert_allclose(z["cuts"],long["cuts"][m,n:end],rtol=0,atol=1e-7)
        np.testing.assert_array_equal(z["labels"],long["labels"][m,n:end])
    print("PASS first-hit policies, exact synthetic E037 continuation, independent signed cut audit")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--action",choices=("test","compute","aggregate"),default="compute");ap.add_argument("--source",type=Path);ap.add_argument("--archive",type=Path);ap.add_argument("--output",type=Path);ap.add_argument("--workers",type=int,default=16);ap.add_argument("--cap",type=int,default=CAP);a=ap.parse_args()
    if a.action=="test":test();return
    assert a.archive and a.output and a.cap>=400 and a.workers>=1
    if a.action=="compute":
        assert a.source and a.source.resolve()!=a.archive.resolve()
        e33.warmup();n=min(a.workers,os.cpu_count() or 1,71)
        print("Graph workers",n,flush=True)
        with ProcessPoolExecutor(max_workers=n) as pool:
            for f in as_completed([pool.submit(compute,(g,a.source,a.archive,a.cap)) for g in old.GRAPHS]):f.result()
    aggregate(a.archive,a.output,a.cap)
if __name__=="__main__":
    with threadpool_limits(limits=1):main()
