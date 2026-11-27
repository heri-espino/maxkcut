"""Build 96 dashboard pages directly from archived compact data; no LaTeX or optimization."""
from __future__ import annotations
import argparse,csv,json,os,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pymupdf as fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import awesomefigures as af
from matplotlib import font_manager
NAMES=("Repeated GW","Random sweep","KMeansNM")
COLORS=("#4477AA","#228833","#AA3377")
def style():
    af.set_style("heri",font="STIX",rasterize=True,use_tex=False)
    plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","pdf.fonttype":42,"axes.unicode_minus":False,"font.size":10,"axes.titlesize":11})
    font_manager.findfont("STIXGeneral",fallback_to_default=False)
def save(fig,p):
    fig.tight_layout(pad=1.2,rect=(0,0,1,.95));af.save_figure(fig,p,dpi=600);plt.close(fig)
def labels(axs):
    for j,ax in enumerate(np.asarray(axs).flat):af.panel_label(ax,chr(65+j))
def dashboard(job):
    path,k,title,x,graphs,lo,hi=job
    style();fig,axs=plt.subplots(3,2,figsize=(7.0,9.1))
    a,b,c,d,e,f=axs.flat;t=np.arange(x.shape[1]);color=COLORS[k]
    for y in x:a.plot(t,y,color=color,alpha=.16,lw=.65)
    a.plot(t,x.mean(axis=0),color=color,lw=2);a.axhline(100,color=".35",ls="--",lw=.8);a.set(title="All 71 graph trajectories",xlabel="Attempt",ylabel="Reference attained (%)",ylim=(lo,hi))
    b.plot(t,x.mean(axis=0),label="Mean",color=color);b.plot(t,np.median(x,axis=0),label="Median",color=".2",ls="--")
    b.fill_between(t,*np.quantile(x,[.25,.75],axis=0),color=color,alpha=.15,label="Middle 50% of graphs");b.axhline(100,color=".35",ls=":");b.set(title="Across-graph summary",xlabel="Attempt",ylabel="Reference attained (%)",ylim=(lo,hi));b.legend(fontsize=8)
    c.hist(x[:,-1],bins=np.linspace(lo,hi,21),color=color,alpha=.8);c.axvline(100,color=".35",ls="--");c.set(title="Final solution quality",xlabel="Reference attained (%)",ylabel="Graphs",xlim=(lo,hi))
    denom=100-x[:,0];valid=denom>1e-9;gap=100*(x[valid,-1]-x[valid,0])/denom[valid]
    if len(gap):d.hist(gap,bins=16,color=color,alpha=.8)
    d.axvline(100,color=".35",ls="--");d.set(title="Initial gap closed (eligible graphs)",xlabel="Gap closed (%)",ylabel="Graphs")
    e.plot(t,100*np.mean(x>x[:,0,None]+1e-9,axis=0),color=color);e.set(title="Graphs improved since initialization",xlabel="Attempt",ylabel="Graphs (%)",ylim=(0,103))
    order=np.argsort(x[:,-1]);f.plot(np.arange(len(order)),x[order,-1],".",color=color,ms=3);f.axhline(100,color=".35",ls="--");f.set(title="Final graph ranking",xlabel="Rank (ascending quality)",ylabel="Reference attained (%)",ylim=(lo,hi))
    fig.suptitle(title,fontsize=15)
    labels(axs);save(fig,Path(path))
def make_report(data,out,workers):
    out=Path(out);data=Path(data);out.mkdir(parents=True,exist_ok=True);figdir=out/"figures";figdir.mkdir(exist_ok=True);datadir=out/"data";datadir.mkdir(exist_ok=True)
    for name in ("summary.json","per_graph.csv","dashboard_curves.npz","density_sample.npz","storage_manifest.json"):shutil.copy2(data/name,datadir/name)
    s=json.loads((datadir/"summary.json").read_text());assert s["complete"] and s["graphs"]==71 and s["starts"]==30 and s["new_eva_solves"]==0
    with np.load(datadir/"dashboard_curves.npz") as z:q=z["quality"];diag=z["diagnostics"];act=z["active_count"]
    graphs=[r["graph"] for r in csv.DictReader((datadir/"per_graph.csv").open())]
    assert q.shape==(71,30,3,101) and np.all(np.diff(q,axis=3)>=-1e-9)
    lo=min(100,5*np.floor(q.min()/5));hi=max(102,5*np.ceil(q.max()/5))
    jobs=[];dash=[]
    def add(k,title,x,name):
        jobs.append((str(figdir/(name+".pdf")),k,title,x,graphs,lo,hi))
        dash.append((name,title,float(x[:,-1].mean()),float(np.median(x[:,-1]))))
    for r in range(30):
        for k,m in enumerate(NAMES):add(k,f"Repetition {r+1:02d} / {m}",q[:,r,k],f"rep_{r+1:02d}_{k}")
    for mode,fn in (("Mean",np.mean),("Median",np.median)):
        for k,m in enumerate(NAMES):add(k,f"{mode} over 30 repetitions / {m}",fn(q[:,:,k],axis=1),f"aggregate_{mode.lower()}_{k}")
    effective=min(workers,os.cpu_count() or 1,96)
    print("Dashboard figure workers:",effective,flush=True)
    with ProcessPoolExecutor(max_workers=effective) as pool:list(pool.map(dashboard,jobs,chunksize=1))
    assert len(dash)==96
    target=out/"Experiment037_Dashboard_Appendix.pdf"
    with fitz.open() as combined:
        toc=[]
        for i,(name,title,mean,med) in enumerate(dash):
            with fitz.open(figdir/(name+".pdf")) as page:
                assert len(page)==1 and len(page[0].get_text().strip())>30,name
                combined.insert_pdf(page)
            toc.append([1,title,i+1])
        combined.set_toc(toc)
        combined.save(target,garbage=4,deflate=True)
    with fitz.open(target) as doc:
        assert len(doc)==96
        for i,page in enumerate(doc):
            assert len(page.get_text().strip())>30,i
            for block in page.get_text("blocks"):
                if block[6]==0:
                    assert block[0]>=-1 and block[1]>=-1 and block[2]<=page.rect.width+1 and block[3]<=page.rect.height+1,(i,block)
        previews=out/"previews";previews.mkdir(exist_ok=True)
        for i in (0,1,2,89,90,95):
            doc[i].get_pixmap(matrix=fitz.Matrix(.9,.9)).save(previews/f"page-{i+1:02d}.png")
    (out/"dashboard_index.json").write_text(json.dumps([{"page":i+1,"figure":name+".pdf","title":title,"final_mean":mean,"final_median":med} for i,(name,title,mean,med) in enumerate(dash)],indent=2)+"\n",encoding="utf8")
    print("Verified",len(dash),"dashboard pages; no numerical recomputation",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=16);a=p.parse_args();make_report(a.data,a.output,a.workers)
