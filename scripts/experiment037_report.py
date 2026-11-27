"""Build 44 shared-axis comparison pages directly from archived compact data; no LaTeX or optimization."""
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
from matplotlib.transforms import Bbox
NAMES=("Repeated GW","Random sweep","KMeansNM")
COLORS=("#4477AA","#228833","#AA3377")
def style():
    af.set_style("heri",font="STIX",rasterize=True,use_tex=False)
    plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","pdf.fonttype":42,"axes.unicode_minus":False,"font.size":10,"axes.titlesize":11})
    font_manager.findfont("STIXGeneral",fallback_to_default=False)
def save(fig,p):
    # AwesomeFigures defaults to a tight crop with only 0.035 inches of padding.
    # Use the full fixed-size canvas instead: PDF font metrics can exceed Agg's
    # tight bounding box. Reserve explicit space for the title and panel labels.
    fig.tight_layout(pad=1.2,rect=(.035,.025,.975,.92))
    width,height=fig.get_size_inches()
    try:
        af.save_figure(fig,p,dpi=600,bbox_inches=Bbox.from_bounds(0,0,width,height))
    finally:
        plt.close(fig)

def verify_page(page, name):
    assert abs(page.rect.width-504)<.1 and abs(page.rect.height-655.2)<.1,(name,page.rect)
    assert len(page.get_text().strip())>30,name
    for block in page.get_text("blocks"):
        if block[6]==0:
            assert block[0]>=-1 and block[1]>=-1 and block[2]<=page.rect.width+1 and block[3]<=page.rect.height+1,(name,block)
def labels(axs):
    for j,ax in enumerate(np.asarray(axs).flat):af.panel_label(ax,chr(65+j))
METHOD_KEYS=("gw","random-sweep","kmeansnm")
STYLES=("-", "--", "-.")
TOL=1e-9

def legend(ax):
    ax.legend(fontsize=8,loc="best")

def reference(ax):
    ax.axhline(100,color=".35",ls=":",lw=.8,zorder=0)

def curves(ax,t,x,fn):
    for k,m in enumerate(NAMES):
        ax.plot(t,fn(x[:,k,:],axis=0),label=m,color=COLORS[k],ls=STYLES[k],lw=1.7)

def shared_dashboard(job):
    path,title,x,graphs,lo,hi=job
    # x is graph x method x attempt: the exact same graph order in all methods.
    style();fig,axs=plt.subplots(3,2,figsize=(7.0,9.1))
    a,b,c,d,e,f=axs.flat;t=np.arange(x.shape[-1]);final=x[:,:,-1]
    for ax,fn,heading in ((a,np.mean,"Across-graph mean"),(b,np.median,"Across-graph median")):
        curves(ax,t,x,fn);reference(ax)
        ax.set(title=heading,xlabel="Attempt",ylabel="Reference attained (%)",ylim=(lo,hi))
    legend(a)
    for k,m in enumerate(NAMES):
        vals=np.sort(final[:,k])
        c.step(vals,np.arange(1,len(vals)+1)/len(vals),where="post",color=COLORS[k],ls=STYLES[k],label=m,lw=1.5)
    c.axvline(100,color=".35",ls=":",lw=.8)
    c.set(title="Final quality: empirical CDF",xlabel="Reference attained (%)",ylabel="Fraction of graphs",xlim=(lo,hi),ylim=(0,1.02))
    # Paired comparison uses the same graph in both methods, not sorted ranks.
    for k in (1,2):
        delta=np.sort(final[:,k]-final[:,0])
        win=int((delta>TOL).sum());tie=int((abs(delta)<=TOL).sum());loss=len(delta)-win-tie
        d.step(delta,np.arange(1,len(delta)+1)/len(delta),where="post",color=COLORS[k],ls=STYLES[k],label=f"{NAMES[k]}: {win}/{tie}/{loss}",lw=1.5)
    d.axvline(0,color=".35",ls=":",lw=.8)
    d.set(title="Paired final advantage vs GW",xlabel="Difference (percentage points)",ylabel="Fraction of graphs",ylim=(0,1.02))
    d.legend(fontsize=7,title="Wins / ties / losses",title_fontsize=7)
    # Split the canonical graph list into two legible labeled panels.
    for ax,ids in ((e,np.arange(36)),(f,np.arange(36,len(graphs)))):
        for k,m in enumerate(NAMES):
            ax.plot(np.arange(len(ids)),final[ids,k],color=COLORS[k],ls=STYLES[k],marker=".",ms=3,lw=.85,label=m)
        reference(ax)
        ax.set(title=f"Final quality: {graphs[ids[0]]}–{graphs[ids[-1]]}",ylabel="Reference attained (%)",ylim=(lo,hi),xlim=(-.5,len(ids)-.5))
        ax.set_xticks(np.arange(len(ids)),[graphs[i] for i in ids],rotation=90,fontsize=6)
    fig.suptitle(title,fontsize=15,y=.96)
    labels(axs);save(fig,Path(path))

def graph_dashboard(job):
    path,title,q,graphs,refs=job
    # Each graph gets an individual axis with all three methods and 30-start IQR.
    style();fig,axs=plt.subplots(3,2,figsize=(7.0,9.1))
    t=np.arange(q.shape[-1])
    for i,ax in enumerate(axs.flat):
        if i>=len(graphs):
            ax.set_visible(False);continue
        for k,m in enumerate(NAMES):
            z=q[i,:,k,:]
            ax.plot(t,z.mean(axis=0),color=COLORS[k],ls=STYLES[k],label=m,lw=1.5)
            ax.fill_between(t,*np.quantile(z,[.25,.75],axis=0),color=COLORS[k],alpha=.13,linewidth=0)
        reference(ax)
        ax.set(title=f"{graphs[i]} | reference {refs[i]:g}",xlabel="Attempt",ylabel="Reference attained (%)")
        # The local axis is shared across methods; keep 100% visible.
        bottom=min(100,float(q[i].min()));top=max(100,float(q[i].max()))
        margin=max(.25,.06*(top-bottom))
        ax.set_ylim(bottom-margin,top+margin)
        if i==0:legend(ax)
        af.panel_label(ax,chr(65+i))
    fig.suptitle(title,fontsize=15,y=.96)
    save(fig,Path(path))

def render(job):
    kind,args=job
    if kind=="shared":shared_dashboard(args)
    elif kind=="graphs":graph_dashboard(args)
    else:raise ValueError(kind)

def make_report(data,out,workers):
    out=Path(out);data=Path(data);out.mkdir(parents=True,exist_ok=True)
    figdir=out/"figures";figdir.mkdir(exist_ok=True);datadir=out/"data";datadir.mkdir(exist_ok=True)
    for name in ("summary.json","per_graph.csv","dashboard_curves.npz","density_sample.npz","storage_manifest.json"):
        shutil.copy2(data/name,datadir/name)
    s=json.loads((datadir/"summary.json").read_text())
    assert s["complete"] and s["graphs"]==71 and s["starts"]==30 and s["new_eva_solves"]==0
    assert tuple(m["method"] for m in s["methods"])==METHOD_KEYS
    with np.load(datadir/"dashboard_curves.npz",allow_pickle=False) as z:
        q=z["quality"];refs=z["reference"]
    with (datadir/"per_graph.csv").open(newline="",encoding="utf8") as f:rows=list(csv.DictReader(f))
    graphs=[r["graph"] for r in rows]
    assert len(set(graphs))==71 and graphs==sorted(graphs,key=lambda g:int(g[1:]))
    assert q.shape==(71,30,3,101) and np.isfinite(q).all()
    assert np.all(np.diff(q,axis=3)>=-TOL)
    np.testing.assert_allclose(q[:,:,:,0],np.repeat(q[:,:,0:1,0],3,axis=2),rtol=0,atol=TOL)
    np.testing.assert_allclose(refs,[float(r["reference"]) for r in rows],rtol=0,atol=TOL)
    final=q[:,:,:,-1]
    for k,m in enumerate(METHOD_KEYS):
        for stat,values in (("mean",final[:,:,k].mean(axis=1)),("median",np.median(final[:,:,k],axis=1)),("best",final[:,:,k].max(axis=1))):
            np.testing.assert_allclose(values,[float(r[m+"_"+stat]) for r in rows],rtol=0,atol=TOL)
        np.testing.assert_allclose(final[:,:,k].mean(),s["methods"][k]["mean"],rtol=0,atol=TOL)
    lo=min(100,5*np.floor(q.min()/5));hi=max(102,5*np.ceil(q.max()/5))
    jobs=[];dash=[]
    def add_shared(title,x,name,kind):
        jobs.append(("shared",(str(figdir/(name+".pdf")),title,x,graphs,lo,hi)))
        dash.append(dict(figure=name+".pdf",title=title,kind=kind,graphs=graphs,
                         final_mean={m:float(x[:,k,-1].mean()) for k,m in enumerate(METHOD_KEYS)},
                         final_median={m:float(np.median(x[:,k,-1])) for k,m in enumerate(METHOD_KEYS)}))
    # Aggregate pages first, so the direct comparison is immediately visible.
    for mode,fn in (("Mean",np.mean),("Median",np.median)):
        add_shared(f"{mode} of 30 starts per graph | All 71 G-set instances",fn(q,axis=1),f"comparison_{mode.lower()}","aggregate")
    for rep in range(30):
        add_shared(f"Paired start {rep+1:02d} | All 71 G-set instances",q[:,rep],f"comparison_rep_{rep+1:02d}","paired_start")
    for i in range(0,71,6):
        names=graphs[i:i+6];name=f"instances_{i//6+1:02d}"
        title=f"{names[0]}–{names[-1]} | 30-start mean and middle 50%"
        jobs.append(("graphs",(str(figdir/(name+".pdf")),title,q[i:i+6],names,refs[i:i+6])))
        dash.append(dict(figure=name+".pdf",title=title,kind="individual_graphs",graphs=names))
    assert len(jobs)==len(dash)==44
    assert [g for d in dash if d["kind"]=="individual_graphs" for g in d["graphs"]]==graphs
    # Preserve the original compact numeric summary, including its historical
    # 96-page layout field. The new layout is described separately in the index.
    for i in (0,2,32):
        render(jobs[i])
        with fitz.open(figdir/dash[i]["figure"]) as p:
            assert len(p)==1
            verify_page(p[0],dash[i]["figure"])
    print("Aggregate, paired-start and individual-graph preflights passed",flush=True)
    effective=max(1,min(workers,os.cpu_count() or 1,len(jobs)-3))
    print("Dashboard figure workers:",effective,flush=True)
    with ProcessPoolExecutor(max_workers=effective) as pool:
        list(pool.map(render,[job for i,job in enumerate(jobs) if i not in (0,2,32)],chunksize=1))
    target=out/"Experiment037_Dashboard_Appendix.pdf"
    with fitz.open() as combined:
        toc=[]
        for i,d in enumerate(dash):
            with fitz.open(figdir/d["figure"]) as p:
                assert len(p)==1
                verify_page(p[0],d["figure"])
                combined.insert_pdf(p)
            toc.append([1,d["title"],i+1]);d["page"]=i+1
        combined.set_toc(toc)
        combined.save(target,garbage=4,deflate=True)
    with fitz.open(target) as doc:
        assert len(doc)==44
        for i,page in enumerate(doc):verify_page(page,i)
        previews=out/"previews";previews.mkdir(exist_ok=True)
        for i in (0,1,2,31,32,43):
            doc[i].get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(previews/f"page-{i+1:02d}.png")
    (out/"dashboard_index.json").write_text(json.dumps(dash,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print("Verified",len(dash),"shared-axis dashboard pages, all 71 graphs; no numerical recomputation",flush=True)
    # Compact final graph comparison in the run log, for convenient review.
    print("GRAPH,REFERENCE,GW_MEAN,SWEEP_MEAN,KMEANSNM_MEAN")
    for i,g in enumerate(graphs):
        print(f"{g},{refs[i]:g},"+",".join(f"{v:.9f}" for v in final[i].mean(axis=0)))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=16);a=p.parse_args();make_report(a.data,a.output,a.workers)
