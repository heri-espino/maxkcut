"""Build reproducible STIX2 appendix and exactly 96 dashboard figures from compact audited data."""
from __future__ import annotations
import argparse,csv,json,os,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
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
    fig.tight_layout(pad=1.2);af.save_figure(fig,p,dpi=600);plt.close(fig)
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
    labels(axs);save(fig,Path(path))
def make_report(data,out,workers):
    out=Path(out);data=Path(data);out.mkdir(parents=True,exist_ok=True);figdir=out/"figures";figdir.mkdir(exist_ok=True);datadir=out/"data";datadir.mkdir(exist_ok=True)
    for name in ("summary.json","per_graph.csv","dashboard_curves.npz","density_sample.npz","storage_manifest.json"):shutil.copy2(data/name,datadir/name)
    s=json.loads((datadir/"summary.json").read_text());assert s["complete"] and s["graphs"]==71 and s["starts"]==30 and s["new_eva_solves"]==0
    with np.load(datadir/"dashboard_curves.npz") as z:q=z["quality"];diag=z["diagnostics"];act=z["active_count"]
    graphs=[r["graph"] for r in csv.DictReader((datadir/"per_graph.csv").open())]
    assert q.shape==(71,30,3,101) and np.all(np.diff(q,axis=3)>=-1e-9)
    lo=min(0 if q.min()<0 else 100,5*np.floor(q.min()/5));hi=max(102,5*np.ceil(q.max()/5))
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
    style();t=np.arange(101);rows=s["methods"];gm=q[:,:,:,-1].mean(axis=1)
    figs=[]
    def mainfig(fig,name,caption):save(fig,figdir/(name+".pdf"));figs.append((name,caption))
    fig,axs=plt.subplots(1,2,figsize=(7,3.1))
    for k,m in enumerate(NAMES):
        axs[0].plot(t,q[:,:,k].mean(axis=(0,1)),label=m,color=COLORS[k]);axs[1].plot(t,np.median(q[:,:,k],axis=(0,1)),color=COLORS[k],label=m)
    for ax,title in zip(axs,("Mean","Median")):ax.set(title=title,xlabel="Attempt",ylabel="Reference attained (%)");ax.axhline(100,color=".4",ls="--")
    axs[0].legend(fontsize=8);labels(axs);mainfig(fig,"convergence","Incumbent quality averaged or median-pooled across the 71 graphs and 30 paired starts. Stopped KMeansNM curves are carried forward, not new assignments.")
    fig,ax=plt.subplots(figsize=(7,2.7));eff=np.array([r["effect_pp"] for r in rows]);ci=np.array([[r["ci_low"],r["ci_high"]] for r in rows]);ax.errorbar(eff,np.arange(3),xerr=[eff-ci[:,0],ci[:,1]-eff],fmt="o",capsize=3);ax.axvline(0,ls="--",color=".4");ax.set(yticks=np.arange(3),yticklabels=NAMES,xlabel="Difference vs random sweep (pp)");ax.invert_yaxis();mainfig(fig,"paired_effects","Paired differences in mean final percent of reference; 95 percent graph bootstrap intervals, 10,000 resamples, unadjusted.")
    fig,axs=plt.subplots(1,2,figsize=(7,3.1))
    for k,r in enumerate(rows):
        axs[0].scatter(r["seconds"],r["mean"],color=COLORS[k],label=NAMES[k]);axs[1].scatter(r["intervals"],r["mean"],color=COLORS[k])
    for ax,title in zip(axs,("Recorded seconds","Examined intervals / assignments")):ax.set(xlabel=title,ylabel="Final reference attained (%)",xscale="symlog")
    axs[0].legend(fontsize=8);labels(axs);mainfig(fig,"quality_cost","Quality versus recorded effort. Attempt, interval and Lloyd assignment costs differ; this is not an equal-time study. Parallel execution can affect wall times.")
    fig,ax=plt.subplots(figsize=(7,3));delta=(gm-gm[:,1,None]).T;scale=max(abs(delta).max(),.01);im=ax.imshow(delta,aspect="auto",cmap="RdBu_r",vmin=-scale,vmax=scale);ax.set(yticks=range(3),yticklabels=NAMES,xticks=np.arange(0,71,5),xticklabels=graphs[::5]);fig.colorbar(im,ax=ax,label="Difference (pp)");mainfig(fig,"graph_heatmap","Graphwise mean quality differences against random sweep; red favors the compared method.")
    fig,axs=plt.subplots(1,2,figsize=(7,3.1))
    for k,m in enumerate(NAMES):
        for j,ax in enumerate(axs):ax.plot(t,diag[j,k],color=COLORS[k],label=m)
    for ax,lab in zip(axs,("Euclidean inertia / n","Spherical inertia / n")):ax.set(xlabel="Active attempt",ylabel=lab)
    axs[0].legend(fontsize=8);labels(axs);mainfig(fig,"inertia","Candidate geometry for active trajectories only. Late KMeansNM means are conditional on not yet stopping.")
    fig,axs=plt.subplots(1,3,figsize=(7.5,3.0))
    with np.load(datadir/"density_sample.npz") as z:
        for k,(m,ax) in enumerate(zip(("gw","random-sweep","kmeansnm"),axs)):
            x=z[m];ax.hexbin(x[:,0],x[:,1],gridsize=28,mincnt=1,bins="log");ax.set(title=NAMES[k],xlabel="Spherical inertia reduction",ylabel="Candidate gain (pp)")
    labels(axs);mainfig(fig,"inertia_cut","Systematic subsample of candidate-level inertia reduction and change in reference attainment. Log density; exploratory, not independent observations.")
    fig,axs=plt.subplots(2,2,figsize=(7,5))
    for k,m in enumerate(NAMES):
        for ax,j in zip(axs.flat,(6,4,5,3)):ax.plot(t[1:],diag[j,k,1:],color=COLORS[k],label=m)
    for ax,lab in zip(axs.flat,("Strict record fraction","Candidate Hamming turnover","Projective angle (rad)","Minority fraction")):ax.set(xlabel="Active attempt",ylabel=lab)
    axs.flat[0].legend(fontsize=8);labels(axs);mainfig(fig,"diagnostics","Active trajectory diagnostics. Turnover and angle compare successive candidate partitions/normals, invariant to global label reversal. KMeansNM direction is the difference of its two centers.")
    fig,axs=plt.subplots(1,2,figsize=(7,3.2))
    for ax,key,title in zip(axs,("records","attempts"),("Strict incumbent records","Actual update attempts")):ax.bar(range(3),[r[key] for r in rows],color=COLORS);ax.set(xticks=range(3),xticklabels=NAMES,ylabel=title);ax.tick_params(axis="x",rotation=20)
    labels(axs);mainfig(fig,"records","Mean strict improvements and actual update counts. Initial assignment is excluded; carried-forward steps are not attempts.")
    for a in range(0,71,12):
        fig,axs=plt.subplots(4,3,figsize=(7,9))
        for j,ax in enumerate(axs.flat):
            if a+j>=71:ax.axis("off");continue
            for k,m in enumerate(NAMES):ax.plot(t,q[a+j,:,k].mean(axis=0),color=COLORS[k],label=m,lw=1)
            ax.axhline(100,color=".4",ls="--",lw=.7);ax.set_title(graphs[a+j])
            if j==0:ax.legend(fontsize=6)
        fig.supxlabel("Attempt");fig.supylabel("Reference attained (%)")
        mainfig(fig,f"graphs_{a//12+1}","Graphwise mean incumbent trajectories over 30 starts; graph-specific vertical scales and reference line at 100 percent.")
    # LaTeX source uses the shared template. Each dashboard occupies exactly one page.
    def esc(x):return str(x).replace("%",r"\%").replace("_",r"\_")
    def fmt(v):return f"{v:.3f}"
    tex=[]
    for k,r in enumerate(rows):
        tex.append(f"{NAMES[k]} & {fmt(r['mean'])} & {fmt(r['median'])} & {r['effect_pp']:+.3f} & [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] & {r['wins']}/{r['ties']}/{r['losses']} "+r"\\")
    (out/"results.tex").write_text("\n".join(tex)+"\n",encoding="utf8")
    winner=max(range(3),key=lambda k:rows[k]["mean"])
    (out/"findings.tex").write_text(r"\newcommand{\PrincipalFinding}{"+f"{NAMES[winner]} attains the highest mean final reference percentage ({rows[winner]['mean']:.3f}\\%). The other methods achieve "+", ".join(f"{NAMES[k]} {rows[k]['mean']:.3f}\\%" for k in range(3) if k!=winner)+".}\n"+r"\newcommand{\AuditCount}{"+f"{s['audited_records']:,}"+"}\n",encoding="utf8")
    lines=[]
    for j,(name,cap) in enumerate(figs):
        if j<8 and j%2==0:lines.append(r"\clearpage")
        if j>=8:lines.append(r"\clearpage")
        height=".37" if j<8 else ".82"
        lines.append(r"\resultfigure["+height+r"\textheight]{"+name+"}{"+esc(cap)+"}{}")
    (out/"main_figures.tex").write_text("\n".join(lines)+"\n",encoding="utf8")
    lines=[]
    for i,(name,title,mean,med) in enumerate(dash):
        if i:lines.append(r"\clearpage")
        lines.extend([r"\appendixheading{"+("PAIRED REPETITIONS" if i<90 else "AGGREGATE DASHBOARDS")+"}{"+title+"}{Best-known reference normalization}",
          r"\small Final across-graph mean: "+f"{mean:.3f}"+r"\% \quad Median: "+f"{med:.3f}"+r"\% \par",
          r"\begin{center}\includegraphics[width=\linewidth,height=.80\textheight,keepaspectratio]{figures/"+name+r".pdf}\end{center}"])
    (out/"dashboards.tex").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("Generated",len(dash),"dashboard pages and",len(figs),"main figures")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=16);a=p.parse_args();make_report(a.data,a.output,a.workers)
