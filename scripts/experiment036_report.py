"""E036: independent archive audit and publication figures, no new optimization."""
from pathlib import Path
import argparse, json, os, textwrap
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import awesomefigures as af
import pymupdf as fitz
from threadpoolctl import threadpool_limits
import experiment033_raw_hyperplane_sweep as e33
import experiment036_inertia_guided as ex
from experiment036_publication_audit import verify

ROOT=ex.ROOT/"full"
METHODS=("gw","random-sweep")+ex.ARMS
NAMES=dict(zip(METHODS,["Repeated GW","Random sweep","Spherical 2-means","Centroid sweep",
                        "Antipodal sweep","Inertia sweep","Hybrid sweep"]))
def audit_graph(g):
    # A worker owns one graph; all archive reads are immutable.
    with threadpool_limits(limits=1):
        return _audit_graph(g)

def _audit_graph(g):
    records=[]; seeds=[]; events=[]; curves=[]
    graph,y,rec=e33.inputs(g)
    b,_=e33.verify(g,"full",e33.STARTS,e33.STEPS)
    data=verify(g,"full",graph,y,rec,b)
    denominator=b["cuts"][:,0,0].mean()
    assert denominator>0
    for s in range(e33.STARTS):
        initial=b["cuts"][s,0,0]
        for arm in METHODS:
            if arm in METHODS[:2]:
                a=METHODS.index(arm)
                cs=b["cuts"][s,a];bs=b["best"][s,a]
                rows=[]
                for t in range(len(cs)):
                    x=np.unpackbits(b["labels"][s,a,t])[:len(y)]
                    gg=ex.geometry(y,x)[0]
                    r=b["normals"][s,a,t]
                    old=np.unpackbits(b["labels"][s,a,np.argmax(cs[:max(1,t)])])[:len(y)]
                    old_r=b["normals"][s,a,np.argmax(cs[:max(1,t)])]
                    h=np.mean(x!=old)
                    rows.append([t,cs[t],bs[t],b["seconds"][s,a,t] if t else 0,
                                 b["intervals"][s,a,t],*gg,min(h,1-h),
                                 np.arccos(np.clip(abs(r@old_r),0,1)),
                                 b["angles"][s,t] if a else np.nan,
                                 float(t>0 and bs[t]>bs[t-1]+ex.EPS),
                                 float(t>0 and bs[t]>bs[t-1]+ex.EPS),
                                 float(b["valid"][s,t]) if a else 1,3,0])
                d=np.asarray(rows)
            else:
                d,nn,xx,ev=data[s,arm]
                if len(ev):
                    f=pd.DataFrame(ev,columns=["iteration","theta","euclidean_inertia","spherical_inertia","cut"])
                    f["graph"]=g;f["seed"]=s;f["method"]=arm;events.append(f)
            f=pd.DataFrame(d,columns=ex.COLUMNS)
            f["graph"]=g;f["seed"]=s;f["method"]=arm
            f["gain_pct"]=100*(f.best-initial)/denominator
            f["candidate_gain_pct"]=100*(f.cut-initial)/denominator
            f["cumulative_seconds"]=f.seconds.cumsum()
            records.append(f)
            seeds.append(dict(graph=g,seed=s,method=arm,initial_cut=initial,
                              final_cut=d[-1,2],gain_pct=f.gain_pct.iloc[-1],
                              seconds=f.seconds.sum(),attempts=len(d)-1,records=int(f.cut_record.sum()),
                              intervals=int(f.intervals.iloc[1:].sum()),fallbacks=int(f.fallback.sum()),
                              invalid=int((f.valid==0).sum()),
                              best_inertia=float(f.loc[f.cut.idxmax(),"spherical_inertia"])))
            curves.append(dict(graph=g,seed=s,method=arm,curve=np.pad(f.gain_pct.to_numpy(),(0,101-len(f)),mode="edge")))
    print("Audited",g,flush=True)
    return records,seeds,events,curves

def main(workers=16):
    records=[]; seeds=[]; events=[]; curves=[]
    if workers < 1:
        raise ValueError("workers must be positive")
    available=len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else (os.cpu_count() or 1)
    effective=min(workers,available,len(e33.GRAPHS))
    print(f"Audit workers: requested={workers}, effective={effective}, available={available}",flush=True)
    def collect(results):
        for rr,ss,ee,cc in results:
            records.extend(rr);seeds.extend(ss);events.extend(ee);curves.extend(cc)
    # map preserves graph order, independent of execution order; aggregation
    # and PDF creation are single-writer. Spawn is safe on Windows.
    if effective==1:
        collect(map(audit_graph,e33.GRAPHS))
    else:
        with ProcessPoolExecutor(max_workers=effective) as pool:
            collect(pool.map(audit_graph,e33.GRAPHS,chunksize=1))
    df=pd.concat(records,ignore_index=True);seed=pd.DataFrame(seeds)
    ev=pd.concat(events,ignore_index=True)
    df.to_csv(ROOT/"iterations.csv",index=False,lineterminator="\n")
    seed.to_csv(ROOT/"per_seed.csv",index=False,lineterminator="\n")
    ev.to_csv(ROOT/"event_profiles.csv",index=False,lineterminator="\n")
    pg=seed.groupby(["graph","method"],sort=False).mean(numeric_only=True).reset_index()
    pg.to_csv(ROOT/"per_graph.csv",index=False,lineterminator="\n")
    gain=pg.pivot(index="graph",columns="method",values="gain_pct")[list(METHODS)]
    effects=gain.subtract(gain["random-sweep"],axis=0)
    rng=np.random.default_rng(3601); idx=rng.integers(0,len(gain),size=(10000,len(gain)))
    rows=[]
    for m in METHODS:
        e=effects[m].to_numpy();ci=np.quantile(e[idx].mean(axis=1),[.025,.975])
        sub=seed[seed.method==m];p=pg[pg.method==m]
        rows.append(dict(method=m,gain_pct=gain[m].mean(),advantage_pp=e.mean(),ci_low=ci[0],ci_high=ci[1],
                         wins=int((e>1e-9).sum()),ties=int((abs(e)<=1e-9).sum()),losses=int((e<-1e-9).sum()),
                         seconds=p.seconds.mean(),records=p.records.mean(),attempts=p.attempts.mean(),
                         intervals=p.intervals.mean(),invalid=int(sub.invalid.sum())))
    summary=pd.DataFrame(rows);summary.to_csv(ROOT/"method_summary.csv",index=False,lineterminator="\n")
    metadata=dict(complete=True,graphs=len(gain),paired_starts=len(gain)*e33.STARTS,steps=ex.STEPS,
                  audited_records=len(df),new_records=int(df.method.isin(ex.ARMS).sum()),
                  reused_records=int(df.method.isin(METHODS[:2]).sum()),new_eva_solves=0,
                  sampled_event_records=len(ev),methods=rows,
                  centroid_angle_audit="cosine absolute tolerance 1e-12; angle range and NaN checked",
                  computation_commit="007ebcc326797e60edb791f1ed34a53f85a20505",
                  audit_workers_requested=workers,audit_workers_effective=effective)
    (ROOT/"summary.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf8")
    make_figures(df,seed,pg,gain,effects,summary,curves,ev,metadata)

def make_figures(df,seed,pg,gain,effects,summary,curves,ev,metadata):
    af.set_style("heri",font="STIX",rasterize=True,use_tex=False)
    plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","pdf.fonttype":42,
                        "ps.fonttype":42,"axes.unicode_minus":False})
    font_manager.findfont("STIXGeneral",fallback_to_default=False)
    cols=af.color_palette("tol-bright")
    colors=dict(zip(METHODS,cols))
    figs=[]
    def save(fig,name,title,caption):
        path=ROOT/(name+".pdf")
        fig.tight_layout(pad=1.3)
        af.save_figure(fig,path,dpi=600)
        plt.close(fig)
        figs.append((path,title,caption))
    def labels(axes):
        for k,ax in enumerate(np.asarray(axes).flat):af.panel_label(ax,chr(65+k))
    def lines(ax,x,series):
        for m,z in series.items():ax.plot(x,z,label=NAMES[m],color=colors[m],linewidth=1.3)
    avg={m:np.mean([v["curve"] for v in curves if v["method"]==m],axis=0) for m in METHODS}
    fig,ax=plt.subplots(figsize=(7.5,4))
    lines(ax,np.arange(101),avg);ax.set(xlabel="Attempt (stopped Lloyd curves held constant)",ylabel="Mean cut gain (%)");ax.legend(ncol=2)
    save(fig,"convergence","Incumbent convergence","Equal-graph means of paired starts. The common initial GW cut is the zero reference; stopped Lloyd curves are held at the last incumbent. Every method has at most 100 attempts.")
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    q=summary.set_index("method").loc[list(METHODS)]
    yy=np.arange(len(q));ax.errorbar(q.advantage_pp,yy,xerr=[q.advantage_pp-q.ci_low,q.ci_high-q.advantage_pp],fmt="o",capsize=3)
    ax.axvline(0,color="grey",linestyle="--");ax.set(yticks=yy,yticklabels=[NAMES[m] for m in METHODS],xlabel="Advantage over random sweep (percentage points)");ax.invert_yaxis()
    save(fig,"paired_effects","Paired benchmark effects","Point estimates and 95% graph-bootstrap intervals from 10,000 equal-graph resamples. These intervals are descriptive and not multiplicity-adjusted.")
    fig,axes=plt.subplots(1,2,figsize=(8,4))
    for m in METHODS:
        p=q.loc[m];axes[0].scatter(p.seconds,p.gain_pct,color=colors[m],label=NAMES[m]);axes[1].scatter(p.intervals,p.gain_pct,color=colors[m])
    for ax in axes:ax.set_ylabel("Mean gain (%)");ax.set_xscale("symlog",linthresh=.001)
    axes[0].set_xlabel("Method seconds");axes[1].set_xlabel("Examined intervals");axes[0].legend(fontsize=7)
    labels(axes);save(fig,"quality_cost","Quality and computational effort","Archived E033 timings and new E036 timings are from different runs, so timing comparisons are descriptive, not a controlled hardware benchmark. Attempts and intervals are not equivalent units of cost.")
    fig,ax=plt.subplots(figsize=(8,5))
    im=ax.imshow(effects[list(ex.ARMS)].T,aspect="auto",cmap="RdBu_r",interpolation="nearest")
    ax.set(yticks=np.arange(len(ex.ARMS)),yticklabels=[NAMES[m] for m in ex.ARMS],xticks=np.arange(0,len(gain),5),xticklabels=gain.index[::5]);fig.colorbar(im,ax=ax,label="Advantage (pp)")
    save(fig,"graph_heatmap","Graphwise heterogeneity","Each column is one G-set graph; each row is a new method. Positive values indicate an improvement over the archived random-tangent sweep.")
    fig,axes=plt.subplots(1,2,figsize=(8,4))
    for m in METHODS:
        p=df[df.method==m].groupby("iteration")
        axes[0].plot(p.spherical_inertia.mean(),label=NAMES[m],color=colors[m])
        axes[1].plot(p.euclidean_inertia.mean(),color=colors[m])
    axes[0].set_ylabel("Spherical inertia / n");axes[1].set_ylabel("Euclidean inertia / n")
    for ax in axes:ax.set_xlabel("Active attempt")
    axes[0].legend(fontsize=7);labels(axes)
    save(fig,"inertia_trajectories","Geometry through iterations","Candidate-level means across active trajectories. Lloyd terminates early, so late active-only averages have selection effects; neither inertia metric is the original signed graph objective.")
    fig,axes=plt.subplots(2,3,figsize=(8,6.5))
    for ax,m in zip(axes.flat,ex.ARMS):
        p=df[(df.method==m)&(df.iteration>0)].copy()
        initial=df[(df.method==m)&(df.iteration==0)][["graph","seed","spherical_inertia"]].rename(columns={"spherical_inertia":"initial_inertia"})
        p=p.merge(initial,on=["graph","seed"]);p["reduction"]=p.initial_inertia-p.spherical_inertia
        ax.hexbin(p.reduction,p.candidate_gain_pct,gridsize=45,mincnt=1,bins="log",cmap="viridis")
        ax.set(title=NAMES[m],xlabel="Inertia reduction / n",ylabel="Candidate gain (%)")
    axes.flat[-1].axis("off");labels(list(axes.flat)[:-1])
    save(fig,"inertia_cut_density","Geometry versus actual cut","Each hexbin represents candidate partitions relative to their own initial geometry and cut. Density is logarithmic. Cross-graph pooling can confound correlations; this figure is exploratory.")
    fig,axes=plt.subplots(2,2,figsize=(8,6))
    for m in METHODS:
        p=df[(df.method==m)&(df.iteration>0)].groupby("iteration")
        for ax,k in zip(axes.flat,["cut_record","turnover","projective_angle","balance"]):
            ax.plot(p[k].mean(),label=NAMES[m],color=colors[m])
    for ax,k in zip(axes.flat,["Strict record probability","Hamming turnover","Projective angle (rad)","Minority-side fraction"]):
        ax.set(xlabel="Active attempt",ylabel=k)
    axes.flat[0].legend(fontsize=6,ncol=2);labels(axes)
    save(fig,"search_diagnostics","Search diagnostics","Record rates, candidate turnover, projective normal angles and cluster balance. Global label reversal is ignored for the Hamming and angular metrics.")
    fig,axes=plt.subplots(1,2,figsize=(8,4))
    axes[0].bar(np.arange(len(METHODS)),q.records,color=[colors[m] for m in METHODS])
    axes[1].bar(np.arange(len(METHODS)),q.attempts,color=[colors[m] for m in METHODS])
    for ax in axes:ax.set_xticks(np.arange(len(METHODS)),[NAMES[m] for m in METHODS],rotation=60,ha="right")
    axes[0].set_ylabel("Mean strict records");axes[1].set_ylabel("Mean attempted updates");labels(axes)
    save(fig,"records","Improvement frequency","A record is a strict increase in the best original signed cut, not a decrease of inertia. The Lloyd arm stops on geometric stagnation.")
    for g in ex.e33.PILOT:
        fig,axes=plt.subplots(2,2,figsize=(8,6))
        for ax,m in zip(axes.flat,ex.ARMS[1:]):
            p=ev[(ev.graph==g)&(ev.method==m)&(ev.iteration==1)]
            ax2=ax.twinx()
            ax.step(p.theta,p.cut,where="mid",color="tab:blue",linewidth=1)
            ax2.plot(p.theta,p.spherical_inertia,color="tab:orange",linewidth=1)
            ax.set(title=NAMES[m],xlabel=r"Rotation $\theta$",ylabel="Signed cut")
            ax2.set_ylabel("Spherical inertia / n")
        labels(axes);save(fig,"event_"+g,"Event landscapes: "+g,"First rotation, paired seed zero. Blue: signed cut; orange: spherical partition inertia. Each point corresponds to an open event interval. Event profiles are sampled diagnostics, not independent observations.")
    for first in range(0,len(e33.GRAPHS),12):
        fig,axes=plt.subplots(4,3,figsize=(8,10.5))
        for j,ax in enumerate(axes.flat):
            if first+j>=len(e33.GRAPHS):ax.axis("off");continue
            g=e33.GRAPHS[first+j]
            for m in METHODS:
                z=np.mean([v["curve"] for v in curves if v["graph"]==g and v["method"]==m],axis=0)
                ax.plot(z,color=colors[m],linewidth=.85,label=NAMES[m])
            ax.set_title(g)
            if j==0:ax.legend(fontsize=5,ncol=2)
        fig.supxlabel("Attempt");fig.supylabel("Mean cut gain (%)")
        save(fig,f"graph_curves_{first//12+1}","Graphwise convergence","All 71 graph-specific curves, averaging 16 paired starts. Vertical scales are graph-specific; stopped Lloyd histories are held constant.")
    # Standalone appendix: A4 text and figure pages. Compose PDF pages without
    # flattening existing vector figures. No historical hard-coded findings.
    doc=fitz.open()
    def page(title,body,figure=None):
        p=doc.new_page(width=595,height=842)
        # Create text with Matplotlib's bundled STIXGeneral, not a platform font.
        from io import BytesIO
        f=plt.figure(figsize=(595/72,842/72))
        f.text(.075,.958,"MAX-CUT  /  EXPERIMENT 036",fontsize=10,weight="bold")
        f.text(.925,.958,"NUMERICAL APPENDIX",ha="right",fontsize=9)
        f.text(.075,.904,title,fontsize=21,weight="bold")
        ypos=.86
        for line in body:
            for l in textwrap.wrap(line,width=98):
                f.text(.075,ypos,l,fontsize=11);ypos-=.019
            ypos-=.012
        f.text(.075,.035,"Fixed EVA factors  |  Original signed cuts",fontsize=9)
        f.text(.925,.035,str(len(doc)),ha="right",fontsize=9)
        buf=BytesIO();f.savefig(buf,format="pdf");plt.close(f)
        with fitz.open(stream=buf.getvalue(),filetype="pdf") as src:p.show_pdf_page(p.rect,src,0)
        if figure:
            with fitz.open(figure) as src:p.show_pdf_page(fitz.Rect(35,842*(1-ypos)+12,560,790),src,0,keep_proportion=True)
    top=summary.loc[summary.gain_pct.idxmax()]
    page("Geometry-guided hyperplane rounding",[
        "71 G-set graphs; 16 paired starts each; 100 attempted updates per sweep arm. All methods share the initial GW normal. The EVA factors are frozen; no new SDP solve or discrete local refinement.",
        f"Audited {metadata['audited_records']:,} candidate records, including {metadata['reused_records']:,} archived E033 records. Best equal-graph mean: {NAMES[top.method]}, {top.gain_pct:.3f}% above the initial GW baseline.",
        "The primary estimand is the paired gain versus the 100-attempt random sweep. Each graph has equal weight. Percentages use the graph's mean initial GW cut as denominator.",
        "Centroid, antipodal and hybrid sweeps retain the best signed cut; spherical Lloyd and the inertia-only sweep follow geometric descent while independently retaining the best cut encountered.",
        "This fixed-budget experiment complements E035's adaptive stopping study. It does not prove global optimality, dominance for every graph, or a stronger approximation guarantee."
    ])
    lines=["Method: mean gain; difference versus random sweep [95% graph-bootstrap interval]; W/T/L."]
    for row in summary.itertuples():
        lines.append(f"{NAMES[row.method]}: {row.gain_pct:.3f}%; {row.advantage_pp:+.3f} pp [{row.ci_low:+.3f}, {row.ci_high:+.3f}]; {row.wins}/{row.ties}/{row.losses}.")
    lines.append("Intervals are descriptive, not adjusted for multiple comparisons. Historical baseline timings are not a controlled equal-time comparison.")
    page("Paired numerical results",lines)
    for path,title,caption in figs:page(title,[caption],path)
    out=ROOT/"Experiment036_Numerical_Appendix.pdf"
    doc.set_metadata({"title":"Experiment036 | Geometry-guided hyperplane sweep"})
    doc.save(out,garbage=4,deflate=True);doc.close()
    with fitz.open(out) as proof:
        assert len(proof)==len(figs)+2 and all(len(p.get_text())>30 for p in proof)
        fonts=sorted({f[3] for p in proof for f in p.get_fonts(full=True)})
        assert any("STIX" in f.upper() for f in fonts)
        (ROOT/"font_audit.json").write_text(json.dumps(dict(pages=len(proof),fonts=fonts,mathtext="stix",dpi=600),indent=2)+"\n")
    report="# Experiment036: geometry-guided hyperplane sweep\n\n[Download the numerical appendix](Experiment036_Numerical_Appendix.pdf).\n\n71 graphs, 16 paired starts, fixed 100 attempts, no EVA solves. Archived baselines reused. All archived cuts independently rescored.\n\n"
    report+="| Method | Mean gain (%) | Advantage (pp) | 95% graph-bootstrap interval | W/T/L |\n|---|---:|---:|---|---|\n"
    for r in summary.itertuples():report+=f"| {NAMES[r.method]} | {r.gain_pct:.3f} | {r.advantage_pp:+.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.wins}/{r.ties}/{r.losses} |\n"
    (ROOT/"report.md").write_text(report,encoding="utf8",newline="\n")
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=16)
    args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.workers)
