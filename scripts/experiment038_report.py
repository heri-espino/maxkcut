"""Report E038 compact data only: no optimization or repeated audit."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import pymupdf as fitz
from experiment037_report import style,save,verify_page,labels,legend,reference,NAMES,COLORS,STYLES
import matplotlib.pyplot as plt

def read(path):
    with path.open(newline="",encoding="utf8") as f:return list(csv.DictReader(f))
def main(data,out):
    data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fdir=out/"figures";fdir.mkdir(exist_ok=True)
    summary=json.loads((data/"summary.json").read_text());assert summary["complete"] and summary["graphs"]==71
    p=np.array(summary["patiences"]);methods=("gw","random-sweep","kmeansnm")
    rows=read(data/"patience_summary.csv");pg=read(data/"per_graph_policy.csv")
    graphs=list(dict.fromkeys(r["graph"] for r in pg));assert len(graphs)==71
    with np.load(data/"curves.npz") as f:z={k:f[k] for k in f.files}
    assert z["mean"].shape==(71,3,summary["cap"]+1)
    cap=summary["cap"];t=np.arange(cap+1);end=max(int(r["stop"]) for r in read(data/"per_start_policy.csv") if int(r["patience"])==300)
    t=t[:end+1];pages=[]
    def new(title):
        style();fig,axs=plt.subplots(3,2,figsize=(7,9.1));fig.suptitle(title,fontsize=15,y=.96);return fig,axs
    def emit(fig,axs,name,title):
        labels(axs);path=fdir/(name+".pdf");save(fig,path)
        with fitz.open(path) as doc:assert len(doc)==1;verify_page(doc[0],name)
        pages.append((path,title))
    title="Patience sensitivity | 71 G-set graphs, 30 starts";fig,axs=new(title)
    for k,m in enumerate(methods):
        rr=[r for r in rows if r["method"]==m]
        assert [int(r["patience"]) for r in rr]==list(p)
        for ax,key,lab in zip(axs.flat,("mean_quality","marginal_pp","mean_attempts","mean_seconds","mean_intervals","mean_gain_from_demo"),("Final quality (%)","Marginal gain (pp)","Attempts","Search seconds","Evaluated intervals / assignments","Change vs 100-iteration demo (pp)")):
            v=np.array([float(r[key]) for r in rr]);ax.plot(p,v,color=COLORS[k],ls=STYLES[k],marker=".",label=NAMES[k]);ax.set(xlabel="Patience",ylabel=lab)
            if key=="marginal_pp":
                ax.fill_between(p,[float(r["ci_low"]) for r in rr],[float(r["ci_high"]) for r in rr],color=COLORS[k],alpha=.12)
    legend(axs.flat[0]);emit(fig,axs,"patience",title)
    title="Adaptive convergence | patience 300";fig,axs=new(title)
    for k,m in enumerate(methods):
        for ax,key,fn in ((axs.flat[0],"mean",np.mean),(axs.flat[1],"median",np.median)):
            ax.plot(t,fn(z[key][:,k,:end+1],axis=0),color=COLORS[k],ls=STYLES[k],label=NAMES[k]);ax.set(xlabel="Attempt",ylabel="Best-known cut attained (%)");reference(ax)
        h=z["hazard"][:,k].sum(axis=0);good=h[:,0]>0
        axs.flat[2].plot(np.flatnonzero(good),h[good,1]/h[good,0],color=COLORS[k],ls=STYLES[k],label=NAMES[k]);axs.flat[2].set(xlabel="Prior failures",ylabel="Conditional record probability")
        axs.flat[3].plot(np.flatnonzero(good),h[good,0],color=COLORS[k],ls=STYLES[k]);axs.flat[3].set(xlabel="Prior failures",ylabel="At-risk attempts",yscale="log")
        for ax,ids in ((axs.flat[4],np.arange(36)),(axs.flat[5],np.arange(36,71))):
            ax.plot(range(len(ids)),z["mean"][ids,k,end],color=COLORS[k],ls=STYLES[k],marker=".",ms=3,lw=.85)
            ax.set_xticks(range(len(ids)),[graphs[i] for i in ids],rotation=90,fontsize=6);ax.set(ylabel="Best-known cut attained (%)");reference(ax)
    legend(axs.flat[0]);emit(fig,axs,"convergence",title)
    for i in range(0,71,6):
        title=f"{graphs[i]} to {graphs[min(i+5,70)]} | mean and middle 50%";fig,axs=new(title)
        for j,ax in enumerate(axs.flat):
            idx=i+j
            if idx>=71:ax.set_visible(False);continue
            for k,m in enumerate(methods):
                ax.plot(t,z["mean"][idx,k,:end+1],color=COLORS[k],ls=STYLES[k],label=NAMES[k])
                ax.fill_between(t,z["low"][idx,k,:end+1],z["high"][idx,k,:end+1],color=COLORS[k],alpha=.12,linewidth=0)
            reference(ax);ax.set(title=graphs[idx],xlabel="Attempt",ylabel="Best-known cut attained (%)")
            if j==0:legend(ax)
        emit(fig,axs,f"graphs_{i//6+1:02d}",title)
    # A compact methods/interpretation page before results, within same PDF.
    title="Experiment038 | methods and interpretation";style();fig=plt.figure(figsize=(7,9.1));fig.suptitle(title,fontsize=15,y=.96)
    lines=["71 canonical G-set graphs; 30 matched E037 starts per graph.",
           "All cuts scored with original signed edge weights; frozen EVA factors.",
           "Repeated GW: independent Gaussian hyperplanes, retain best cut.",
           "Random Sweep: Gaussian tangent at incumbent, exact semicircle search;",
           "retain strictly improving normal and best cut.",
           "KMeansNM: antipodal initialization, spherical Lloyd assignments;",
           "retain best signed cut; stop at stable labels or patience.",
           "Patience counts consecutive real attempts without a strict cut record.",
           "P = 300 is primary; smaller policies are first hits on the same path.",
           f"Safety cap: {cap} attempts; cap-hit trajectories are censored.",
           "Stopped curves are held constant only for plotting, not extra attempts.",
           "Quality = 100 × cut / archived best-known cut; not necessarily optimum.",
           "Equal starts and attempt limits do not imply equal computational cost.",
           "Marginal gain is change from preceding patience; first point is zero.",
           "Intervals are nominal 95% paired graph-bootstrap intervals.",
           "Record hazard uses active attempts only; report at-risk denominators.",
           "See compact CSVs for all paired contrasts, cap counts and timings.",
           f"New independently rescored records: {summary['new_audited_records']:,}."]
    for j,line in enumerate(lines):fig.text(.075,.88-j*.039,line,fontsize=10.5,va="top")
    path=fdir/"methods.pdf";save(fig,path);pages.insert(0,(path,title))
    target=out/"Experiment038_Dashboard_Appendix.pdf"
    with fitz.open() as combined:
        for i,(path,title) in enumerate(pages):
            with fitz.open(path) as d:verify_page(d[0],path.name);combined.insert_pdf(d)
        combined.set_toc([[1,title,i+1] for i,(_,title) in enumerate(pages)])
        combined.save(target,garbage=4,deflate=True)
    with fitz.open(target) as d:
        assert len(d)==15
        for i,page in enumerate(d):verify_page(page,i)
        prev=out/"previews";prev.mkdir(exist_ok=True)
        for i in (0,1,2,3,14):d[i].get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(prev/f"page-{i+1}.png")
    print("Verified 15 pages, all 71 graph panels",flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--output",required=True);a=ap.parse_args();main(a.data,a.output)
