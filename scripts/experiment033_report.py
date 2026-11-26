"""Produce the final Experiment033 report appendix and 71 graph learning curves."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from experiment033_raw_hyperplane_sweep import ROOT, GRAPHS, STARTS, STEPS, verify

def table(path, rows):
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    root=ROOT/"full"
    curves=[]; per=[]; trace=[]; timings=[]; counts=[]
    rng=np.random.default_rng(3301)
    for g in GRAPHS:
        d,meta=verify(g,"full",STARTS,STEPS)
        b=d["best"]; c=d["cuts"]
        baseline=float(b[:,0,0].mean())
        if baseline<=0:
            raise ValueError(f"{g}: nonpositive mean baseline; normalization must be revisited")
        # Normalize by each graph's mean initial GW cut, not a certified optimum.
        normalized=100*(b-b[:,0,0][:,None,None])/baseline
        curves.append(normalized.mean(axis=0))
        timings.append(np.cumsum(d["seconds"],axis=2).mean(axis=0))
        counts.append(np.cumsum(d["intervals"],axis=2).mean(axis=0))
        effect=normalized[:,1,-1]-normalized[:,0,-1]
        per.append(dict(graph=g,vertices=meta["vertices"],rank=meta["rank"],
                        initial_gw_mean=baseline,gw_best_mean=float(b[:,0,-1].mean()),
                        sweep_best_mean=float(b[:,1,-1].mean()),
                        gw_gain_pct=float(normalized[:,0,-1].mean()),
                        sweep_gain_pct=float(normalized[:,1,-1].mean()),
                        paired_gain_pct=float(effect.mean()),
                        paired_wins=int(np.sum(effect>1e-9)),paired_ties=int(np.sum(np.abs(effect)<=1e-9)),
                        paired_losses=int(np.sum(effect<-1e-9)),
                        gw_seconds=float(timings[-1][0,-1]),sweep_seconds=float(timings[-1][1,-1]),
                        mean_sweep_intervals=float(counts[-1][1,-1]),
                        numerical_fallbacks=int((~d["valid"]).sum())))
        for s in range(STARTS):
            for t in range(STEPS+1):
                trace.append(dict(graph=g,restart=s,iteration=t,gw_candidate=c[s,0,t],
                                  gw_best=b[s,0,t],sweep_candidate=c[s,1,t],sweep_best=b[s,1,t],
                                  gw_seconds=d["seconds"][s,0,t],sweep_seconds=d["seconds"][s,1,t],
                                  sweep_intervals=d["intervals"][s,1,t],sweep_valid=bool(d["valid"][s,t])))
        print("Verified",g,flush=True)
    table(root/"per_graph.csv",per)
    table(root/"iterations.csv",trace)
    a=np.array(curves); t=np.array(timings); count=np.array(counts)
    delta=a[:,1]-a[:,0]
    ids=rng.integers(0,len(GRAPHS),size=(10000,len(GRAPHS)))
    ci=np.quantile(delta[ids,-1].mean(axis=1),[.025,.975])
    final=delta[:,-1]
    summary=dict(complete=True,graphs=len(GRAPHS),starts=STARTS,steps=STEPS,
                 new_eva_solves=0,paired_trajectories=len(GRAPHS)*STARTS,
                 candidate_records=len(GRAPHS)*STARTS*2*(STEPS+1),
                 mean_gw_gain_pct=float(a[:,0,-1].mean()),
                 mean_sweep_gain_pct=float(a[:,1,-1].mean()),
                 mean_paired_gain_pct=float(final.mean()),
                 graph_bootstrap_95_pct=[float(x) for x in ci],
                 graph_wins=int((final>1e-9).sum()),graph_ties=int((abs(final)<=1e-9).sum()),
                 graph_losses=int((final<-1e-9).sum()),
                 numerical_fallbacks=sum(x["numerical_fallbacks"] for x in per),
                 note="Same Gaussian proposal count, not equal time or evaluated intervals; no downstream refinement.")
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    x=np.arange(STEPS+1)
    fig,ax=plt.subplots(figsize=(7.2,4))
    for arm,name in enumerate(["Repeated random GW","Random-tangent sweep"]):
        ax.plot(x,a[:,arm].mean(axis=0),label=name)
    ax.set(xlabel="Additional Gaussian draws / sweeps",ylabel="Mean gain from initial GW (%)",
           title="All 71 graphs: best-so-far cut quality")
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(root/"convergence.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4))
    for arm,name in enumerate(["Repeated random GW","Random-tangent sweep"]):
        ax.plot(t[:,arm].mean(axis=0),a[:,arm].mean(axis=0),label=name)
    ax.set(xlabel="Mean cumulative measured algorithm time (s)",ylabel="Mean gain from initial GW (%)",
           title="Observed quality versus average cumulative cost")
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(root/"time_quality.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4))
    order=np.argsort(final)
    ax.bar(np.arange(len(final)),final[order])
    ax.axhline(0,color="black",linewidth=.7)
    ax.set_xticks(np.arange(len(final))[::4],[GRAPHS[j] for j in order[::4]],rotation=90)
    ax.set(ylabel="Sweep minus repeated GW (percentage points)",title="Graph-wise final paired gains")
    fig.tight_layout();fig.savefig(root/"graph_gains.pdf");plt.close(fig)
    with PdfPages(root/"all_71_curves.pdf") as pdf:
        for first in range(0,len(GRAPHS),12):
            fig,axes=plt.subplots(4,3,figsize=(8.27,11.69))
            for j,ax in enumerate(axes.flat):
                k=first+j
                if k>=len(GRAPHS):ax.axis("off");continue
                ax.plot(x,a[k,0],label="GW")
                ax.plot(x,a[k,1],label="Sweep")
                ax.set_title(GRAPHS[k]);ax.grid(alpha=.15)
                if j==0:ax.legend(fontsize=7)
                if j>=9:ax.set_xlabel("Iteration")
            fig.supylabel("Gain from initial GW (%)")
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    md = f"""# Experiment033: raw hyperplane rounding

71 frozen EVA factors; 16 paired starts per graph; 100 random-tangent sweeps or 100 fresh GW draws; no Flip, KL, ILS, K-means or EVA solves.

- Mean gain relative to graph-specific initial GW: repeated GW {summary['mean_gw_gain_pct']:.4f}%, sweep {summary['mean_sweep_gain_pct']:.4f}%.
- Equal-graph mean final sweep-minus-GW difference: {summary['mean_paired_gain_pct']:+.4f} percentage points.
- Descriptive graph-bootstrap 95% interval: [{ci[0]:.4f}, {ci[1]:.4f}] percentage points.
- Graph wins/ties/losses: {summary['graph_wins']}/{summary['graph_ties']}/{summary['graph_losses']}.
- Independently rescored candidate records: {summary['candidate_records']}.
- Numerical fallback count: {summary['numerical_fallbacks']}.

The curves average seed-level incumbent gains normalized by each graph's positive mean initial GW cut, then give equal weight to each graph. The time plot uses mean cumulative measured time, not a matched-time experiment. Timing excludes setup, warmup, RNG and external verification. One sweep explores many event intervals; equal draws do not imply equal work. No certified optimum or improved approximation ratio is claimed.

See convergence.pdf, time_quality.pdf, graph_gains.pdf, all_71_curves.pdf, per_graph.csv and iterations.csv. The exact original signed partitions, normals, timestamps and source hashes are retained in graphs/.
"""
    (root/"report.md").write_text(md,encoding="utf8")
    # Standalone appendix, suitable for appending after the existing illustrated explainer.
    tex=r"""\documentclass[11pt]{article}
\usepackage[margin=25mm]{geometry}
\usepackage{graphicx,amsmath,booktabs,pdfpages}
\begin{document}
\section*{Numerical experiment: raw Hyperplane Sweep versus GW}
We use all 71 canonical G-set graphs, with one fixed archived EVA embedding per graph and 16 shared initial GW normals. At each of 100 subsequent steps, the two methods receive the same fresh Gaussian draw. GW uses its normalized direction as a new independent hyperplane; Hyperplane Sweep projects the draw perpendicular to its current incumbent and searches the entire resulting semicircle. Both retain their best cut so far. No EVA, clustering or discrete local-search refinement is performed.
For graph \(g\), restart \(s\), method \(m\), and step \(t\), define \(B_{gms}(t)\) as the best cut found so far and \(\bar C_g\) as the mean initial GW cut. The plotted gain is
\[
G_m(t)=\frac{100}{71\cdot16}\sum_{g,s}
 \frac{B_{gms}(t)-B_{gms}(0)}{\bar C_g}.
\]
""" + (r"""
After 100 steps, repeated GW improved by %.4f\%% and the sweep by %.4f\%%.
The difference is %+.4f percentage points (descriptive graph-bootstrap 95\%% interval [%.4f, %.4f]).
The sweep wins/ties/loses on %d/%d/%d graphs by mean final cut.
All %d candidate scores were independently rescored using the original signed graph edges.
\begin{center}
\includegraphics[width=.98\linewidth]{convergence.pdf}
\end{center}
\clearpage
\section*{Computational cost and variation among graphs}
Equal numbers of Gaussian draws do not imply equal computational work: a sweep examines many event intervals, while GW samples just one partition. The time plot averages measured cumulative algorithm times and is not an equal-time experiment. Compilation, random number generation, archive I/O and external verification are excluded. The following pages show individual learning curves for every graph. These are empirical results, not a certificate of global optimality.
\begin{center}
\includegraphics[width=.90\linewidth]{time_quality.pdf}

\includegraphics[width=.90\linewidth]{graph_gains.pdf}
\end{center}
\includepdf[pages=-]{all_71_curves.pdf}
\end{document}
""" % (summary["mean_gw_gain_pct"],summary["mean_sweep_gain_pct"],summary["mean_paired_gain_pct"],ci[0],ci[1],summary["graph_wins"],summary["graph_ties"],summary["graph_losses"],summary["candidate_records"]))
    (root/"appendix.tex").write_text(tex,encoding="utf8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
