"""Regenerate Experiment022 family-effect plots from archived CSVs.

Requires Matplotlib. No graph search, EVA solver, or additional restart is run.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results",type=Path,default=Path("results/gset/experiment022-graph-structure-methods"))
    args=ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    counts=read_csv(args.results/"method_by_family.csv")
    methods=["GW screen-16","Sample antipodal","Farthest pair","Spherical k-means++","Mean perturbation","Independent centers","PCA antipodal","Sample pair","Original K-MeansNM"]
    keys=["gw-screen16","sample-antipodal","farthest-pair","spherical-pp","mean-perturb","independent","pca-antipodal","sample-pair","paired"]
    families=["random","planar","toroidal"]
    lookup={(r["method"],r["family"]):r for r in counts}
    arr=np.array([[100*float(lookup[m,f]["shared_wins"])/float(lookup[m,f]["graphs"]) for f in families] for m in keys])
    fig,ax=plt.subplots(figsize=(10.4,6.4))
    im=ax.imshow(arr,aspect="auto",cmap="cividis",vmin=0,vmax=70)
    for i,m in enumerate(keys):
        for j,f in enumerate(families):
            r=lookup[m,f]
            color="black" if arr[i,j]>44 else "white"
            ax.text(j,i,f'{int(r["shared_wins"])}/{int(r["graphs"])}',ha='center',va='center',color=color,fontweight='bold',fontsize=12)
    ax.set_xticks(range(len(families)),['Random (30)','Planar (24)','Toroidal (17)'],fontsize=12)
    ax.set_yticks(range(len(keys)),methods,fontsize=11)
    ax.set_title('Who finds the best Max-Cut? — G-set topology',fontsize=16,loc='left',pad=16)
    bar=fig.colorbar(im,ax=ax,pad=0.02);bar.set_label('Fraction of graphs with best observed cut (%)')
    ax.set_xlabel('Published G-set generator family')
    fig.text(.05,.015,'Experiment021: 9 methods × 50 restarts × 71 graphs. Best-cut ties count for every tied method.\nNot an equal-work comparison; GW screen-16 evaluates 16 additional initial cuts per restart.',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1))
    out=args.results/"topology_best_cut_heatmap.svg"
    fig.savefig(out,format="svg",bbox_inches="tight")
    plt.close(fig)
    # Out-of-sample retrospective policy comparison
    loo=read_csv(args.results/"leave_one_graph_out.csv")
    rows=[]
    for allowed in [1,0]:
        for policy in ["global","family"]:
            chosen=[x for x in loo if int(x["screen16_allowed"])==allowed and x["policy"]==policy]
            rows.append((f'{policy.capitalize()}'+(' · nine methods' if allowed else ' · no GW screen-16'),
                         sum(float(x["regret"]) for x in chosen)/len(chosen),
                         sum(float(x["gain"]) for x in chosen)/len(chosen)))
    fig,ax=plt.subplots(figsize=(10.2,4.4))
    x=np.arange(len(rows))
    heights=[r[1] for r in rows]
    bars=ax.barh(x,heights,color=plt.cm.cividis(np.linspace(0.2,0.8,len(x))))
    for bar,row in zip(bars,rows):
        ax.text(bar.get_width()+0.04,bar.get_y()+bar.get_height()/2,f'{row[1]:.3f}',va='center',fontsize=11)
    ax.set_yticks(x,[r[0] for r in rows]);ax.invert_yaxis()
    ax.set_xlim(0,max(heights)*1.15)
    ax.set_xlabel('Average held-out regret (cut units; lower is better)')
    ax.set_title('Leave-one-graph-out method selection: topology does not help yet',loc='left',fontsize=14,pad=14)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.04,.01,'Retrospective selection from the 50 saved restarts per method; no new compute-budget-matched searches.',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1))
    fig.savefig(args.results/"topology_policy_loo.svg",format="svg",bbox_inches="tight")
    plt.close(fig)
    print(out)

if __name__=="__main__":
    main()