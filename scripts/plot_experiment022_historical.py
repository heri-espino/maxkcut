"""Regenerate retrospective historical-method figures from Experiment022 CSVs.

Run: python scripts/plot_experiment022_historical.py
Does not perform any graph optimization.
"""
from __future__ import annotations

import csv
from pathlib import Path
import matplotlib.pyplot as plt

DATA = Path("results/gset/experiment022-graph-structure-methods")


def rows(path):
    with path.open(encoding="utf8",newline="") as f:
        return list(csv.DictReader(f))


def plot_raw_family_wins(root=DATA):
    rr=rows(root/"historical_raw_rounding_by_family.csv")
    families=["random","planar","toroidal"]
    methods=["GW/FJ_EVA","KMeans2N_EVA","KMeansNM_EVA"]
    labels={"GW/FJ_EVA":"GW/FJ hyperplane","KMeans2N_EVA":"Independent 2-means","KMeansNM_EVA":"Antipodal 2-means"}
    fig,ax=plt.subplots(figsize=(10,5.5))
    for j,m in enumerate(methods):
        vals=[int(next(x["shared_best"] for x in rr if x["family"]==fam and x["method"]==m)) for fam in families]
        bars=ax.bar([i+(j-1)*.25 for i in range(3)],vals,width=.23,label=labels[m])
        ax.bar_label(bars,padding=3,fontsize=10)
    ax.set_xticks(range(3),["Random (27)","Planar family (23)","Toroidal (12)"])
    ax.set_ylabel("Graphs achieving the best of three raw-rounding scores (ties counted)")
    ax.set_title("Early 20-restart EVA rounding: structure-stratified observed wins")
    ax.legend(loc="upper left",ncol=3,fontsize=9)
    ax.set_ylim(0,30)
    fig.text(.09,.02,"62 saved graphs; no flip/KL/ILS. Not comparable to Experiment021's 50-start full pipelines.",fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1))
    out=root/"historical_raw_rounding_family_wins.svg"
    fig.savefig(out,format="svg")
    plt.close(fig)
    return out


def plot_g63_historical(root=DATA):
    rr=rows(root/"historical_g63_methods.csv")
    def lookup(experiment,method):
        group=[r for r in rr if r["experiment"]==experiment and r["method"]==method]
        if len(group)!=1:
            raise ValueError(f"Missing or duplicate historical G63 method: {experiment}/{method}")
        return float(group[0]["best_cut"])
    selection=[
       ("GW hyperplane · 20 starts","008","FJ_EVA"),
       ("Antipodal K-means · 20 starts","008","K-MeansNM_EVA"),
       ("MST multiedge + flips · 24 variants","009","MST-multi-edge-l2"),
       ("K-means + flips · 1,000 starts","008","K-MeansNM-Flip_EVA"),
       ("Cut-aware K-means + KL","012","cut-aware/CutAwareKMeansKL"),
       ("Mean-start EM + KL · 4 episodes","016","EM-mean-em-kl"),
       ("Multi-group k=4 + KL · 2 starts","017","multigroup-k4-paired"),
       ("Mean-perturb init + KL · 3 starts","018","kmeans-init-pilot-mean-perturb"),
       ("Farthest-pair + ILS · 50 starts","019","full-pipeline-farthest-pair"),
       ("Saved K-means + flips + KL + ILS","013","SavedKMeansFlip+KL+ILS"),
       ("Elite-start EM + CEM · inherited best","016","EM-elite-em-cem"),
    ]
    names=[a for a,_,_ in selection]
    vals=[lookup(e,m) for _,e,m in selection]
    fig,ax=plt.subplots(figsize=(11.2,6.4))
    bars=ax.barh(range(len(names)),[v-25000 for v in vals],left=25000)
    ax.set_yticks(range(len(names)),names,fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(25000,27150)
    ax.set_xlabel("Best observed feasible G63 signed cut (not a common evaluation budget)")
    ax.set_title("G63 historical experiments: method and budget are confounded")
    for bar,v in zip(bars,vals):
        ax.text(v+16,bar.get_y()+bar.get_height()/2,f"{v:,.0f}",va="center",fontsize=8)
    fig.text(.08,.012,"CAUTION: EM elite 26,917 is retained from the input; the algorithm did not improve it.\nG63 alone cannot establish topology-dependent method dominance.",fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1))
    out=root/"historical_g63_method_bests.svg"
    fig.savefig(out,format="svg")
    plt.close(fig)
    return out


if __name__=="__main__":
    for file in (plot_raw_family_wins(),plot_g63_historical()):
        print(file)