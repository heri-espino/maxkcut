"""Experiment018: compare 9 ways to initialize 2-centroid spherical KMeansNM.

All variants use the SAME fixed original EVA Y and original signed graph.
Paired antipodal = exact historical KMeansNM baseline. Every method gets
identical Lloyd iteration budgets, exact 2-group signed cut evaluation,
same number of restarts and matched post-clustering generalized KL passes.

Computation-heavy gw-screen16 spends 16 extra graph-cut evaluations at
initialization; it is labeled as such rather than called a free gain.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .eva_kmeans_initializers import INIT2
from .eva_multigroup_kmeans_cli import run_graph
from .eva_cli import _git_push
from .sweep import discover_graphs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs",nargs="+",default=["G63"])
    p.add_argument("--gset-dir",type=Path,default=Path("data"))
    p.add_argument("--factor-dir",type=Path,default=Path(
        "results/gset/experiment003-eva-numba",
    ))
    p.add_argument("--output-dir",type=Path,default=Path(
        "results/gset/experiment018-kmeans-initializations",
    ))
    p.add_argument("--methods",nargs="+",choices=INIT2,
                   default=list(INIT2))
    p.add_argument("--restarts",type=int,default=6)
    p.add_argument("--start-restart",type=int,default=0)
    p.add_argument("--chunk-size",type=int,default=2)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--max-iter",type=int,default=60)
    p.add_argument("--top-merges",type=int,default=2)
    p.add_argument("--kl-passes",type=int,default=30)
    p.add_argument("--rounding-seed",type=int,default=2026)
    p.add_argument("--kl-seed",type=int,default=2026)
    p.add_argument("--target",type=float,default=27047.)
    p.add_argument("--push",action="store_true")
    p.add_argument("--push-branch",default=None)
    args=p.parse_args()
    if args.push and not args.push_branch:
        p.error("--push needs --push-branch")
    for path in discover_graphs(args.gset_dir,args.graphs):
        _,rank=run_graph(
            path,factor_dir=args.factor_dir,dest=args.output_dir,
            ks=(2,),inits=tuple(args.methods),
            restarts=args.restarts,
            start_restart=args.start_restart,
            chunk_size=args.chunk_size,workers=args.workers,
            rounding_seed=args.rounding_seed,kl_seed=args.kl_seed,
            max_iter=args.max_iter,top_merges=args.top_merges,
            kl_passes=args.kl_passes,target=args.target,
        )
        print(f"{path.stem}: best initialization after KL: "
              f"{rank[0]['init']} cut={rank[0]['best_KL']:g}; "
              f"historical G63 reference=26917",
              flush=True)
    if args.push:
        _git_push(
            args.output_dir,
            "experiment018 compare G63 KMeansNM initializers",
            allowed_branch=args.push_branch,
        )


if __name__=="__main__":
    main()
