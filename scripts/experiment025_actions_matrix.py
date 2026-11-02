"""Plan bounded GitHub Actions shards for Experiment025.

Each full-batch dispatch contains 71 graphs x 3 method protocols = 213
independent graph-method jobs, below GitHub's 256-job matrix limit.
Six explicit batches cover all 71 x 18 combinations once. A separate
3 x 18 pilot validates the selected active PCAG103A-13 runner first.

No numerical experiment runs in this planner; this module uses stdlib only.
"""
from __future__ import annotations

import argparse
import json

GRAPHS = tuple([f"G{i}" for i in range(1, 68)] + ["G70", "G72", "G77", "G81"])
METHODS = (
    "gw-rr", "control-paired", "gw-threshold",
    "cutaware-kmeans", "hyperplane-sweep", "multigroup-k4",
    "multigroup-k8", "em-hard", "em-cem",
    "em-gibbs", "mst-l2-longest", "mst-l2-balanced",
    "mst-l2-diameter", "mst-l2-bestcut", "mst-l2-multicut",
    "mst-l1-bestcut", "mst-linf-bestcut", "control-tabu",
)
PILOT_GRAPHS = ("G1", "G14", "G11")
METHOD_BATCHES = tuple(tuple(METHODS[k:k + 3]) for k in range(0, 18, 3))
assert len(GRAPHS) == 71 and len(METHODS) == 18
assert len(METHOD_BATCHES) == 6 and all(len(b) == 3 for b in METHOD_BATCHES)


def matrix(mode: str, *, batch: int = 0, graph: str = "G11",
           method: str = "gw-rr", phase: str = "full") -> dict:
    if mode == "pilot":
        cases = [(g, m, "pilot") for g in PILOT_GRAPHS for m in METHODS]
    elif mode == "full-batch":
        if batch not in range(6):
            raise ValueError("batch must be 0-5")
        cases = [(g, m, "full") for g in GRAPHS for m in METHOD_BATCHES[batch]]
    elif mode == "seed-batch":
        if batch not in range(6) or phase not in ("pilot","full"):
            raise ValueError("Invalid seed batch or phase")
        graphs=PILOT_GRAPHS if phase=="pilot" else GRAPHS
        cases=[(g,m,phase) for g in graphs for m in METHOD_BATCHES[batch]]
    elif mode == "single":
        if graph not in GRAPHS or method not in METHODS or phase not in ("pilot", "full"):
            raise ValueError("Unknown graph, method, or phase")
        cases = [(graph, method, phase)]
    elif mode == "coverage":
        cases = [("G11", "gw-rr", "coverage")]
    else:
        raise ValueError(f"Unsupported experiment scope: {mode}")
    assert 1 <= len(cases) <= 256
    return {"include": [
        {"graph": g, "method": m, "phase": p} for g, m, p in cases
    ]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True,
                   choices=("pilot", "full-batch", "seed-batch", "single", "coverage"))
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--graph", default="G11")
    p.add_argument("--method", default="gw-rr")
    p.add_argument("--phase", default="full", choices=("pilot", "full"))
    args = p.parse_args()
    print(json.dumps(matrix(args.mode, batch=args.batch, graph=args.graph,
                            method=args.method, phase=args.phase),
                     separators=(",", ":"), ensure_ascii=True))


if __name__ == "__main__":
    main()
