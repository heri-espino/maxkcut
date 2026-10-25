"""Run all canonical G-set *full convex* SDPs safely, one subprocess each.

A batch sweep is NOT a claim that every SDP finished. The default graph
size cap is 1000 vertices; graph-size skips and timeouts are reported.
Explicit opt-in is required for larger dense SDP cones.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sdp_batch import inventory, run_full_sdp_batch
from .study_compare import write_study_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--graphs", nargs="+", default=["ALL"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/gset/full-sdp-batch"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print graph sizes and dense matrix memory floors; never run SDP")
    parser.add_argument("--max-n", type=int, default=1000,
                        help="Conservative full-SDP vertex size cap (default: 1000)")
    parser.add_argument("--allow-large", action="store_true",
                        help="Required to lift full-SDP graph-size cap above 1000")
    parser.add_argument("--acknowledge-huge-risk", action="store_true",
                        help="Additional explicit opt-in for graph size >3000; "
                             "the OS could still run out of RAM")
    parser.add_argument("--solver", choices=("SCS",), default="SCS")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--scs-indirect", action="store_true",
                        help="Experimental SCS sparse iterative linear solver; "
                             "does NOT eliminate PSD eigenprojections")
    parser.add_argument("--seconds-per-graph", type=float, default=600.,
                        help="Child-process wall time limit per graph")
    parser.add_argument("--total-seconds", type=float, default=3600.,
                        help="Overall run wall budget; continuing later is supported")
    parser.add_argument("--max-graphs", type=int,
                        help="Optional prefix test; e.g. first two graphs")
    parser.add_argument("--cuda-results", type=Path, default=Path(
        "reports/gset/all71-cuda-baseline.jsonl"
    ), help="Saved CUDA low-rank JSONL; when present, automatically "
            "write an exact graph-verified full SDP vs CUDA CSV")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip reading the pre-existing CUDA baseline")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry failed/timed-out cases; successful graph archives "
                             "are still independently verified before skipping")
    args = parser.parse_args()
    rows = inventory(args.gset_dir, args.graphs)
    if args.dry_run:
        print(json.dumps({
            "graphs": len(rows),
            "allowed_by_n": sum(r["vertices"] <= args.max_n for r in rows),
            "skipped_by_n": sum(r["vertices"] > args.max_n for r in rows),
            "memory_note": "One dense float64 Gram only; true SDP RAM can be much larger",
            "items": [
                {
                    **r,
                    "selected_by_n": r["vertices"] <= args.max_n,
                }
                for r in rows
            ],
        }, indent=2))
        return
    outcome = run_full_sdp_batch(
        args.gset_dir, args.output_dir, names=args.graphs,
        max_n=args.max_n, allow_large=args.allow_large,
        acknowledge_huge_risk=args.acknowledge_huge_risk,
        solver=args.solver, eps=args.eps, max_iters=args.max_iters,
        restarts=args.restarts, seed=args.seed,
        cpu_threads=args.cpu_threads, scs_indirect=args.scs_indirect,
        seconds_per_graph=args.seconds_per_graph,
        total_seconds=args.total_seconds, max_graphs=args.max_graphs,
        retry_failed=args.retry_failed,
    )

    if not args.no_compare:
        if args.cuda_results.is_file():
            cases_dir = args.output_dir / "cases"
            paths = []
            for case_file in cases_dir.glob("G*.json"):
                case = json.loads(case_file.read_text(encoding="utf8"))
                if case.get("status") == "complete":
                    paths.append(Path(case["source_jsonl"]))
            target = args.output_dir / "sdp-vs-cuda.csv"
            comparisons = write_study_comparison(
                paths, args.cuda_results, args.gset_dir, target,
            )
            print(f"Saved {len(comparisons)} graph comparisons: {target}")
        else:
            print(f"CUDA baseline not present ({args.cuda_results}); "
                  "retained full SDP summary only.")
    return outcome


if __name__ == "__main__":
    main()
