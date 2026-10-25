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


def _compare_saved_runs(output_dir: Path, cuda: Path,
                        gset_dir: Path):
    """Rebuild the CSV from previously saved completed cases only."""
    case_dir = Path(output_dir) / "cases"
    if not case_dir.is_dir():
        raise FileNotFoundError(
            f"No full SDP cases directory: {case_dir}"
        )
    paths = []
    for case_file in case_dir.glob("G*.json"):
        case = json.loads(case_file.read_text(encoding="utf8"))
        if case.get("status") == "complete":
            paths.append(Path(case["source_jsonl"]))
    output = Path(output_dir) / "sdp-vs-cuda.csv"
    rows = write_study_comparison(paths, cuda, gset_dir, output)
    duplicate_graphs = [
        r for r in rows if (r["cuda_candidate_runs"] or 0) > 1
    ]
    print(
        f"Saved {len(rows)} graph comparisons, "
        f"{len(paths)} successful full SDPs: {output}",
        flush=True,
    )
    if duplicate_graphs:
        print(
            f"CUDA source had multiple archived runs for "
            f"{len(duplicate_graphs)} graphs; selected latest complete "
            "verified run per graph. Audit cuda_run_id, "
            "cuda_candidate_runs and cuda_selection_policy columns.",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--graphs", nargs="+", default=["ALL"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/gset/full-sdp-batch"))
    parser.add_argument("--compare-only", action="store_true",
                        help="Regenerate sdp-vs-cuda.csv from completed saved cases; "
                             "never launch a convex SDP subprocess")
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
    parser.add_argument("--no-wall-limit", action="store_true",
                        help="Remove BOTH per-graph and total wall-clock timeouts. "
                             "Does NOT remove SCS iteration limits, graph-size "
                             "safeguards, or protect against running out of RAM")
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
    if args.compare_only:
        if args.no_compare:
            parser.error("--compare-only and --no-compare are incompatible")
        if not args.cuda_results.is_file():
            parser.error(f"CUDA results JSONL does not exist: {args.cuda_results}")
        return _compare_saved_runs(
            args.output_dir, args.cuda_results, args.gset_dir
        )
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
    if args.no_wall_limit and (
        args.seconds_per_graph != 600. or args.total_seconds != 3600.
    ):
        parser.error(
            "--no-wall-limit cannot be combined with explicit "
            "--seconds-per-graph or --total-seconds settings"
        )
    outcome = run_full_sdp_batch(
        args.gset_dir, args.output_dir, names=args.graphs,
        max_n=args.max_n, allow_large=args.allow_large,
        acknowledge_huge_risk=args.acknowledge_huge_risk,
        solver=args.solver, eps=args.eps, max_iters=args.max_iters,
        restarts=args.restarts, seed=args.seed,
        cpu_threads=args.cpu_threads, scs_indirect=args.scs_indirect,
        seconds_per_graph=(
            None if args.no_wall_limit else args.seconds_per_graph
        ),
        total_seconds=(
            None if args.no_wall_limit else args.total_seconds
        ), max_graphs=args.max_graphs,
        retry_failed=args.retry_failed,
    )

    if not args.no_compare:
        if args.cuda_results.is_file():
            try:
                _compare_saved_runs(
                    args.output_dir, args.cuda_results, args.gset_dir
                )
            except (ValueError, OSError, KeyError) as exc:
                # The completed numerical results are already durable.
                # A corrupted/incomplete comparison must not claim the
                # entire SDP batch failed or require rerunning solvers.
                print(
                    f"Comparison not generated: {type(exc).__name__}: {exc}. "
                    "Full SDP results are safe; run --compare-only after fixing.",
                    flush=True,
                )
        else:
            print(f"CUDA baseline not present ({args.cuda_results}); "
                  "retained full SDP summary only.")
    return outcome


if __name__ == "__main__":
    main()
