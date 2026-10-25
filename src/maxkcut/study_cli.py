"""Compare *saved* full convex SDP and CUDA low-rank Max-Cut results.

Does not invoke a numerical optimizer, install a GPU backend, or allocate an
SDP matrix. Use `maxkcut` to solve the full convex SDP first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .study_compare import (
    UNSOLVED_THESIS_GRAPHS, full_sdp_preflight,
    write_study_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdp", type=Path, nargs="*", default=[],
        help="One or more original full-SDP maxkcut JSONLs, one run per graph",
    )
    parser.add_argument(
        "--cuda", type=Path, default=Path(
            "reports/gset/all71-cuda-baseline.jsonl"
        ),
        help="One CUDA low-rank JSONL, e.g. the saved 71-graph baseline",
    )
    parser.add_argument(
        "--gset-dir", type=Path, default=Path("data"),
        help="Canonical G-set data for exact normalized graph identity checks",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/gset/full-sdp-vs-cuda.csv"),
    )
    parser.add_argument(
        "--preflight-seven", action="store_true",
        help="Only print dense full-SDP memory lower bounds for thesis's "
             "seven excluded graphs; no solver is launched",
    )
    args = parser.parse_args()
    if args.preflight_seven:
        print(json.dumps(
            full_sdp_preflight(args.gset_dir, UNSOLVED_THESIS_GRAPHS),
            indent=2,
        ))
        return
    rows = write_study_comparison(
        args.sdp, args.cuda, args.gset_dir, args.output,
    )
    print(f"Saved: {args.output} ({len(rows)} graphs)")
    for row in rows:
        if row["sdp_best_cut"] is None and row["cuda_best_cut"] is None:
            continue
        print(
            f"{row['graph']}: full SDP={row['sdp_best_cut']}, "
            f"CUDA={row['cuda_best_cut']}, "
            f"paper={row['paper_cut']}, "
            f"identity={row['graph_identity']}, "
            f"SDP status={row['sdp_status']}"
        )


if __name__ == "__main__":
    main()
