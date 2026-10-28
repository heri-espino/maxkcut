"""Offline, source-attributed G-set literature comparison; never runs an SDP.

Reference cuts are feasible (lower bounds) unless an independent
published upper bound matches the cut. Even then, transferring a
published proof to *local* data requires checking signed edge identity.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from uuid import uuid4


REFERENCE_STATUS = {"published_proven_optimum", "reported_feasible_cut_unproved"}
ROUNDERS = ("eva_FJ_best", "eva_KMeans2N_best", "eva_KMeansNM_best")
FIELDS = (
    "graph", "vertices", "edges", "eva_status", "eva_sweeps",
    "eva_FJ_best", "eva_KMeans2N_best", "eva_KMeansNM_best",
    "eva_best_cut", "reference_cut", "gap_to_reference",
    "percent_of_reference", "reference_status", "published_upper_bound",
    "source_key", "source_url", "graph_identity_evidence", "graph_sha256",
)


def _rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf8") as stream:
        return list(csv.DictReader(stream))


def load_references(path: Path) -> dict[str, dict]:
    """Load source-attributed reference table, rejecting missing provenance."""
    output = {}
    for r in _rows(path):
        graph = r["graph"]
        if not re.fullmatch(r"G[1-9][0-9]*", graph) or graph in output:
            raise ValueError(f"Invalid or duplicate reference graph: {graph!r}")
        n, m, best = (int(r[c]) for c in (
            "reference_vertices", "reference_edges", "reference_cut"
        ))
        if n < 2 or m < 0 or best < 0:
            raise ValueError(f"{graph}: invalid reference metadata")
        if r["reference_status"] not in REFERENCE_STATUS:
            raise ValueError(f"{graph}: invalid reference status")
        if not r["source_key"] or not r["source_url"].startswith("https://"):
            raise ValueError(f"{graph}: missing source provenance")
        bound = r.get("published_upper_bound", "")
        if bound:
            if int(bound) < best:
                raise ValueError(f"{graph}: feasible cut exceeds published bound")
            if (r["reference_status"] == "published_proven_optimum"
                    and int(bound) != best):
                raise ValueError(f"{graph}: cannot claim proven optimum")
        elif r["reference_status"] == "published_proven_optimum":
            raise ValueError(f"{graph}: proof requires matching upper bound")
        output[graph] = r
    return output


def compare(eva_summary: Path, reference_csv: Path) -> list[dict]:
    """Join exact graph ID and (n, m); never infer identity from size alone.

    All results are clearly labeled as comparisons conditional on exact
    weighted-edge identity with the graph used by the cited publication.
    """
    refs = load_references(reference_csv)
    eva = {}
    for row in _rows(eva_summary):
        graph = row["graph"]
        if graph in eva:
            raise ValueError(f"Duplicate EVA row: {graph}")
        eva[graph] = row

    output = []
    for graph, ref in sorted(refs.items(), key=lambda item: int(item[0][1:])):
        src = eva.get(graph, {})
        present = bool(src)
        size_match = (
            present and str(src.get("vertices", "")) == ref["reference_vertices"]
            and str(src.get("edges", "")) == ref["reference_edges"]
        )
        scores = []
        for key in ROUNDERS:
            raw = src.get(key, "")
            if raw in ("", None):
                continue
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"{graph}: nonfinite EVA cut")
            scores.append(value)

        best = max(scores) if len(scores) == len(ROUNDERS) else None
        if best is not None and best.is_integer():
            best = int(best)
        comparable = size_match and best is not None
        paper_cut = int(ref["reference_cut"])
        # A graph label and matching n/m are NOT a signed weighted edge proof.
        identity = (
            "not_run" if not present else
            "size_or_edge_count_mismatch" if not size_match else
            "size_and_edge_count_match_only"
        )
        row = {
            "graph": graph,
            "vertices": src.get("vertices", ""),
            "edges": src.get("edges", ""),
            "eva_status": src.get("status", "not_run"),
            "eva_sweeps": src.get("eva_sweeps", ""),
            **{method: src.get(method, "") for method in ROUNDERS},
            "eva_best_cut": best if best is not None else "",
            "reference_cut": paper_cut,
            "gap_to_reference": paper_cut - best if comparable else "",
            "percent_of_reference": (
                round(100 * best / paper_cut, 4)
                if comparable and paper_cut > 0 else ""
            ),
            "reference_status": ref["reference_status"],
            "published_upper_bound": ref["published_upper_bound"],
            "source_key": ref["source_key"],
            "source_url": ref["source_url"],
            "graph_identity_evidence": identity,
            "graph_sha256": src.get("graph_sha256", ""),
        }
        output.append(row)
    return output


def write_comparison(eva_summary: Path, reference_csv: Path, output: Path) -> Path:
    rows = compare(eva_summary, reference_csv)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_name(target.name + "." + uuid4().hex + ".tmp")
    try:
        with scratch.open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        scratch.replace(target)
    finally:
        scratch.unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eva-summary", type=Path,
                        default=Path("results/gset/literature-comparison/"
                                     "experiment004-eva-precision-source.csv"))
    parser.add_argument("--references", type=Path,
                        default=Path("data/gset_literature_reference_71.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gset/literature-comparison/"
                                     "experiment004-eva-precision-comparison.csv"))
    args = parser.parse_args()
    result = write_comparison(args.eva_summary, args.references, args.output)
    print(f"Compared {len(_rows(result))} graphs: {result}")


if __name__ == "__main__":
    main()
