"""Compare saved Max-Cut heuristic results to *paper* G-set references.

The comparison distinguishes proven optima (matching upper bounds) from
unproved best-known cuts and flags impossible apparent bound violations.
It never infers optimality from filename alone: graph identity must be checked.
"""
from __future__ import annotations

import csv
from pathlib import Path
import uuid

from .references import reference_for_graph


COLUMNS = [
    "graph", "n", "m", "status", "our_best_cut", "our_mean_cut",
    "completed_restarts", "paper_cut", "paper_upper_bound",
    "reference_status", "gap_to_paper_cut", "percent_of_paper_cut",
    "percent_of_optimum_if_verified", "assessment",
    "paper_reference", "input_sha256",
]


def compare_rows(rows: list[dict]) -> list[dict]:
    """Augment CSV rows with properly classified literature references."""
    output = []
    for row in rows:
        ref = reference_for_graph(row["graph"])
        ours = row.get("best_cut")
        our_value = None if ours in (None, "") else float(ours)
        if ref is None:
            assessment = "no_reference_in_paper_table_1"
        elif row.get("status") not in ("complete", "budget_limited") or our_value is None:
            assessment = "run_incomplete_or_failed"
        elif row.get("n") not in (None, "", ref["reference_vertices"], str(ref["reference_vertices"])):
            assessment = "GRAPH_SIZE_MISMATCH_check_identity"
        elif our_value > ref["published_upper_bound"] + 1e-8:
            assessment = "ABOVE_PUBLISHED_UPPER_BOUND_verify_graph_and_objective"
        elif our_value > ref["published_cut"] + 1e-8:
            assessment = "ABOVE_PUBLISHED_CUT_independently_verify"
        elif abs(our_value - ref["published_cut"]) <= 1e-8:
            assessment = "matches_paper_cut"
        else:
            assessment = "below_paper_cut"

        comparable = (
            ref is not None and our_value is not None
            and assessment in ("matches_paper_cut", "below_paper_cut",
                               "ABOVE_PUBLISHED_CUT_independently_verify")
            and ref["published_cut"] > 0
        )
        output.append({
            "graph": row["graph"],
            "n": row.get("n"),
            "m": row.get("m"),
            "status": row.get("status"),
            "our_best_cut": our_value,
            "our_mean_cut": row.get("mean_cut"),
            "completed_restarts": row.get("completed_restarts"),
            "paper_cut": ref["published_cut"] if ref else None,
            "paper_upper_bound": ref["published_upper_bound"] if ref else None,
            "reference_status": ref["reference_status"] if ref else "unavailable",
            "gap_to_paper_cut": (
                ref["published_cut"] - our_value if comparable else None
            ),
            "percent_of_paper_cut": (
                100 * our_value / ref["published_cut"] if comparable else None
            ),
            # This is an achieved quality ratio only AFTER identity verification.
            "percent_of_optimum_if_verified": (
                100 * our_value / ref["published_cut"]
                if comparable and ref["reference_status"] == "proven_optimum"
                else None
            ),
            "assessment": assessment,
            "paper_reference": ref["source"] if ref else None,
            "input_sha256": row.get("input_sha256"),
        })
    return output


def write_comparison(summary_file: Path, target: Path | None = None) -> Path:
    """Safely derive comparison.csv from a saved sweep summary.csv.

    This uses only previously saved observations and invokes no optimization.
    """
    summary_file = Path(summary_file)
    if target is None:
        target = summary_file.with_name("comparison.csv")
    target = Path(target)
    with summary_file.open("r", newline="", encoding="utf8") as stream:
        rows = list(csv.DictReader(stream))
    results = compare_rows(rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with scratch.open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(results)
        scratch.replace(target)
    finally:
        scratch.unlink(missing_ok=True)
    return target
