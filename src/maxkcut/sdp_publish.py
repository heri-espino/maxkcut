"""Publish an auditable, small, privacy-conscious SDP result snapshot.

The active SDP batch writes atomically finalized cases under reports/.
This exporter reads them from a *different* process and writes curated
CSV/Markdown under the tracked results/ tree; it never launches solvers.

Dense n*n SDP Gram matrices, raw solver logs, local Windows paths and
unverified/incomplete trial distributions are intentionally excluded.
Git commit/push is opt-in via the companion command-line wrapper.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .archive import load_method
from .sdp_batch import _read_saved_result
from .study_compare import COLUMNS, compare_saved_runs


PUBLIC_SUMMARY = (
    "graph", "vertices", "edges", "status", "best_cut", "best_method",
    "sdp_solver_status", "attempts", "wall_seconds", "input_sha256",
)
METHOD_FIELDS = (
    "graph", "n", "method", "restarts", "best_cut", "worst_cut",
    "mean_cut", "median_cut", "sample_std_cut", "best_restart",
    "sdp_solver_status",
)
TRIAL_FIELDS = (
    "graph", "n", "method", "restart", "cut_weight",
    "rounding_seconds", "rounding_iterations", "solver_status",
    "solver_iterations", "sdp_objective_primal_uncertified",
    "sdp_original_solve_seconds", "artifact_run_id",
)
PROVENANCE_FIELDS = (
    "graph", "run_id", "input_sha256", "git_commit_at_solve",
    "solver", "solver_status", "eps", "max_iters",
    "solver_iterations", "solver_seconds_original",
    "cpu_threads", "restarts", "seed", "cvxpy_version", "scs_version",
)


def _numeric_sort(path: Path) -> int:
    name = path.stem
    if not name.startswith("G") or not name[1:].isdigit():
        raise ValueError(f"Unexpected graph case name: {path.name}")
    return int(name[1:])


def _as_csv(rows: list[dict], columns: tuple[str, ...]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in columns})
    return handle.getvalue()


def _write_if_changed(path: Path, text: str) -> bool:
    data = text.encode("utf8")
    if path.is_file() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=".publish-sdp-", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return True


def _verified_trial_records(
    archive: Path, jsonl: Path, manifest: dict,
) -> list[dict]:
    """Check *every* reported restart score against verified NPZ archives.

    `_read_saved_result` already verifies original graph checksum,
    the normalized weighted-edge snapshot and rescored partition arrays.
    This additional check prevents publishing individually corrupted
    or interleaved JSONL scores even if each method's best cut agrees.
    """
    with Path(jsonl).open("r", encoding="utf8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    by_method = {m: {} for m in manifest["methods"]}
    for row in rows:
        method = row.get("method")
        if method not in by_method:
            raise ValueError(f"Unexpected rounding method {method!r}")
        index = row.get("restart")
        if (not isinstance(index, int)
                or index in by_method[method]):
            raise ValueError("Missing/duplicate rounding restart index")
        by_method[method][index] = float(row["cut_weight"])

    for method, settings in manifest["methods"].items():
        _, scores = load_method(archive / settings["partition_file"])
        observed = by_method[method]
        if set(observed) != set(range(len(scores))):
            raise ValueError(f"Incomplete rounding restart indices for {method}")
        actual = np.asarray([observed[i] for i in range(len(scores))])
        if not np.allclose(actual, scores, rtol=0, atol=1e-8):
            raise ValueError(
                f"JSONL rounding values differ from archived partitions: {method}"
            )
    return rows


def publish_snapshot(
    batch_dir: Path, dest: Path, *,
    cuda_results: Path | None = None,
) -> dict:
    """Export independently verified complete SDP data, never raw Gram caches.

    Input case files use atomic JSON writes, so this is safe to run in a
    second terminal while the SDP batch is solving its next graph.
    A 'started' record is not counted as completed. A missing/corrupt
    completed archive is explicitly labeled 'unverified_complete'.
    """
    batch_dir, dest = Path(batch_dir), Path(dest)
    cases_dir = batch_dir / "cases"
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"Missing SDP cases directory: {cases_dir}")
    plan_path = batch_dir / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing reproducibility plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    input_dir = Path(plan["input_dir"])
    records = []
    method_rows = []
    trial_rows = []
    provenance = []
    verified_paths = []
    bad_cases = []

    for file in sorted(cases_dir.glob("G*.json"), key=_numeric_sort):
        case = json.loads(file.read_text(encoding="utf8"))
        name = file.stem
        if case.get("graph") != name:
            raise ValueError(f"Inconsistent graph identity in {file}")
        state = case.get("status", "unknown")
        numeric = None
        if state == "complete":
            try:
                numeric = _read_saved_result(
                    Path(case["source_jsonl"]),
                    name, int(case["settings"]["restarts"]),
                    input_dir / f"{name}.txt",
                )
                archive = Path(numeric["archive_path"])
                manifest = json.loads(
                    (archive / "manifest.json").read_text(encoding="utf8")
                )
                trial_records = _verified_trial_records(
                    archive, Path(case["source_jsonl"]), manifest,
                )
            except (OSError, ValueError, TypeError, KeyError) as exc:
                state = "unverified_complete"
                bad_cases.append(f"{name}: {type(exc).__name__}: {exc}")
        summary = {
            key: case.get(key) for key in PUBLIC_SUMMARY
        }
        summary["status"] = state
        if numeric is None:
            # Never publish a cut as verified for unfinished/corrupt cases.
            for key in ("best_cut", "best_method", "sdp_solver_status"):
                summary[key] = None
        else:
            summary.update({
                "best_cut": numeric["best_cut"],
                "best_method": numeric["best_method"],
                "sdp_solver_status": numeric["sdp_solver_status"],
                "input_sha256": numeric["input_sha256"],
            })
            verified_paths.append(Path(case["source_jsonl"]))
            for item in trial_records:
                trial_rows.append({
                    "graph": name, "n": item["n"], "method": item["method"],
                    "restart": item["restart"],
                    "cut_weight": item["cut_weight"],
                    "rounding_seconds": item.get("rounding_seconds"),
                    "rounding_iterations": item.get("rounding_iterations"),
                    "solver_status": item.get("solver_status"),
                    "solver_iterations": item.get("solver_iterations"),
                    "sdp_objective_primal_uncertified":
                        item.get("sdp_objective_primal"),
                    "sdp_original_solve_seconds":
                        item.get("sdp_original_solve_seconds"),
                    "artifact_run_id": item["artifact_run_id"],
                })
            for method, stats in manifest["methods"].items():
                method_rows.append({
                    "graph": name, "n": manifest["n"],
                    "method": method, "sdp_solver_status":
                        numeric["sdp_solver_status"], **stats,
                })
            settings = manifest.get("settings", {})
            versions = manifest.get("versions", {})
            info = manifest.get("solver_info", {})
            provenance.append({
                "graph": name, "run_id": manifest["run_id"],
                "input_sha256": manifest["graph_sha256"],
                "git_commit_at_solve": manifest.get("git", {}).get("commit"),
                "solver": settings.get("solver"),
                "solver_status": numeric["sdp_solver_status"],
                "eps": settings.get("eps"),
                "max_iters": settings.get("max_iters"),
                "solver_iterations": info.get("solver_iterations"),
                "solver_seconds_original":
                    info.get("sdp_original_solve_seconds"),
                "cpu_threads": settings.get("cpu_threads"),
                "restarts": settings.get("restarts"),
                "seed": settings.get("seed"),
                "cvxpy_version": versions.get("cvxpy"),
                "scs_version": versions.get("scs"),
            })
        records.append(summary)

    results = {"verified_complete": len(verified_paths),
               "total_cases": len(records),
               "unverified": bad_cases,
               "updated_files": []}
    contents = {
        "summary.csv": _as_csv(records, PUBLIC_SUMMARY),
        "method-summary.csv": _as_csv(method_rows, METHOD_FIELDS),
        "rounding-trials.csv": _as_csv(trial_rows, TRIAL_FIELDS),
        "provenance.csv": _as_csv(provenance, PROVENANCE_FIELDS),
    }
    if cuda_results is not None and Path(cuda_results).is_file():
        try:
            compared = compare_saved_runs(
                verified_paths, Path(cuda_results), input_dir,
            )
            public_comparison_columns = tuple(
                column for column in COLUMNS
                if column not in ("sdp_source", "cuda_source")
            )
            contents["sdp-vs-cuda.csv"] = _as_csv(
                compared, public_comparison_columns,
            )
            results["compared_graphs"] = len(compared)
        except (OSError, ValueError, KeyError) as exc:
            results["comparison_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
    else:
        results["comparison_skipped"] = "CUDA baseline not present"

    counts = {}
    numerical_status_counts = {}
    for row in records:
        key = str(row["status"])
        counts[key] = counts.get(key, 0) + 1
        if key == "complete":
            numerical = str(row["sdp_solver_status"])
            numerical_status_counts[numerical] = (
                numerical_status_counts.get(numerical, 0) + 1
            )
    lines = [
        "# G-set — Full convex SDP results",
        "",
        "This directory is a **snapshot of completed, archive-verified** "
        "CVXPY/SCS Max-Cut SDPs. Updated by the local publication command. "
        "No GPU low-rank substitute or new SDP solve is run here.",
        "",
        f"**Verified complete:** {results['verified_complete']} / 71 canonical graphs.",
        "",
        "## Files",
        "",
        "- [summary.csv](summary.csv): per-graph saved status and best feasible cut.",
        "- [method-summary.csv](method-summary.csv): GW/FJ, K-Means2N, "
        "K-MeansNM distributions (20 roundings per method in the main study).",
        "- [rounding-trials.csv](rounding-trials.csv): every independently "
        "recorded rounding score and its method/restart.",
        "- [provenance.csv](provenance.csv): graph SHA, solver options, "
        "versions, iteration count and source commit.",
    ]
    if "sdp-vs-cuda.csv" in contents:
        lines.append(
            "- [sdp-vs-cuda.csv](sdp-vs-cuda.csv): normalized graph-verified "
            "comparison against saved nonconvex low-rank CUDA results."
        )
    lines += [
        "",
        "## Status counts",
        "",
        "| Saved status | Graphs |",
        "| --- | ---: |",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| {status} | {count} |")
    lines += [
        "",
        "## Numerical SDP solver statuses",
        "",
        "| CVXPY/SCS status | Completed graphs |",
        "| --- | ---: |",
    ]
    for status, count in sorted(numerical_status_counts.items()):
        lines.append(f"| {status} | {count} |")
    lines += [
        "",
        "Runs marked `optimal_inaccurate` reached only the solver's "
        "weaker numerical termination criteria. Review feasibility and "
        "duality diagnostics before drawing conclusions.",
        "",
        "## Interpretation and limitations",
        "",
        "- A rounded cut is a **feasible Max-Cut lower bound**, not "
        "a proof of the integer optimum.",
        "- An SCS `optimal` status is numerical termination of the "
        "**continuous relaxation**, not proof of the best integer cut.",
        "- The SDP primal objective is **not** asserted to be a certified "
        "upper bound, and a `started` status may be stale after a crash.",
        "- Graphs marked `unverified_complete` have no published cut.",
        "- Results use the canonical graphs in the repository's `data/` "
        "directory; original compact graph/partition archives and huge "
        "Gram matrix caches are retained locally under ignored `reports/`.",
        "- Paths identifying the local workstation, solver logs and "
        "multi-GB matrices are intentionally excluded.",
        "",
        "Project: [maxkcut](../../../README.md).",
        "",
    ]
    if bad_cases:
        lines += [
            "**Warning:** Some completed case files could not be independently "
            "verified for this publication; inspect the local solver archives.",
            "",
        ]
    contents["README.md"] = "\n".join(lines)

    # Do not publish stale comparison data from a prior run if current
    # canonical comparison failed or the baseline has disappeared.
    stale = dest / "sdp-vs-cuda.csv"
    if "sdp-vs-cuda.csv" not in contents and stale.exists():
        stale.unlink()
        results["updated_files"].append(str(stale))

    for name, body in contents.items():
        file = dest / name
        if _write_if_changed(file, body):
            results["updated_files"].append(str(file))
    results["statuses"] = dict(sorted(counts.items()))
    return results
