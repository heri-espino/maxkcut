"""Read-only monitoring of an existing full convex SDP batch.

A per-graph case is persisted as status='started' before its CPU solver
subprocess is launched. This alone cannot prove the subprocess is alive.
SCS verbose progress is available only if --solver-progress was enabled
BEFORE starting that graph. The numerical residuals and gaps are not a
reliable percentage of work remaining.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

_LOG_LINE = re.compile(r"^\s*(\d{1,9})\s*\|\s*(.*)$")
_NUMERIC = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?|inf|nan)$",
    re.IGNORECASE,
)


def latest_scs_iteration(log_path: Path) -> dict | None:
    """Parse the last visible SCS progress row without loading its whole log.

    SCS verbose rows have the usual form
      iter| primal_residual dual_residual gap objective ...
    (spaces between numeric columns, unlike the header's pipes).
    The first three metrics are reported without claiming a progress %.
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        return None
    with log_path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 65536))
        content = handle.read().decode("utf8", errors="replace")
    matched = None
    for line in content.splitlines():
        found = _LOG_LINE.match(line)
        if not found:
            continue
        tokens = found.group(2).split()
        if len(tokens) < 3 or not all(
            _NUMERIC.fullmatch(x) for x in tokens[:3]
        ):
            continue
        matched = {
            "iteration": int(found.group(1)),
            "primal_residual": float(tokens[0]),
            "dual_residual": float(tokens[1]),
            "duality_gap": float(tokens[2]),
        }
    return matched


def batch_status(output_dir: Path) -> dict:
    """Return saved per-graph batch status without starting any solver."""
    root = Path(output_dir)
    cases = root / "cases"
    if not cases.is_dir():
        raise FileNotFoundError(f"No saved full-SDP cases in {cases}")
    state_counts: dict[str, int] = {}
    in_progress = []
    verified_complete = 0
    for path in sorted(cases.glob("G*.json"),
                       key=lambda p: int(p.stem[1:])):
        case = json.loads(path.read_text(encoding="utf8"))
        state = str(case.get("status", "unknown"))
        state_counts[state] = state_counts.get(state, 0) + 1
        if state == "complete":
            verified_complete += 1
        if state != "started":
            continue
        log = Path(case["log_path"]) if case.get("log_path") else None
        stamp = log.stat().st_mtime if log and log.exists() else None
        info = latest_scs_iteration(log) if log else None
        if info and "max_iters" in case.get("settings", {}):
            info["iteration_cap"] = case["settings"]["max_iters"]
        in_progress.append({
            "graph": case.get("graph", path.stem),
            "status": "started_not_proof_process_is_alive",
            "solver_log": str(log) if log else None,
            "log_age_seconds": round(max(0, time.time() - stamp), 1)
                if stamp is not None else None,
            "scs_progress": info,
            "note": (
                "A missing SCS iteration row is normal when the child was "
                "launched without --solver-progress or its output is buffered."
                if info is None else
                "Residuals and duality gap measure convergence, "
                "not percentage complete."
            ),
        })
    return {
        "output_dir": str(root.resolve()),
        "graphs_with_case_records": sum(state_counts.values()),
        "completed_case_records": verified_complete,
        "statuses": dict(sorted(state_counts.items())),
        "started_cases_not_guaranteed_live": in_progress,
        "note": (
            "Read-only metadata inspection; 'complete' is a saved status, "
            "not a fresh archive verification. Start state may persist after "
            "a hard crash. Check process table to establish liveness."
        ),
    }
