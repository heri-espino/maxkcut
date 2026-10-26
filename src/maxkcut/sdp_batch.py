"""Resource-bounded, resumable G-set *full convex SDP* batch runner.

Each selected instance executes in its own Python subprocess using the
unchanged CVXPY/SCS maxkcut CLI. Wall-clock limits are configurable,
including an explicit unlimited-time option; independently verified
successful graph archives are reusable.
Do not conflate this with the CUDA low-rank nonconvex solver or the fast
local-search sweep.

Default n<=1000 is deliberately conservative. Attempts at large dense
SDP cones require explicit opt-in; wall-clock timeouts are not memory
limits and may not prevent Windows OOM/system pressure.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .archive import verify_archive
from .io import load_graph, edge_arrays
from .sweep import discover_graphs


EXPECTED = ("FJ", "K-Means2N", "K-MeansNM")
SUMMARY_FIELDS = (
    "graph", "vertices", "edges", "status", "best_cut", "best_method",
    "sdp_solver_status", "attempts", "wall_seconds", "reason", "source_jsonl",
    "archive_path", "log_path", "input_sha256",
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=".sdp-batch-", suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _graph_header(path: Path) -> tuple[int, int]:
    if path.suffix.lower() == ".txt":
        with path.open("r", encoding="utf8") as handle:
            fields = handle.readline().split()
        if len(fields) != 2:
            raise ValueError(f"Invalid G-set n m header in {path}")
        n, m = (int(x) for x in fields)
        if n < 2 or m < 1:
            raise ValueError(f"Invalid G-set graph dimensions in {path}")
        return n, m
    # Legacy .mat inputs need parsing to establish n and m.
    a = load_graph(path)
    return int(a.shape[0]), int(len(edge_arrays(a)[0]))


def inventory(gset_dir: Path, names: list[str] | None = None) -> list[dict]:
    rows = []
    for path in discover_graphs(gset_dir, names):
        n, m = _graph_header(path)
        rows.append({
            "graph": path.stem,
            "vertices": n,
            "edges": m,
            "one_float64_gram_gib": round(n * n * 8 / 2 ** 30, 3),
            "input_path": str(path.resolve()),
        })
    return rows


def _settings(*, solver: str, eps: float, max_iters: int, restarts: int,
              seed: int, cpu_threads: int | None, scs_indirect: bool,
              seconds_per_graph: float | None) -> dict:
    return {
        "solver": solver, "eps": eps, "max_iters": max_iters,
        "restarts": restarts, "seed": seed,
        "cpu_threads": cpu_threads, "scs_indirect": scs_indirect,
        "seconds_per_graph": seconds_per_graph,
        "k": 2, "solver_family": "full_convex_sdp_cvxpy",
    }


def _same_numerical_settings(old: dict | None, new: dict) -> bool:
    """Allow only time-budget changes within a completed numerical study.

    Legacy study plans stored seconds_per_graph alongside actual solver
    parameters. That execution guard is not part of numerical solver
    identity: changing it must not invalidate archived, verified results.
    """
    if not isinstance(old, dict):
        return False
    old_numeric = {k: v for k, v in old.items()
                   if k != "seconds_per_graph"}
    new_numeric = {k: v for k, v in new.items()
                   if k != "seconds_per_graph"}
    return old_numeric == new_numeric


def _timeout_seconds(seconds_per_graph: float | None,
                     remaining: float | None) -> float | None:
    """None means unlimited; otherwise cap a process by remaining time."""
    limits = [v for v in (seconds_per_graph, remaining) if v is not None]
    return min(limits) if limits else None


def _read_saved_result(jsonl: Path, graph: str, restarts: int,
                       graph_path: Path) -> dict:
    """Only accept an independently rescored, completed *full SDP* run."""
    if not jsonl.is_file():
        raise ValueError("Missing saved JSONL")
    rows = []
    with jsonl.open("r", encoding="utf8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != len(EXPECTED) * restarts:
        raise ValueError(
            f"Expected {len(EXPECTED)*restarts} full-SDP trials, found {len(rows)}"
        )
    ids = {row.get("artifact_run_id") for row in rows}
    if len(ids) != 1 or not all(isinstance(v, str) and v for v in ids):
        raise ValueError("Missing or mixed saved archive identifiers")
    run_id = next(iter(ids))
    archive = jsonl.parent / "artifacts" / run_id / f"{graph}-k2"
    manifest_file = archive / "manifest.json"
    if not manifest_file.is_file():
        raise ValueError("Missing completed SDP manifest")
    manifest = json.loads(manifest_file.read_text(encoding="utf8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Archive status {manifest.get('status')!r} is not complete")
    input_hash = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    if manifest.get("graph_sha256") != input_hash:
        raise ValueError("Input graph SHA256 differs from archive")
    if manifest.get("k") != 2 or manifest.get("graph") != graph:
        raise ValueError("Incorrect graph/k in SDP archive")
    if set(manifest.get("methods", {})) != set(EXPECTED):
        raise ValueError("Incomplete SDP method archives")
    verified = verify_archive(archive)
    if verified["status"] != "complete":
        raise ValueError("Partition archive did not verify as complete")
    for method in EXPECTED:
        if verified["methods"][method]["verified_partitions"] != restarts:
            raise ValueError(f"Incomplete archived roundings for {method}")
    grouped = {method: [] for method in EXPECTED}
    for row in rows:
        method = row.get("method")
        if (row.get("graph") != graph or row.get("k") != 2
                or method not in grouped
                or row.get("solver") is None
                or row.get("solver_status") not in ("optimal", "optimal_inaccurate")):
            raise ValueError("Wrong solver, graph, method or SDP status")
        grouped[method].append(float(row["cut_weight"]))
    if any(len(values) != restarts for values in grouped.values()):
        raise ValueError("Incorrect per-method trial count")
    best_method = max(EXPECTED, key=lambda k: max(grouped[k]))
    best_cut = float(max(grouped[best_method]))
    if abs(best_cut - verified["methods"][best_method]["best_cut"]) > 1e-8:
        raise ValueError("Reported best cut differs from independently rescored partition")
    return {
        "best_cut": best_cut,
        "best_method": best_method,
        "sdp_solver_status": rows[0]["solver_status"],
        "source_jsonl": str(jsonl.resolve()),
        "archive_path": str(archive.resolve()),
        "input_sha256": input_hash,
    }


def _latest_attempt_folder(folder: Path, number: int) -> Path:
    return folder / f"attempt-{number:03d}"


def _load_case(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf8")) if path.is_file() else None


def _write_summary(root: Path, graph_names: list[str]) -> Path:
    target = root / "summary.csv"
    cases = []
    for graph in graph_names:
        case = _load_case(root / "cases" / f"{graph}.json")
        if case:
            cases.append({name: case.get(name) for name in SUMMARY_FIELDS})
    with target.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(cases)
    return target


def _child_command(*, path: Path, jsonl: Path, solver: str, eps: float,
                   max_iters: int, restarts: int, seed: int,
                   cpu_threads: int | None, scs_indirect: bool, n: int,
                   solver_progress: bool = False) -> list[str]:
    command = [
        sys.executable, "-u", "-m", "maxkcut", "--graphs", path.name,
        "--gset-dir", str(path.parent.resolve()), "--k", "2",
        "--solver", solver, "--eps", str(eps),
        "--max-iters", str(max_iters), "--restarts", str(restarts),
        "--seed", str(seed), "--max-n", str(n),
        "--output", str(jsonl.resolve()),
    ]
    if cpu_threads is not None:
        command += ["--cpu-threads", str(cpu_threads)]
    if scs_indirect:
        command.append("--scs-indirect")
    if solver_progress:
        command.append("--verbose")
    return command


def run_full_sdp_batch(
    gset_dir: Path, output_dir: Path, *,
    names: list[str] | None = None, max_n: int = 1000,
    allow_large: bool = False, acknowledge_huge_risk: bool = False,
    solver: str = "SCS", eps: float = 1e-3,
    max_iters: int = 3000, restarts: int = 20,
    seed: int = 2026, cpu_threads: int | None = 8,
    scs_indirect: bool = False, seconds_per_graph: float | None = 600,
    total_seconds: float | None = 3600, max_graphs: int | None = None,
    retry_failed: bool = False, solver_progress: bool = False,
) -> dict:
    """Run bounded independent full CVXPY SDP child processes and persist status.

    Graph-size guards apply before loading the graph into CVXPY. No large case
    will run accidentally. Timeout kills the child process, though Windows OS
    OOM/commit pressure is not preventable by a timeout. Successful, checksum-
    and settings-matched archives are independently rescored before skipping.
    """
    if (max_n < 2 or
            (seconds_per_graph is not None and seconds_per_graph <= 0) or
            (total_seconds is not None and total_seconds <= 0)):
        raise ValueError("max_n must be >=2; time limits must be positive or None")
    if max_n > 1000 and not allow_large:
        raise ValueError("max_n > 1000 requires --allow-large")
    if max_n > 3000 and not acknowledge_huge_risk:
        raise ValueError(
            "n > 3000 dense SDPs can exhaust 64 GB RAM; "
            "explicit --acknowledge-huge-risk required"
        )
    if solver != "SCS":
        raise ValueError("Bounded batch currently supports SCS only")
    if eps <= 0 or max_iters < 1 or restarts < 1 or seed < 0:
        raise ValueError("Invalid solver parameters")
    if cpu_threads is not None and cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if max_graphs is not None and max_graphs < 1:
        raise ValueError("max_graphs must be positive")

    graph_paths = discover_graphs(gset_dir, names)
    if max_graphs is not None:
        graph_paths = graph_paths[:max_graphs]
    settings = _settings(
        solver=solver, eps=eps, max_iters=max_iters, restarts=restarts,
        seed=seed, cpu_threads=cpu_threads, scs_indirect=scs_indirect,
        seconds_per_graph=seconds_per_graph,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf8"))
        if (not _same_numerical_settings(plan.get("settings"), settings)
                or plan.get("input_dir") != str(Path(gset_dir).resolve())):
            raise ValueError(
                "Numerical settings or dataset changed; use a new --output-dir"
            )
    else:
        plan = {
            "settings": settings, "input_dir": str(Path(gset_dir).resolve()),
        }
    # Preserve the legacy plan's original settings and separately record
    # the new execution policy; no numerical provenance is overwritten.
    policy = {
        "seconds_per_graph": seconds_per_graph,
        "total_seconds": total_seconds,
        "unlimited_wall_time": (
            seconds_per_graph is None and total_seconds is None
        ),
        "solver_progress": bool(solver_progress),
    }
    history = plan.setdefault("execution_policies", [])
    if not history or history[-1] != policy:
        history.append(policy)
    _atomic_json(plan_path, plan)

    count = {
        "complete": 0, "skipped_verified": 0, "skipped_size": 0,
        "failed": 0, "timeout": 0, "skipped_failed": 0,
        "skipped_total_budget": 0,
    }
    started = time.monotonic()
    selected = [path.stem for path in graph_paths]
    for path in graph_paths:
        name = path.stem
        n, m = _graph_header(path)
        case_path = root / "cases" / f"{name}.json"
        previous = _load_case(case_path)
        old_attempts = previous.get("attempts", 0) if previous else 0
        if n > max_n:
            count["skipped_size"] += 1
            # Do not overwrite previous successes from earlier runs.
            if previous is None or previous.get("status") != "complete":
                _atomic_json(case_path, {
                    "graph": name, "vertices": n, "edges": m,
                    "status": "skipped_size", "attempts": old_attempts,
                    "reason": f"n={n} exceeds --max-n={max_n}",
                })
            print(f"{name}: SKIPPED n={n} > limit {max_n}", flush=True)
            continue
        if previous is not None and previous.get("status") == "complete":
            try:
                if (not _same_numerical_settings(
                        previous.get("settings"), settings
                    ) or not previous.get("source_jsonl")):
                    raise ValueError("Settings differ")
                _read_saved_result(
                    Path(previous["source_jsonl"]), name, restarts, path,
                )
                count["skipped_verified"] += 1
                print(f"{name}: verified resume skip", flush=True)
                continue
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        if (previous is not None
                and previous.get("status") in ("failed", "timeout")
                and not retry_failed):
            count["skipped_failed"] += 1
            print(f"{name}: previous {previous['status']}; --retry-failed to retry")
            continue
        remaining = (
            None if total_seconds is None
            else total_seconds - (time.monotonic() - started)
        )
        if remaining is not None and remaining < 1:
            count["skipped_total_budget"] += 1
            print(f"{name}: overall time budget reached", flush=True)
            continue
        attempt_no = old_attempts + 1
        directory = _latest_attempt_folder(root / "runs" / name, attempt_no)
        directory.mkdir(parents=True, exist_ok=False)
        jsonl = directory / f"{name}.jsonl"
        log_path = directory / "solver.log"
        command = _child_command(
            path=path, jsonl=jsonl, solver=solver, eps=eps,
            max_iters=max_iters, restarts=restarts, seed=seed,
            cpu_threads=cpu_threads, scs_indirect=scs_indirect, n=n,
            solver_progress=solver_progress,
        )
        record = {
            "graph": name, "vertices": n, "edges": m,
            "attempts": attempt_no, "status": "started",
            "settings": settings, "runtime_policy": policy,
            "source_jsonl": str(jsonl.resolve()),
            "log_path": str(log_path.resolve()),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        _atomic_json(case_path, record)
        graph_start = time.monotonic()
        limit = _timeout_seconds(seconds_per_graph, remaining)
        label = "unlimited wall time" if limit is None else f"wall limit {limit:.0f}s"
        print(f"{name}: full SDP n={n}; {label}", flush=True)
        try:
            with log_path.open("w", encoding="utf8") as log:
                completed = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    timeout=limit,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Child exited {completed.returncode}; see {log_path}"
                )
            verified = _read_saved_result(jsonl, name, restarts, path)
            record.update(verified)
            record["status"] = "complete"
            record["reason"] = None
            count["complete"] += 1
            print(f"{name}: full SDP complete; "
                  f"best cut={verified['best_cut']}", flush=True)
        except subprocess.TimeoutExpired:
            record["status"] = "timeout"
            record["reason"] = (
                f"Exceeded wall budget of {limit:.0f}s"
            )
            count["timeout"] += 1
            print(f"{name}: timed out; see {log_path}", flush=True)
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            record["reason"] = "Interrupted by user (Ctrl+C)"
            record["wall_seconds"] = time.monotonic() - graph_start
            _atomic_json(case_path, record)
            _write_summary(root, selected)
            raise
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            record["status"] = "failed"
            record["reason"] = f"{type(exc).__name__}: {exc}"
            count["failed"] += 1
            print(f"{name}: FAILED: {record['reason']}", flush=True)
        record["wall_seconds"] = time.monotonic() - graph_start
        _atomic_json(case_path, record)
        _write_summary(root, selected)
    summary = _write_summary(root, selected)
    result = {
        "graphs_selected": len(graph_paths), **count,
        "elapsed_seconds": time.monotonic() - started,
        "summary_csv": str(summary.resolve()),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result
