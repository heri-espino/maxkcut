#!/usr/bin/env python3
"""Run G81's ORIGINAL full convex SDP on a large-RAM Linux VM.

Publishes periodic read-only diagnostic Git checkpoints while the solver runs.
On successful archive verification, publishes method statistics, all restart
scores, compact original graph and partition NPZ archives to GitHub.

IMPORTANT: these are FILE checkpoints, NOT serialized/restoreable SCS
iterates. A VM termination during the SDP means the current numerical
solve must be started over. Other archived successful cases are reusable.

Use in a cloned repository, with Conda environment activated:
    python scripts/run_g81_vm.py --checkpoint-seconds 900 --cpu-threads 16
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

# Script run from the repository root. This also prevents publishing from
# an unrelated repository or accidentally committing data from elsewhere.
ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "data" / "G81.txt"
BATCH = ROOT / "reports" / "gset" / "g81-vm-128gb"
PUBLIC = ROOT / "results" / "gset" / "g81-128gb-vm"
ARCHIVE_MAX_BYTES = 90 * 1024 * 1024
RUN_LOG_MAX_BYTES = 10 * 1024 * 1024


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    temp.write_text(value, encoding="utf8")
    temp.replace(path)


def json_safe(value):
    """Represent non-finite native SCS residuals explicitly as strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def save_json(path: Path, data: dict) -> None:
    save_text(path, json.dumps(
        json_safe(data), indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (json.JSONDecodeError, OSError):
        return None


def sanitize(raw: str) -> str:
    # Solver output should contain diagnostics, not workstation usernames.
    home = str(Path.home())
    return raw.replace(str(ROOT), "<REPO>").replace(home, "<HOME>")


def log_excerpt(file: Path, max_bytes: int) -> str:
    if not file.is_file():
        return ""
    with file.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        data = stream.read()
    # A bounded log copy avoids accidentally adding gigabytes to Git.
    prefix = f"[last {max_bytes} bytes; full log remains under ignored reports/]\n" if size > max_bytes else ""
    return sanitize(prefix + data.decode("utf8", errors="replace"))


def linux_memory() -> dict:
    out: dict = {}
    source = Path("/proc/meminfo")
    if source.is_file():
        for line in source.read_text().splitlines():
            key, _, raw = line.partition(":")
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                out[f"{key}_gib"] = round(int(raw.strip().split()[0]) * 1024 / (1024 ** 3), 2)
    return out


def linux_processes(batch_pid: int) -> list[dict]:
    """Summarize only the batch and its direct children (no other users)."""
    try:
        raw = subprocess.run(
            ["ps", "-eo", "pid,ppid,pcpu,pmem,rss,etimes,comm"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in raw[1:]:
        parts = line.strip().split(None, 6)
        if len(parts) != 7:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if pid != batch_pid and ppid != batch_pid:
            continue
        rows.append({
            "pid": pid, "parent_pid": ppid,
            "cpu_percent_per_logical_processor": parts[2],
            "memory_percent": parts[3],
            "rss_mib": round(int(parts[4]) / 1024, 2),
            "elapsed_seconds": int(parts[5]),
            "process_name": parts[6],
        })
    return rows


def latest_case() -> dict | None:
    return read_json(BATCH / "cases" / "G81.json")


def latest_solver_log(case: dict | None) -> Path | None:
    if not case or not case.get("log_path"):
        return None
    candidate = Path(case["log_path"]).resolve()
    # Do not follow arbitrary case paths: this is a trusted local-run folder.
    if candidate != BATCH.resolve() and BATCH.resolve() not in candidate.parents:
        raise ValueError("Case log path escapes the local G81 batch directory")
    return candidate


def verified_archive_copy(case: dict | None) -> dict:
    """Only publish small compact original graph+rounding partitions after verification."""
    if not case or case.get("status") != "complete":
        return {"status": "not_complete"}
    from maxkcut.archive import verify_archive

    archive = Path(case["archive_path"]).resolve()
    if BATCH.resolve() not in archive.parents:
        raise ValueError("Verified archive path escapes local G81 batch")
    verification = verify_archive(archive)
    if verification.get("status") != "complete" or verification.get("graph") != "G81":
        raise ValueError("G81 archive did not verify as complete")
    original = read_json(archive / "manifest.json")
    if not original or original.get("graph_sha256") != verification["graph_sha256"]:
        raise ValueError("G81 manifest graph identity differs from validated data")
    if original.get("n") != 20000 or original.get("m") != 40000:
        raise ValueError("G81 archive has unexpected graph dimensions")
    if hashlib.sha256(GRAPH.read_bytes()).hexdigest() != original["graph_sha256"]:
        raise ValueError("G81 archive SHA256 differs from the repository dataset")
    destination = PUBLIC / "compact-archives"
    destination.mkdir(parents=True, exist_ok=True)
    files = ["graph.npz", *[
        str(summary["partition_file"]) for summary in original["methods"].values()
    ]]
    for name in files:
        source = archive / name
        if not source.is_file() or source.stat().st_size > ARCHIVE_MAX_BYTES:
            raise ValueError(f"Missing/oversize compact archive: {name}")
        shutil.copyfile(source, destination / name)

    # Preserve scientific provenance, but not local filesystem user paths.
    allowed = (
        "schema_version", "run_id", "created_utc", "status", "graph",
        "graph_sha256", "n", "m", "k", "graph_diagnostics",
        "settings", "methods", "versions",
    )
    clean = {key: original.get(key) for key in allowed}
    clean["git_revision_at_solve"] = original.get("git", {}).get("commit")
    solver_info = original.get("solver_info", {})
    clean["solver_info"] = {
        key: value for key, value in solver_info.items()
        if isinstance(value, (str, int, float, bool, type(None)))
        and "path" not in key.lower() and "cache" not in key.lower()
    }
    save_json(destination / "manifest.json", clean)
    save_json(destination / "verification.json", verification_for_public(verification))
    return {"status": "complete", "verified_methods": list(verification["methods"])}


def verification_for_public(result: dict) -> dict:
    return {key: result[key] for key in ("status", "graph", "graph_sha256", "methods")}


def checkpoint(batch_process: subprocess.Popen | None, *, state: str, exit_code: int | None,
               do_push: bool) -> None:
    from maxkcut.sdp_progress import latest_scs_iteration
    from maxkcut.sdp_publish import publish_snapshot

    PUBLIC.mkdir(parents=True, exist_ok=True)
    case = latest_case()
    solver_log = latest_solver_log(case)
    progress = (
        latest_scs_iteration(solver_log)
        if solver_log is not None else None
    )
    active = batch_process is not None and batch_process.poll() is None
    payload = {
        "graph": "G81", "timestamp_utc": timestamp(),
        "full_convex_sdp": True, "low_rank_cuda": False,
        "snapshot_type": "diagnostic_file_checkpoint",
        "true_scs_iteration_checkpoint": False,
        "cannot_resume_partial_solver_iterations": True,
        "batch_process_running": active,
        "launcher_state": state,
        "launcher_exit_code": exit_code,
        "case_status": case.get("status") if case else "not_started",
        "case_attempts": case.get("attempts") if case else None,
        "best_cut_if_complete": case.get("best_cut") if case and case.get("status") == "complete" else None,
        "solver_status_if_complete": case.get("sdp_solver_status") if case and case.get("status") == "complete" else None,
        "scs_latest_iteration": progress,
        "system_memory": linux_memory(),
        "batch_and_child_processes": linux_processes(batch_process.pid) if active else [],
        "note": "Iteration index and residuals are not a percentage-complete or convergence guarantee.",
    }
    save_json(PUBLIC / "checkpoint.json", payload)
    save_text(PUBLIC / "solver-output.txt", log_excerpt(solver_log, RUN_LOG_MAX_BYTES)
              if solver_log is not None else "No solver output yet.\n")
    save_text(PUBLIC / "batch-output.txt", log_excerpt(
        BATCH / "batch-stdout.txt", 2 * 1024 * 1024,
    ))
    if (BATCH / "plan.json").is_file() and (BATCH / "cases").is_dir():
        try:
            exported = publish_snapshot(BATCH, PUBLIC, cuda_results=None)
            payload["verified_completed_in_this_vm"] = exported["verified_complete"]
            if exported["unverified"]:
                payload["verification_errors"] = exported["unverified"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            payload["publish_snapshot_error"] = f"{type(exc).__name__}: {exc}"
    try:
        payload["compact_archive"] = verified_archive_copy(case)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        payload["compact_archive_error"] = f"{type(exc).__name__}: {exc}"
    save_json(PUBLIC / "checkpoint.json", payload)
    print(
        f"[{payload['timestamp_utc']}] G81 checkpoint: "
        f"case={payload['case_status']} latest_scs={progress} "
        f"memory={payload['system_memory']} push={do_push}",
        flush=True,
    )
    if do_push:
        push_snapshot()


def push_snapshot() -> None:
    """Commit and push only G81 public artifacts. Never force-push or auto-rebase."""
    from subprocess import CalledProcessError

    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if branch != "main":
            raise RuntimeError(f"Git publishing requires main, currently {branch}")
        relative = PUBLIC.relative_to(ROOT).as_posix()
        subprocess.run(["git", "add", "-A", "--", relative], cwd=ROOT, check=True)
        differences = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", relative], cwd=ROOT,
        )
        if differences.returncode == 0:
            return
        if differences.returncode != 1:
            raise RuntimeError("git diff --cached unexpectedly failed")
        subprocess.run(
            ["git", "commit", "-m",
             f"results: G81 128GB VM research checkpoint {timestamp()}",
             "--", relative], cwd=ROOT, check=True,
        )
        subprocess.run(
            ["git", "push", "origin", "HEAD:main"], cwd=ROOT, check=True,
        )
        print("G81 checkpoint pushed to GitHub main.", flush=True)
    except (CalledProcessError, RuntimeError, OSError) as exc:
        # Keep local results on disk and retry at the next checkpoint.
        print(
            f"WARNING: G81 GitHub push failed: {type(exc).__name__}: {exc}. "
            "No force push/pull/rebase was attempted. "
            "Local results persist until this VM is destroyed.",
            file=sys.stderr, flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--checkpoint-seconds", type=int, default=900)
    parser.add_argument("--no-push", action="store_true",
                        help="Create local checkpoints without publishing Git commits")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Explicitly retry a previously failed/timed-out G81 case;"
                             " incomplete iterations still start from zero")
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.checkpoint_seconds < 60:
        parser.error("cpu-threads must be >=1 and checkpoint-seconds >=60")
    if not GRAPH.is_file():
        parser.error(f"Missing canonical graph: {GRAPH}")
    os.chdir(ROOT)
    BATCH.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)

    # Lock out simultaneous G81 runners in this checkout. Linux VM expected.
    lock = BATCH / "runner.pid"
    if lock.exists():
        try:
            previous = int(lock.read_text().strip())
        except ValueError:
            parser.error(f"Invalid runner lock; check manually: {lock}")
        try:
            os.kill(previous, 0)
        except ProcessLookupError:
            pass  # Stale PID from a previous VM process; resume is allowed.
        except PermissionError:
            parser.error(f"Runner lock PID may still be active: {previous}")
        else:
            parser.error(f"Another G81 runner appears active (PID {previous})")
    lock.write_text(str(os.getpid()), encoding="utf8")

    cmd = [
        sys.executable, "-m", "maxkcut.sdp_batch_cli",
        "--graphs", "G81", "--max-n", "20000",
        "--allow-large", "--acknowledge-huge-risk",
        "--no-wall-limit", "--cpu-threads", str(args.cpu_threads),
        "--eps", "1e-3", "--max-iters", "3000",
        "--seed", "2026", "--restarts", "20",
        "--solver-progress", "--no-compare",
        "--output-dir", str(BATCH),
    ]
    if args.retry_failed:
        cmd.append("--retry-failed")
    save_json(PUBLIC / "run-config.json", {
        "graph": "G81", "n": 20000,
        "method": "original_full_convex_CVXPY_SCS",
        "cpu_threads": args.cpu_threads,
        "eps": 1e-3, "max_iters": 3000, "restarts_per_method": 20,
        "seed": 2026, "wall_time_limit": None,
        "checkpoint_seconds": args.checkpoint_seconds,
        "explicit_retry_failed": args.retry_failed,
        "solver_iteration_state_persisted": False,
        "run_command": [str(x).replace(str(ROOT), "<REPO>")
                        .replace(sys.executable, "python") for x in cmd],
        "started_utc": timestamp(),
    })

    child = None
    interrupted = False
    exit_code = None

    def stop(signum, frame):  # noqa: ARG001
        nonlocal interrupted
        interrupted = True
        print(f"Received signal {signum}; stopping batch process group.", flush=True)
        if child and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    old_int = signal.signal(signal.SIGINT, stop)
    old_term = signal.signal(signal.SIGTERM, stop)
    try:
        with (BATCH / "batch-stdout.txt").open("ab") as output:
            child = subprocess.Popen(
                cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            checkpoint(child, state="started", exit_code=None, do_push=not args.no_push)
            next_checkpoint = time.monotonic() + args.checkpoint_seconds
            while child.poll() is None:
                if interrupted:
                    break
                time.sleep(5)
                if time.monotonic() >= next_checkpoint:
                    checkpoint(child, state="running", exit_code=None,
                               do_push=not args.no_push)
                    next_checkpoint = time.monotonic() + args.checkpoint_seconds
            if interrupted and child.poll() is None:
                try:
                    child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    # Do not send SIGKILL unless explicitly requested:
                    # batch still owns the numerical child subprocess.
                    print("Batch is still shutting down; no forced kill.", flush=True)
                    child.wait()
            exit_code = child.wait()
            checkpoint(
                child, state="interrupted" if interrupted else "finished",
                exit_code=exit_code, do_push=not args.no_push,
            )
            print(f"G81 batch exit code: {exit_code}", flush=True)
            # The batch CLI itself can exit 0 after recording an individual
            # 'failed' case. The VM runner must NOT call that research success.
            case = latest_case()
            if exit_code != 0:
                return exit_code if exit_code > 0 else 1
            if case is None or case.get("status") != "complete":
                print(
                    "G81 was not completed even though the batch command exited 0. "
                    "Inspect saved case/log/checkpoints on GitHub.",
                    file=sys.stderr, flush=True,
                )
                return 2
            try:
                verified_archive_copy(case)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(
                    f"G81 completed record is not independently verified: {exc}",
                    file=sys.stderr, flush=True,
                )
                return 3
            return 0
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        if child is None or child.poll() is not None:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
