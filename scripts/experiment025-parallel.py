#!/usr/bin/env python
"""Experiment025 direct-VM parallel scheduler (no EVA solve; no Actions jobs).

One thread per graph supervises one Conda Python subprocess per method/chunk.
Different graphs may run concurrently. Only the parent process performs Git
commits and Git LFS pushes, under a lock, after a method checkpoint has finished.
This keeps each graph's shared GW normal archive single-writer.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path("results/gset/experiment025-gw16-bestcut-quality")
RESULTS_BRANCH = "experiment025-results"
PILOT = ("G1", "G14", "G11")
PUBLISH_LOCK = threading.Lock()
STOP = threading.Event()


def git(*args: str, allow_no_change: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", *args], text=True, capture_output=True,
                            encoding="utf-8", errors="replace")
    if result.returncode and not (allow_no_change and result.returncode == 1):
        raise RuntimeError(f"git {' '.join(args)} failed ({result.returncode}): "
                           f"{result.stderr[-2500:]}")
    return result


def push_with_confirmation() -> None:
    """Treat an ambiguous push as success only if GitHub has our exact SHA.

    No force pushes, automatic rebases or retries for non-fast-forward errors.
    A transient transport failure is retried at most twice, with receipt checks.
    """
    import re

    local_sha = git("rev-parse", "HEAD").stdout.strip()
    transport = re.compile(
        r"RPC failed|curl [0-9]+|Connection was aborted|"
        r"unexpected disconnect|remote end hung up|connection reset",
        re.IGNORECASE)
    for attempt in range(3):
        try:
            git("push", "origin", f"HEAD:{RESULTS_BRANCH}")
            return
        except RuntimeError as error:
            # A Git client may exit 1 after GitHub actually updated its ref.
            try:
                result = git("ls-remote", "origin", f"refs/heads/{RESULTS_BRANCH}")
                remote_sha = result.stdout.split()[0] if result.stdout.strip() else ""
            except RuntimeError:
                remote_sha = ""
            if remote_sha == local_sha:
                print("CONFIRMED remote SHA after ambiguous Git push failure",
                      flush=True)
                return
            if attempt == 2 or not transport.search(str(error)):
                raise
            delay = 2 * (attempt + 1)
            print(f"Transient push failure (attempt {attempt + 1}/3); "
                  f"retry in {delay}s; remote SHA != local SHA", flush=True)
            time.sleep(delay)


def publish(graph: str, phase: str, method: str, completed: int) -> None:
    """Publish only a finished graph's files; no other process mutates Git."""
    path = ROOT / phase / graph
    with PUBLISH_LOCK:
        git("add", "-A", "--", path.as_posix())
        # Root .gitignore excludes *.log: explicitly archive THIS method's
        # diagnostic transcript without exposing other ignored files or
        # causing the VM wrapper's clean-worktree check to reject old logs.
        logfile = path / "logs" / (method + "-parallel.log")
        if logfile.is_file():
            git("add", "-f", "--", logfile.as_posix())
        staged = git("diff", "--cached", "--quiet", allow_no_change=True)
        if staged.returncode == 0:
            return
        git("-c", "user.name=Experiment025 VM Runner",
            "-c", "user.email=experiment025-vm@users.noreply.github.com",
            "commit", "-m", f"experiment025: {phase} {graph} {method} {completed}/16 parallel checkpoint")
        # Do not rebase/reset/force-push while other graphs write to disk.
        # A competing writer causes a rejected push and a fail-closed stop.
        push_with_confirmation()
        print(f"PUBLISHED {phase} {graph} {method}: {completed}/16", flush=True)


def run_graph(graph: str, phase: str, methods: tuple[str, ...], chunk: int) -> dict:
    completed_methods = 0
    for method in methods:
        if STOP.is_set():
            return {"graph": graph, "status": "stopped", "methods": completed_methods}
        target = ROOT / phase / graph / method
        logdir = ROOT / phase / graph / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        logfile = logdir / (method + "-parallel.log")
        while True:
            if STOP.is_set():
                return {"graph": graph, "status": "stopped", "methods": completed_methods}
            before = 0
            status_file = target / "status.json"
            if status_file.exists():
                before = int(json.loads(status_file.read_text(encoding="utf-8"))
                             .get("starts_completed", 0))
            # Operational I/O retry wrapper only; numerical source stays frozen.
            command = [sys.executable, "scripts/experiment025-windows-io-guard.py",
                       "single", "--graph", graph, "--method", method,
                       "--phase", phase, "--max-new-seeds", str(chunk)]
            print(f"START {phase} {graph} {method}: {before}/16", flush=True)
            with logfile.open("a", encoding="utf-8") as output:
                output.write("\n=== " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                             + " " + " ".join(command) + " ===\n")
                output.flush()
                process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                         check=False)
                output.write(f"\nEXIT_CODE={process.returncode}\n")
                output.flush()
            # Even a failed seed may have flushed useful journal events.
            # Publish its isolated graph folder before reporting the failure.
            current = before
            if status_file.exists():
                current = int(json.loads(status_file.read_text(encoding="utf-8"))
                              .get("starts_completed", 0))
            publish(graph, phase, method, current)
            if process.returncode:
                raise RuntimeError(f"{graph}/{method} failed ({process.returncode}); "
                                   f"see {logfile}")
            if current == 16:
                record = json.loads(status_file.read_text(encoding="utf-8"))
                if record.get("status") != "complete_16_best_seen" or record.get(
                        "completed_seed_indices") != list(range(16)):
                    raise RuntimeError(f"Invalid 16-seed coverage for {graph}/{method}")
                completed_methods += 1
                break
            if current <= before:
                raise RuntimeError(f"No new GW seeds for {graph}/{method}; inspect {logfile}")
    return {"graph": graph, "status": "complete", "methods": completed_methods}


def acquire_lock() -> Path:
    p = Path(git("rev-parse", "--git-path", "experiment025-parallel.lock").stdout.strip())
    if not p.is_absolute():
        p = Path.cwd() / p
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Parallel VM lock already exists: {p}. "
                           "Inspect active jobs before removing a stale lock.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()}\n")
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    parser.add_argument("--graph", default="G11")
    parser.add_argument("--all-graphs", action="store_true")
    parser.add_argument("--only-graph", action="store_true")
    parser.add_argument("--method", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-seeds", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 1 <= args.checkpoint_seeds <= 16:
        parser.error("workers must be 1..32 and checkpoint-seeds must be 1..16")
    if args.phase == "pilot" and args.all_graphs:
        parser.error("pilot is restricted to G1, G14 and G11")
    # Import only the frozen Experiment025 numerical protocol. This scheduler
    # lives under scripts/, outside the optimizer source SHA256 fingerprint.
    from maxkcut.experiment025_bestcut import GRAPH_IDS, METHODS
    if args.graph not in GRAPH_IDS:
        parser.error("Unrecognized graph")
    if args.method and args.method not in METHODS:
        parser.error("Unrecognized method")
    graphs = ((args.graph,) if args.only_graph else PILOT) if args.phase == "pilot" else (
        tuple(GRAPH_IDS) if args.all_graphs else (args.graph,))
    methods = (args.method,) if args.method else tuple(METHODS)
    plan = dict(phase=args.phase, graph_count=len(graphs),
                method_count=len(methods), graph_method_cells=len(graphs)*len(methods),
                requested_workers=args.workers,
                usable_parallel_graphs=min(args.workers, len(graphs)),
                checkpoint_seeds=args.checkpoint_seeds,
                graph_ids=list(graphs), methods=list(methods), new_eva_solves=0)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0
    branch = git("branch", "--show-current").stdout.strip()
    if branch != RESULTS_BRANCH:
        raise RuntimeError(f"Expected dedicated {RESULTS_BRANCH} branch, found {branch!r}. "
                           "Use experiment025-vm.ps1 to prepare the correct branch.")
    if git("remote", "get-url", "origin").stdout.strip().lower().find(
            "heri-espino/maxkcut") < 0:
        raise RuntimeError("Unexpected origin; refusing result publication")
    lock = acquire_lock()
    failures = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_graph, graph, args.phase, methods,
                                   args.checkpoint_seeds): graph for graph in graphs}
            for future in as_completed(futures):
                graph = futures[future]
                try:
                    outcome = future.result()
                    print("DONE " + json.dumps(outcome), flush=True)
                except Exception as exc:
                    STOP.set()
                    failures.append((graph, str(exc)))
                    print(f"FAILED {graph}: {exc}", file=sys.stderr, flush=True)
        if failures:
            print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
            return 1
        return 0
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
