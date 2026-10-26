"""Export public convex SDP benchmark snapshots and optionally Git-push them.

Run in a second terminal while maxkcut-sdp-batch continues computing.
Only completed independently verified SDP partitions are summarized.
Pushing to the current GitHub repository is EXPLICIT (--push).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from .sdp_publish import publish_snapshot


PUBLIC_FILES = frozenset({
    "README.md", "summary.csv", "method-summary.csv",
    "rounding-trials.csv", "provenance.csv", "sdp-vs-cuda.csv",
})


def _git(repo: Path, *args: str, check: bool = True):
    return subprocess.run(
        ["git", *args], cwd=repo,
        capture_output=True, text=True, check=check,
    )


def _push_publication(dest: Path, complete: int) -> bool:
    """Stage/commit/push ONLY this sanitized publication directory.

    Refuse non-main branch, paths outside repository, or extra files.
    Never call git pull, reset, force-push or add a large solver cache.
    """
    dest = Path(dest).resolve()
    project = Path(_git(dest.parent, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    try:
        pathspec = dest.relative_to(project).as_posix()
    except ValueError as exc:
        raise ValueError(
            "Publication directory must be inside the current Git repository"
        ) from exc
    if pathspec.startswith("reports/"):
        raise ValueError("Cannot publish ignored reports/ or solver Gram caches")
    extra = [p.name for p in dest.iterdir()
             if p.is_dir() or p.name not in PUBLIC_FILES]
    if extra:
        raise ValueError(
            f"Refusing to stage unrecognized publication files: {extra}"
        )
    branch = _git(project, "symbolic-ref", "--short", "HEAD").stdout.strip()
    if branch != "main":
        raise RuntimeError(
            f"Current branch is {branch!r}, not 'main'. "
            "Switch to main before publishing for your supervisor."
        )
    _git(project, "add", "-A", "--", pathspec)
    change = _git(
        project, "diff", "--cached", "--quiet", "--", pathspec,
        check=False,
    )
    if change.returncode == 0:
        print("Publication is unchanged; no Git commit/push needed.", flush=True)
        return False
    if change.returncode != 1:
        raise RuntimeError(
            f"Could not inspect staged publication: {change.stderr}"
        )
    message = f"results: publish full convex SDP progress ({complete}/71)"
    _git(project, "commit", "-m", message, "--", pathspec)
    try:
        _git(project, "push", "origin", "HEAD:main")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Local result commit was created, but GitHub push failed. "
            "Do NOT force-push; inspect 'git status' and reconcile remote "
            "changes before retrying. Original git error: "
            f"{exc.stderr.strip()}"
        ) from exc
    print(f"Published GitHub main: {message}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/gset/full-sdp-batch"),
        help="Locally saved ignored SDP results from the running batch",
    )
    parser.add_argument(
        "--dest", type=Path, default=Path("results/gset/full-sdp"),
        help="Small, tracked GitHub publication folder",
    )
    parser.add_argument(
        "--cuda-results", type=Path,
        default=Path("reports/gset/all71-cuda-baseline.jsonl"),
        help="Optional saved CUDA comparison baseline; no CUDA is run",
    )
    parser.add_argument("--no-cuda", action="store_true",
                        help="Publish only original full convex SDP results")
    parser.add_argument(
        "--push", action="store_true",
        help="Explicitly commit/push curated files to GitHub main; no force push",
    )
    parser.add_argument(
        "--watch-seconds", type=int,
        help="Refresh and optionally push snapshots repeatedly; "
             "run in a second PowerShell terminal, Ctrl+C to stop",
    )
    args = parser.parse_args()
    if args.watch_seconds is not None and args.watch_seconds < 60:
        parser.error("--watch-seconds must be at least 60")

    def publish_once() -> None:
        result = publish_snapshot(
            args.output_dir, args.dest,
            cuda_results=None if args.no_cuda else args.cuda_results,
        )
        print(json.dumps(result, indent=2), flush=True)
        if args.push:
            _push_publication(args.dest, result["verified_complete"])

    if args.watch_seconds is None:
        publish_once()
        return
    try:
        while True:
            publish_once()
            time.sleep(args.watch_seconds)
    except KeyboardInterrupt:
        print("Publication watcher stopped; SDP batch is unaffected.")


if __name__ == "__main__":
    main()
