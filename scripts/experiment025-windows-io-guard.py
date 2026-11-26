#!/usr/bin/env python
"""Launch frozen Experiment025 code with bounded Windows checkpoint I/O retries.

The numerical implementation and its source fingerprint under src/maxkcut/
remain IDENTICAL. This guard is deliberately outside that source tree: it only
retries transient Windows access-denied/sharing violations during atomic
checkpoint replacement inside Experiment025's result directory.

Use: python scripts/experiment025-windows-io-guard.py single --graph G21 ...
No external packages, privilege changes, or antivirus exclusions are needed.
"""
from __future__ import annotations

import os
import runpy
import sys
import time
from pathlib import PurePosixPath

_ORIGINAL_REPLACE = os.replace
_RESULT_FOLDER = "/results/gset/experiment025-gw16-bestcut-quality/"
_RETRYABLE_WINERRORS = frozenset((5, 32))  # Access denied; sharing violation.
_RETRIES = 10
_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0)


def _checkpoint_replace(source: os.PathLike[str] | str,
                        target: os.PathLike[str] | str) -> bool:
    """Only retry atomic JSON/NPZ replacement in Experiment025's result tree."""
    src = PurePosixPath(os.fspath(source).replace("\\", "/"))
    dst = PurePosixPath(os.fspath(target).replace("\\", "/"))
    normalized = "/" + str(dst).lstrip("/").casefold()
    if _RESULT_FOLDER not in normalized:
        return False
    if dst.suffix not in (".json", ".npz"):
        return False
    return src.name in (dst.name + ".tmp",
                        dst.stem + ".temporary.npz")


def retry_checkpoint_replace(source: os.PathLike[str] | str,
                             target: os.PathLike[str] | str,
                             *, replace_func=None, sleep_func=None,
                             attempts: int = _RETRIES) -> None:
    """Retry known transient Windows errors; propagate every other error.

    The same already-written temporary file is retried, so the atomic
    replace is not changed into a non-atomic read/write or delete/copy.
    A failed final attempt leaves the temporary file for forensic recovery.
    """
    if replace_func is None:
        replace_func = _ORIGINAL_REPLACE
    if sleep_func is None:
        sleep_func = time.sleep
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if not _checkpoint_replace(source, target):
        replace_func(source, target)
        return

    for attempt in range(attempts):
        try:
            replace_func(source, target)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in _RETRYABLE_WINERRORS:
                raise
            if attempt + 1 == attempts:
                raise
            delay = _DELAYS[min(attempt, len(_DELAYS) - 1)]
            print(
                f"[experiment025-io] Windows lock on {target}; "
                f"retry {attempt + 1}/{attempts - 1} after {delay:g}s "
                f"(WinError {error.winerror})",
                file=sys.stderr, flush=True,
            )
            sleep_func(delay)


def main() -> None:
    if not sys.argv[1:]:
        raise SystemExit("Usage: python scripts/experiment025-windows-io-guard.py "
                         "single|batch|summary [arguments]")
    # This is an I/O-only launcher: import unchanged source after installing
    # the narrowly scoped guard; its versioned scientific fingerprint is intact.
    os.replace = retry_checkpoint_replace
    try:
        runpy.run_module("maxkcut.experiment025_bestcut",
                         run_name="__main__", alter_sys=True)
    finally:
        os.replace = _ORIGINAL_REPLACE


if __name__ == "__main__":
    main()
