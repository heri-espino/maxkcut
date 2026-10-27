"""Bounded retries for transient Windows locks during atomic file replacement.

Preserves the crucial atomic rename operation: NEVER fall back to writing
directly into the destination, which could corrupt an experiment archive.

When Windows antivirus, indexing or an open reader temporarily locks a
file, os.replace can raise WinError 5/32/33. Retry only this narrow
category, with a short bounded delay. A persistent permission problem
must still fail loudly; non-Windows behavior is unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
import time


def _retryable(error: PermissionError) -> bool:
    if os.name != "nt":
        return False
    code = getattr(error, "winerror", None)
    return code in (5, 32, 33) or (code is None and isinstance(error, PermissionError))


def replace_atomic(source: str | Path, destination: str | Path, *,
                   attempts: int = 7, delay: float = 0.05) -> None:
    """Replace atomically, retrying Windows sharing/permission races only.

    The source file must already be closed. Max delay through six retries
    is 0.05+0.10+0.20+0.40+0.40+0.40 = 1.55 seconds.
    """
    if attempts < 1 or delay < 0:
        raise ValueError("Invalid atomic replacement retry settings")
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if not _retryable(exc) or attempt + 1 >= attempts:
                raise
            time.sleep(min(0.4, delay * 2 ** attempt))
