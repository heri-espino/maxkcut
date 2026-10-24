"""Disk cache for expensive, reproducible SDP solutions.

One cache entry is tied to the exact input .mat bytes and numerical solver
settings/versions. Cached matrices must still pass the usual factorization and
feasibility diagnostics before scientific use.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import numpy as np


CACHE_FORMAT_VERSION = 1


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def cache_identity(graph_path: Path, *, k: int, solver: str, eps: float,
                   max_iters: int, scs_indirect: bool) -> dict:
    """Construct a cache identity; include input SHA256 and solver versions."""
    return {
        "format": CACHE_FORMAT_VERSION,
        "graph_sha256": hashlib.sha256(Path(graph_path).read_bytes()).hexdigest(),
        "k": int(k),
        "solver": solver.upper(),
        "eps": float(eps) if solver.upper() == "SCS" else None,
        "max_iters": int(max_iters) if solver.upper() == "SCS" else None,
        "scs_indirect": bool(scs_indirect) if solver.upper() == "SCS" else False,
        "versions": {
            "cvxpy": _package_version("cvxpy"),
            "scs": _package_version("scs") if solver.upper() == "SCS" else None,
            "clarabel": _package_version("clarabel") if solver.upper() == "CLARABEL" else None,
            "mosek": _package_version("Mosek") if solver.upper() == "MOSEK" else None,
            "sdpa": _package_version("sdpa-python") if solver.upper() == "SDPA" else None,
        },
    }


def cache_path(graph_path: Path, cache_dir: Path, identity: dict) -> Path:
    """Derive a safe, deterministic pathname from the complete cache identity."""
    signature = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / f"{Path(graph_path).stem}-k{identity['k']}-{digest}.npz"


def load_cached_sdp(path: Path, identity: dict, *, n: int):
    """Return (X, original_solver_diagnostics), or None for a missing entry."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        matrix = np.array(stored["gram"], dtype=float, copy=True)
    if metadata.get("identity") != identity:
        raise ValueError(f"Cache metadata mismatch in {path}. Delete this cache file.")
    if matrix.shape != (n, n) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid cached SDP matrix in {path}. Delete this cache file.")
    return matrix, metadata["solver_info"]


def save_cached_sdp(path: Path, identity: dict, x: np.ndarray, info: dict) -> None:
    """Atomically save an SDP solution; never unpickle untrusted cache content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(x, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("SDP cache expects a square Gram matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Refusing to cache a nonfinite Gram matrix.")
    metadata = json.dumps(
        {"identity": identity, "solver_info": info}, sort_keys=True, allow_nan=False
    )
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".maxkcut-", suffix=".npz", dir=path.parent, delete=False
        ) as temporary:
            temp_name = temporary.name
            np.savez(temporary, gram=matrix, metadata=np.array(metadata))
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
