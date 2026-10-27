"""Fingerprint, persist and reload rank-r vectors without pickle or dense X.

Unlike the convex SDP matrix cache, this stores NONCONVEX optimizer factors
keyed by graph source contents, backend/version/device and all hyperparameters.
The cache never claims that a factor is an SDP certificate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


from .atomic_fs import replace_atomic

FACTOR_FORMAT = 1


def factor_identity(graph_path: Path, settings: dict) -> dict:
    path = Path(graph_path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "format": FACTOR_FORMAT,
        "graph_sha256": h.hexdigest(),
        "settings": settings,
    }


def factor_cache_path(cache_dir: Path, graph_path: Path, identity: dict) -> Path:
    token = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf8")
    ).hexdigest()[:24]
    return Path(cache_dir) / f"{Path(graph_path).stem}-k2-lowrank-{token}.npz"


def save_factor(path: Path, identity: dict, vectors: np.ndarray,
                optimizer_info: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "vectors": np.asarray(vectors, dtype=np.float32),
        "identity_json": np.asarray(json.dumps(identity, sort_keys=True)),
        "optimizer_info_json": np.asarray(json.dumps(
            optimizer_info, sort_keys=True, allow_nan=False
        )),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".lowrank-", suffix=".npz",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        replace_atomic(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_factor(path: Path, identity: dict,
                *, n: int, rank: int) -> tuple[np.ndarray, dict] | None:
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        recorded = json.loads(str(saved["identity_json"].item()))
        if recorded != identity:
            raise ValueError(f"Low-rank cache identity mismatch: {path}")
        vectors = saved["vectors"].astype(np.float32)
        info = json.loads(str(saved["optimizer_info_json"].item()))
    if (vectors.shape != (n, rank) or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
        raise ValueError(f"Invalid low-rank cached factors: {path}")
    if info.get("backend") != "torch_lowrank":
        raise ValueError(f"Invalid low-rank cache backend: {path}")
    return vectors, info
