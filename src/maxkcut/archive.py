"""Compact, self-contained and independently verifiable experiment artifacts.

Archives contain a normalized weighted edge list and *all* rounded vertex
partitions. The large raw SDP Gram matrix lives only in the existing
fingerprinted SDP cache, and its path/settings are referenced in the manifest.
No pickles, network services or automatically launched SDP jobs are used.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from .objective import cut_weight


ARCHIVE_FORMAT_VERSION = 1


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata() -> dict:
    """Capture revision and working-tree state without requiring Git."""
    result = {"commit": None, "dirty": None}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        result["commit"] = revision.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        result["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf8", prefix=".manifest-",
            suffix=".json", dir=path.parent, delete=False
        ) as stream:
            temp = Path(stream.name)
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".archive-", suffix=".npz",
            dir=path.parent, delete=False
        ) as stream:
            temp = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:10]


def new_archive(
    root: Path, run_id: str, graph_path: Path, n: int, edges: tuple,
    k: int, settings: dict, *, jsonl_output: Path, argv: list[str] | None = None
) -> dict:
    """Create one case archive with its weighted graph and input identity."""
    path = Path(root) / run_id / f"{Path(graph_path).stem}-k{k}"
    path.mkdir(parents=True, exist_ok=False)
    i, j, weight = edges
    # Indices/weights correspond to the *normalized*, zero-based undirected
    # edge list used by cut_weight, not duplicate dense adjacency entries.
    _atomic_npz(
        path / "graph.npz",
        n=np.asarray(n, dtype=np.int64),
        i=np.asarray(i, dtype=np.int64),
        j=np.asarray(j, dtype=np.int64),
        weights=np.asarray(weight, dtype=np.float64),
    )
    manifest = {
        "schema_version": ARCHIVE_FORMAT_VERSION,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "graph": Path(graph_path).stem,
        "graph_source": str(Path(graph_path).resolve()),
        "graph_sha256": _file_sha256(graph_path),
        "graph_snapshot": "graph.npz",
        "n": int(n),
        "m": int(len(weight)),
        "k": int(k),
        "settings": settings,
        "results_jsonl": str(Path(jsonl_output).resolve()),
        "argv": argv,
        "git": _git_metadata(),
        "platform": {
            "python": sys.version.split()[0],
            "os": platform.platform(),
            "logical_cpus": os.cpu_count(),
        },
        "versions": {
            name: _version(name) for name in (
                "maxkcut", "numpy", "scipy", "cvxpy", "scs", "sdpa-python",
                "clarabel", "threadpoolctl"
            )
        },
        "methods": {},
    }
    _atomic_json(path / "manifest.json", manifest)
    return {"path": path, "manifest": manifest}


def update_manifest(archive: dict, **updates) -> None:
    archive["manifest"].update(updates)
    _atomic_json(archive["path"] / "manifest.json", archive["manifest"])


def method_archive_path(archive: dict, method: str) -> Path:
    # Method names are internal algorithm labels (not user-supplied paths).
    if not method or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in method):
        raise ValueError("Invalid method identifier for archive")
    return archive["path"] / f"{method}.npz"


def save_method(
    archive: dict, method: str, labels_list: list[np.ndarray],
    scores: list[float], *, k: int
) -> dict:
    """Compress all rounded partitions and summarize the entire score sample.

    k=2 uses packed bits: each vertex label occupies one bit rather than an
    8-byte integer. Larger k use the smallest common unsigned integer width.
    The restart axis preserves the exact JSONL restart ordering.
    """
    if not labels_list or len(labels_list) != len(scores):
        raise ValueError("One score is required for every saved partition")
    labels = np.stack(labels_list)
    if labels.ndim != 2 or labels.shape[1] != archive["manifest"]["n"]:
        raise ValueError("Wrong partition shape")
    if np.any(labels < 0) or np.any(labels >= k):
        raise ValueError("Partition labels must be in [0, k)")
    target = method_archive_path(archive, method)
    if k == 2:
        encoded = np.packbits(labels.astype(np.uint8), axis=1, bitorder="little")
        encoding = "bits-little"
    else:
        dtype = np.uint16 if k <= 65536 else np.uint32
        encoded = labels.astype(dtype)
        encoding = "uint16" if k <= 65536 else "uint32"
    score_array = np.asarray(scores, dtype=np.float64)
    _atomic_npz(
        target,
        labels=encoded,
        n=np.asarray(labels.shape[1], dtype=np.int64),
        k=np.asarray(k, dtype=np.int64),
        encoding=np.asarray(encoding),
        scores=score_array,
    )
    best_index = int(np.argmax(score_array))
    summary = {
        "partition_file": target.name,
        "encoding": encoding,
        "restarts": len(labels_list),
        "best_cut": float(score_array.max()),
        "best_restart": best_index,
        "worst_cut": float(score_array.min()),
        "mean_cut": float(score_array.mean()),
        "median_cut": float(np.median(score_array)),
        "sample_std_cut": float(score_array.std(ddof=1)) if len(scores) > 1 else None,
    }
    archive["manifest"]["methods"][method] = summary
    update_manifest(archive)
    return summary


def load_method(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all partitions as an integer matrix and associated cut values."""
    with np.load(path, allow_pickle=False) as arrays:
        n = int(arrays["n"].item())
        k = int(arrays["k"].item())
        encoding = str(arrays["encoding"].item())
        encoded = arrays["labels"]
        scores = arrays["scores"].astype(float)
    if encoding == "bits-little":
        labels = np.unpackbits(encoded, axis=1, bitorder="little")[:, :n]
    elif encoding in ("uint16", "uint32"):
        labels = encoded
    else:
        raise ValueError(f"Unknown partition encoding: {encoding}")
    if labels.ndim != 2 or labels.shape[1] != n or labels.shape[0] != len(scores):
        raise ValueError("Invalid saved partition dimensions")
    if np.any(labels >= k):
        raise ValueError("Partition label exceeds k")
    return labels.astype(np.int64), scores


def verify_archive(case_dir: Path) -> dict:
    """Independently rescore every archived cut from its graph snapshot.

    This verifies consistency of locally stored artifacts, NOT the original
    G-set graph's identity, optimality, or any SDP certificate.
    """
    case_dir = Path(case_dir)
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf8"))
    with np.load(case_dir / manifest["graph_snapshot"], allow_pickle=False) as g:
        n = int(g["n"].item())
        edges = (g["i"].copy(), g["j"].copy(), g["weights"].copy())
    if n != manifest["n"]:
        raise ValueError("Archive graph size does not match its manifest")
    outcomes = {}
    for method, summary in manifest["methods"].items():
        labels, scores = load_method(case_dir / summary["partition_file"])
        if labels.shape[1] != n or labels.shape[0] != summary["restarts"]:
            raise ValueError(f"Partition dimensions differ from manifest: {method}")
        rescored = np.asarray([cut_weight(edges, row) for row in labels])
        if not np.allclose(rescored, scores, atol=1e-8, rtol=0):
            raise ValueError(f"Cut weights do not match archived partitions: {method}")
        outcomes[method] = {
            "verified_partitions": int(len(labels)),
            "best_cut": float(rescored.max()),
        }
    return {
        "archive": str(case_dir),
        "status": manifest["status"],
        "graph": manifest["graph"],
        "graph_sha256": manifest["graph_sha256"],
        "methods": outcomes,
    }
