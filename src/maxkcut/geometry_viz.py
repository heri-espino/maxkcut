"""Read-only geometry and rounding-trace tools for archived Max-Cut studies.

Never runs an SDP optimizer or EVA solver. The only numerical work here is
projection, inspection of an existing factor, and optional ROUNDING replay.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .archive import load_method, verify_archive
from .eva import _identity, _prepared_graph, load_eva_factor
from .io import load_graph, edge_arrays
from .objective import cut_weight

DEFAULT_EVA = Path("results/gset/experiment002-eva")
DEFAULT_SDP_DIRS = (
    Path("reports/gset/full-sdp-batch"),
    Path("reports/gset/full-sdp-threads32"),
)
CACHE_DIR = Path("reports/gset/sdp_cache")


def repo_root():
    return Path(__file__).resolve().parents[2]


def _portable_path(root: Path, value: str) -> Path:
    """Resolve saved Windows or POSIX repository-relative paths.

    Archived paths may contain Windows absolute paths from another machine.
    Refuse anything outside the current repository; fall back to discoverable
    local archives instead of following foreign machine paths.
    """
    raw = str(value).replace("\\", "/")
    if ":" in raw or raw.startswith("//"):
        raise ValueError("Foreign absolute path cannot be trusted")
    candidate = Path(raw)
    root = Path(root).resolve()
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Archive path escapes the repository")
    return candidate


def discover_eva_cases(root=None, source=DEFAULT_EVA):
    root = repo_root() if root is None else Path(root)
    folder = root / source / "cases"
    names = []
    for file in folder.glob("G*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf8"))
            if data.get("status") in ("converged", "iteration_limit"):
                names.append(file.stem)
        except (ValueError, OSError):
            pass
    return sorted(set(names), key=lambda name: int(name[1:]))


def load_eva_artifacts(name, root=None, source=DEFAULT_EVA):
    """Read graph, EVA vectors and metadata after checking exact graph identity."""
    root = repo_root() if root is None else Path(root)
    folder = root / source
    graph_path = root / "data" / f"{name}.txt"
    case = json.loads((folder / "cases" / f"{name}.json").read_text(encoding="utf8"))
    graph_sha = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    if case.get("graph_sha256") != graph_sha or case.get("graph") != name:
        raise ValueError("Graph input differs from the saved EVA case")
    graph = load_graph(graph_path)
    factor_path = folder / "factors" / f"{name}.npz"
    vectors, meta = load_eva_factor(factor_path)
    provenance = case.get("provenance", {})
    csr = _prepared_graph(graph)
    if provenance.get("optimizer_backend") == "numba":
        csr.sort_indices()
    identity = _identity(csr, vectors.shape[1], provenance["seed"], provenance["tolerance"])
    if provenance.get("optimizer_backend") == "numba":
        identity["algorithm"] = "eva-sequential-numba-v1"
    if (vectors.shape[0] != graph.shape[0] or
            meta.get("identity") != identity or
            int(meta.get("sweeps", -1)) != case.get("factor_sweeps")):
        raise ValueError("The EVA factor and saved case have different identities")
    return graph, vectors, case, hashlib.sha256(factor_path.read_bytes()).hexdigest()


def _archive_path(case, root, name, source):
    try:
        archive = _portable_path(root, case["archive_path"])
        if archive.is_dir():
            return archive
    except (KeyError, ValueError):
        pass
    # The run ID is in the saved case, not invented or guessed.
    base = root / source / "archives"
    candidates = []
    for manifest in base.glob(f"*/{name}-k2/manifest.json"):
        record = json.loads(manifest.read_text(encoding="utf8"))
        if record.get("graph_sha256") == case.get("graph_sha256"):
            candidates.append(manifest.parent)
    if len(candidates) != 1:
        raise ValueError("Could not identify a unique saved archive for this graph")
    return candidates[0]


def archived_roundings(name, case, factor_sha=None, root=None, source=DEFAULT_EVA,
                       include_extensions=True):
    """Return {method: [(global_restart, score, vertex_labels), ...]}.

    Verifies archived graph/partitions against the exact canonical graph,
    and extensions against the frozen factor checksum. No reclustering.
    """
    root = repo_root() if root is None else Path(root)
    main = _archive_path(case, root, name, source)
    verified = verify_archive(main)
    if (verified["status"] != "complete" or
            verified["graph"] != name or
            verified["graph_sha256"] != case["graph_sha256"]):
        raise ValueError("Base rounding archive is not verified against the case")
    results = {}
    for method in verified["methods"]:
        labels, scores = load_method(main / f"{method}.npz")
        results[method] = [(i, float(score), label) for i, (label, score)
                           in enumerate(zip(labels, scores))]
    if include_extensions and factor_sha is not None:
        extension = root / source / "rounding-extensions"
        for manifest in sorted((extension / "segments").glob(f"{name}-*/**/r*.json")):
            info = json.loads(manifest.read_text(encoding="utf8"))
            if (info.get("graph_sha256") != case["graph_sha256"] or
                    info.get("factor_sha256") != factor_sha):
                continue
            folder = _portable_path(root, info["archive_path"])
            check = verify_archive(folder)
            method = info["method"]
            if check["status"] != "complete" or check["graph"] != name:
                raise ValueError(f"Invalid rounding extension: {manifest}")
            labels, scores = load_method(folder / f"{method}.npz")
            start = int(info["start_restart"])
            for offset, (label, score) in enumerate(zip(labels, scores)):
                results.setdefault(method, []).append((start + offset, float(score), label))
    for method, values in results.items():
        ids = [entry[0] for entry in values]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Overlapping archived restart indices for {method}")
        values.sort(key=lambda entry: entry[0])
    return results


def load_convex_embedding(name, root=None, cache_dir=CACHE_DIR, max_n=7500):
    """Top-three Gram eigenvectors from an EXISTING CVXPY cache; no solver.

    This is an uncentered spectral Gram embedding, not full-rank PCA or
    a new SDP solve. Large full Gram arrays are intentionally guarded.
    """
    root = repo_root() if root is None else Path(root)
    graph_file = root / "data" / f"{name}.txt"
    graph = load_graph(graph_file)
    n = graph.shape[0]
    if n > max_n:
        raise ValueError(f"Dense Gram visualization disabled for n={n} > {max_n}")
    graph_sha = hashlib.sha256(graph_file.read_bytes()).hexdigest()
    files = sorted((root / cache_dir).glob(f"{name}-k2-*.npz"))
    if not files:
        raise FileNotFoundError(f"No cached convex SDP matrix for {name}")
    valid = []
    for path in files:
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"].item()))
            if meta["identity"].get("graph_sha256") == graph_sha:
                valid.append((path, meta))
    if not valid:
        raise ValueError("Cached SDP matrices do not match the canonical graph")
    path, meta = valid[0]
    with np.load(path, allow_pickle=False) as saved:
        x = np.asarray(saved["gram"], dtype=np.float64)
    if x.shape != (n, n) or not np.isfinite(x).all():
        raise ValueError("Invalid cached SDP matrix")
    x = (x + x.T) * 0.5
    eigenvalues, eigenvectors = eigsh(x, k=min(3, n - 1), which="LA", tol=1e-5)
    order = np.argsort(eigenvalues)[::-1]
    coords = eigenvectors[:, order] * np.sqrt(np.maximum(eigenvalues[order], 0))[None, :]
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    return graph, coords, {"solver": meta["identity"].get("solver"),
                            "cache": str(path), "eigenvalues": eigenvalues[order].tolist(),
                            "note": "Top-three uncentered Gram eigenvector projection"}


def sample_vertices(n, limit=3000, seed=2026):
    if limit < 1:
        raise ValueError("Sample limit must be positive")
    if n <= limit:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=limit, replace=False))


def project_vectors(vectors, kind="PCA", *, selected=None, seed=2026):
    """Project *existing* n-by-p vectors; nonlinear modes are exploratory."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Expected finite n-by-p vectors")
    if selected is not None:
        vectors = vectors[np.asarray(selected)]
    if kind == "PCA":
        from sklearn.decomposition import PCA
        dims = min(3, *vectors.shape)
        model = PCA(n_components=dims, svd_solver="randomized", random_state=seed)
        coords = model.fit_transform(vectors)
        return np.pad(coords, ((0, 0), (0, 3 - dims))), {
            "explained_variance_ratio": model.explained_variance_ratio_.tolist()
        }
    if kind == "t-SNE":
        if len(vectors) < 5:
            raise ValueError("t-SNE requires at least five points")
        from sklearn.manifold import TSNE
        perplexity = min(30., max(2., (len(vectors) - 1) / 3))
        coords = TSNE(n_components=3, perplexity=perplexity, init="pca",
                      learning_rate="auto", random_state=seed).fit_transform(vectors)
        return coords, {"perplexity": perplexity, "warning": "Distances are distorted"}
    if kind == "UMAP":
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ImportError("UMAP requires pip install umap-learn") from exc
        coords = UMAP(n_components=3, random_state=seed).fit_transform(vectors)
        return coords, {"warning": "Nonlinear geometry is distorted"}
    raise ValueError(f"Unknown projection: {kind}")


def vertex_metrics(graph):
    """Signed graphs: degree ignores weights; strength is sum of |w|."""
    return {
        "Degree": np.diff(graph.indptr).astype(int),
        "Absolute weighted degree": np.asarray(abs(graph).sum(axis=1)).ravel(),
        "Signed weighted degree": np.asarray(graph.sum(axis=1)).ravel(),
    }


def positive_edge_communities(graph, *, seed=2026, max_n=5000):
    """Topology-only heuristic: negative edges are EXCLUDED, not treated as affinities."""
    if graph.shape[0] > max_n:
        raise ValueError(f"Community detection limited to n <= {max_n}")
    import networkx as nx
    upper = sparse.triu(graph.maximum(0), 1).tocoo()
    network = nx.Graph()
    network.add_nodes_from(range(graph.shape[0]))
    network.add_weighted_edges_from(
        (int(i), int(j), float(w)) for i, j, w in zip(upper.row, upper.col, upper.data)
    )
    groups = nx.community.louvain_communities(network, seed=seed, weight="weight")
    labels = np.zeros(graph.shape[0], dtype=int)
    for group_number, members in enumerate(groups):
        labels[list(members)] = group_number
    return labels


def pairwise_cosines(vectors, number=10000, seed=2026):
    """Sample correlations without creating an n-by-n Gram matrix."""
    rng = np.random.default_rng(seed)
    n = len(vectors)
    i, j = rng.integers(0, n, size=(2, number))
    valid = i != j
    a, b = vectors[i[valid]], vectors[j[valid]]
    dots = np.einsum("ij,ij->i", a, b)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.clip(dots / np.maximum(norms, 1e-12), -1, 1)


def replay_kmeans(vectors, graph, name, method="K-MeansNM_EVA", *,
                  restart=0, seed=2026, max_iter=100):
    """Replay ROUNDING ONLY, faithful to rounding.py seed and centroid rules.

    Returns one frame per assignment iteration, including current cut and
    best-so-far cut. Does not evolve EVA's underlying vectors.
    """
    if method not in ("K-MeansNM_EVA", "K-Means2N_EVA"):
        raise ValueError("K-means replay requires a k-means method")
    independent = method == "K-Means2N_EVA"
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, int(name[1:]), 2, restart, 1 if independent else 0]
    ))
    k = 2
    centers = rng.standard_normal((k, vectors.shape[1]))
    if not independent:
        centers[1] = -centers[0]
    centers /= np.linalg.norm(centers, axis=1)[:, None]
    edges = edge_arrays(graph)
    frames, previous, best = [], None, -float("inf")
    for iteration in range(1, max_iter + 1):
        labels = np.argmax(vectors @ centers.T, axis=1)
        score = float(cut_weight(edges, labels))
        best = max(score, best)
        frames.append({"step": iteration, "labels": labels.copy(),
                       "cut": score, "best_cut": best,
                       "centroids": centers.copy()})
        if previous is not None and np.array_equal(previous, labels):
            break
        previous = labels.copy()
        for cluster in range(k):
            members = vectors[labels == cluster]
            if not len(members):
                centers[cluster] = vectors[rng.integers(len(vectors))]
            else:
                centroid = np.sum(members, axis=0)
                norm = np.linalg.norm(centroid)
                centers[cluster] = (centroid / norm if norm > 1e-12
                                    else vectors[rng.integers(len(vectors))])
    return frames


def load_sweep_log(name, root=None, source=DEFAULT_EVA):
    root = repo_root() if root is None else Path(root)
    path = root / source / "logs" / f"{name}-sweeps.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("graph") == name:
                rows.append(record)
    # Resumed logs may contain duplicate sweep indices; last wins.
    return sorted({int(row["sweep"]): row for row in rows}.values(),
                  key=lambda row: int(row["sweep"]))


def eva_coordinate_diagnostics(graph, vectors):
    """Explain the *current* EVA greedy coordinate update, without applying it.

    For h_i = sum_j w_ij y_j, the optimal local direction is -h_i/||h_i||
    and the gain from replacing y_i is (||h_i|| + y_i dot h_i)/2.
    This describes the frozen saved factor's coordinate stationarity;
    it does not create historical EVA iterates.
    """
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != graph.shape[0]:
        raise ValueError("Factor and graph vertex counts do not agree")
    field = graph @ vectors
    norms = np.linalg.norm(field, axis=1)
    dot = np.einsum("ij,ij->i", vectors, field)
    cosine = np.ones(len(vectors))
    active = norms > 1e-15
    cosine[active] = np.clip(-dot[active] / norms[active], -1, 1)
    direction = np.zeros_like(field)
    direction[active] = -field[active] / norms[active, None]
    gains = np.zeros(len(vectors))
    gains[active] = np.maximum(0., 0.5 * (norms[active] + dot[active]))
    return {
        "angle_degrees": np.rad2deg(np.arccos(cosine)),
        "potential_cut_relaxation_gain": gains,
        "field_norm": norms,
        "target_direction": direction,
        "active": active,
    }


def discover_eva_snapshots(name, root=None, source=DEFAULT_EVA):
    """Find optional immutable pre-recorded vectors (never reconstruct them).

    Snapshot schema: trajectory/Gxx/sweep-00001.npz with arrays
    vectors[n,p], sweep=integer, graph_sha256=str.
    These snapshots are NOT currently emitted by experiment002.
    """
    root = repo_root() if root is None else Path(root)
    folder = root / source / "trajectory" / name
    return sorted(folder.glob("sweep-*.npz"))


def load_eva_snapshots(name, graph_sha, shape, *, root=None,
                       source=DEFAULT_EVA, maximum=32):
    """Read only saved trajectory frames; consistent graph and dimensions."""
    all_paths = discover_eva_snapshots(name, root, source)
    if not all_paths:
        return []
    selection = sample_vertices(len(all_paths), min(maximum, len(all_paths)))
    frames = []
    for index in selection:
        path = all_paths[index]
        with np.load(path, allow_pickle=False) as saved:
            y = np.asarray(saved["vectors"], dtype=float).copy()
            sweep = int(saved["sweep"].item())
            hash_value = str(saved["graph_sha256"].item())
        if (hash_value != graph_sha or y.shape != tuple(shape)
                or not np.isfinite(y).all()
                or np.max(np.abs(np.linalg.norm(y, axis=1) - 1)) > 1e-5):
            raise ValueError(f"Invalid EVA trajectory frame {path}")
        frames.append({"sweep": sweep, "vectors": y})
    return sorted(frames, key=lambda item: item["sweep"])
