"""Optional PyTorch CPU/CUDA low-rank Max-Cut relaxation.

For k=2 only, parameterize a *feasible rank-r Gram matrix* as X=V V^T
with ||V_i||=1, and maximize

    F(V) = 1/2 sum_{(i,j) in E} w_ij (1 - <V_i,V_j>).

This is a NONCONVEX Burer-Monteiro-style restricted primal heuristic.
The vector score is NOT a certified SDP upper bound and is NOT the
result of solving the original full-rank convex SDP. Pairwise bounds
for k>2 are not covered by this implementation.

The sparse graph-times-dense-V and Adam updates happen on the selected
PyTorch device. GW and spherical K-Means rounders are applied separately
to the returned factors. No dense n-by-n matrix is formed.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse

from .io import edge_arrays


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is an optional dependency. Install a CUDA-enabled "
            "Windows wheel following https://pytorch.org/get-started/locally/ "
            "before using --device cuda."
        ) from exc
    return torch


def torch_device_info(device: str = "cuda") -> dict:
    """Inspect the actual selected PyTorch backend, without CPU fallback."""
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be 'cuda' or 'cpu'")
    torch = _torch()
    result = {
        "requested_device": device,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "gpu_name": None,
    }
    if device == "cuda":
        if not result["cuda_available"]:
            raise RuntimeError(
                "CUDA was explicitly requested, but torch.cuda.is_available() "
                "is False. Install a CUDA-enabled PyTorch wheel and confirm "
                "your NVIDIA driver works. Refusing silent CPU fallback."
            )
        result["gpu_name"] = torch.cuda.get_device_name(0)
    return result


def lowrank_objective(edges: tuple, vectors: np.ndarray) -> float:
    """Max-Cut vector objective of a unit-row factor, evaluated in float64."""
    left, right, weights = edges
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("vectors must be a finite n-by-r matrix")
    if (len(left) and
            (np.max(left) >= vectors.shape[0] or np.max(right) >= vectors.shape[0])):
        raise ValueError("factor has too few vertices")
    dot = np.einsum("ij,ij->i", vectors[left], vectors[right])
    return float(0.5 * np.sum(np.asarray(weights, dtype=float) * (1.0 - dot)))


def optimize_torch_lowrank(adjacency: sparse.spmatrix, *,
                           rank: int = 32, epochs: int = 200,
                           optimizer_starts: int = 2,
                           learning_rate: float = 0.05,
                           seed: int = 2026, device: str = "cuda",
                           log_every: int = 10) -> tuple[np.ndarray, dict]:
    """Run a projected Adam ascent on a sparse, signed Max-Cut vector SDP.

    This routine uses an explicit gradient; no PyTorch autograd graph or dense
    SDP variable is built. Each epoch's gradient is (A V)/2 for minimizing
    the negative vector objective, followed by row normalization.

    Returns (float32 unit-row n-by-r factors, diagnostics with all start
    histories). Results depend on initialization and hyperparameters and
    need independent benchmark validation; no global-optimality claim.
    """
    if not sparse.issparse(adjacency):
        raise ValueError("adjacency must be sparse; dense n-by-n input is disallowed")
    n = adjacency.shape[0]
    if adjacency.shape != (n, n) or n < 2:
        raise ValueError("adjacency must be a square graph")
    if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= n:
        raise ValueError("rank must be a positive integer <= n")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(optimizer_starts, int) or optimizer_starts < 1:
        raise ValueError("optimizer_starts must be a positive integer")
    if not isinstance(log_every, int) or log_every < 1:
        raise ValueError("log_every must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    edges = edge_arrays(adjacency)
    left, right, weights = edges
    if not (np.isfinite(weights).all() and np.isfinite(adjacency.data).all()):
        raise ValueError("graph has non-finite weights")
    # Valid signed undirected graphs have no self-loops and symmetric weights.
    if np.any(adjacency.diagonal()):
        raise ValueError("self-loops are unsupported")
    if (adjacency != adjacency.T).nnz:
        raise ValueError("graph must have symmetric weights")

    device_info = torch_device_info(device)
    torch = _torch()
    target = torch.device(device)
    first = torch.as_tensor(left, dtype=torch.long, device=target)
    second = torch.as_tensor(right, dtype=torch.long, device=target)
    weighted = torch.as_tensor(weights, dtype=torch.float32, device=target)
    indices = torch.stack((
        torch.cat((first, second)),
        torch.cat((second, first)),
    ))
    # Validate sparse COO indices once at construction. This follows the
    # PyTorch sparse-invariants API and avoids the "implicitly disabled"
    # warning observed on native Windows CUDA builds.
    with torch.sparse.check_sparse_tensor_invariants():
        matrix = torch.sparse_coo_tensor(
            indices, torch.cat((weighted, weighted)), (n, n),
            device=target, dtype=torch.float32,
        ).coalesce()
    edge_total = 0.5 * torch.sum(weighted)
    if device == "cuda":
        torch.cuda.synchronize(target)
        torch.cuda.reset_peak_memory_stats(target)
    clock = time.monotonic()

    histories = []
    best_factor = None
    best_objective = -np.inf
    best_start = None

    def evaluate(v):
        # The actual vector objective, not the rounded graph cut.
        return float((
            edge_total -
            0.5 * torch.sum(weighted * torch.sum(v[first] * v[second], dim=1))
        ).item())

    for start in range(optimizer_starts):
        child = np.random.SeedSequence([seed, start])
        child_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        random = torch.Generator(device=target)
        random.manual_seed(child_seed)
        with torch.no_grad():
            initial = torch.randn(
                (n, rank), generator=random, device=target, dtype=torch.float32
            )
            initial = torch.nn.functional.normalize(initial, p=2, dim=1)
        v = torch.nn.Parameter(initial)
        optimizer = torch.optim.Adam([v], lr=learning_rate)

        for epoch in range(epochs + 1):
            if epoch == 0 or epoch % log_every == 0 or epoch == epochs:
                with torch.no_grad():
                    value = evaluate(v.detach())
                if not np.isfinite(value):
                    raise FloatingPointError("Nonfinite low-rank objective")
                histories.append({
                    "optimizer_start": start,
                    "optimizer_seed": child_seed,
                    "epoch": epoch,
                    "vector_objective": value,
                })
                if value > best_objective:
                    best_objective = value
                    best_factor = v.detach().cpu().numpy().copy()
                    best_start = start
            if epoch == epochs:
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                # d(-F)/dV = (A V)/2 for a symmetric adjacency A.
                v.grad = 0.5 * torch.sparse.mm(matrix, v.detach())
            optimizer.step()
            with torch.no_grad():
                v.copy_(torch.nn.functional.normalize(v, p=2, dim=1))

    if device == "cuda":
        torch.cuda.synchronize(target)
        peak = int(torch.cuda.max_memory_allocated(target))
    else:
        peak = None
    elapsed = time.monotonic() - clock
    if best_factor is None:
        raise RuntimeError("No valid low-rank factor produced")
    row_error = float(np.max(np.abs(np.linalg.norm(best_factor, axis=1) - 1)))
    if row_error > 1e-4:
        raise RuntimeError("Low-rank vector normalization failed")
    exact_value = lowrank_objective(edges, best_factor)
    info = {
        "backend": "torch_lowrank",
        "optimization_type": "nonconvex_lowrank_primal_heuristic",
        "sdp_upper_bound_certified": False,
        "rank": rank,
        "epochs": epochs,
        "optimizer_starts": optimizer_starts,
        "learning_rate": float(learning_rate),
        "seed": seed,
        "device": device,
        "torch_version": device_info["torch_version"],
        "torch_cuda_version": device_info["torch_cuda_version"],
        "gpu_name": device_info["gpu_name"],
        "gpu_peak_memory_bytes": peak,
        "optimizer_seconds": elapsed,
        "lowrank_vector_objective": exact_value,
        "best_optimizer_start": best_start,
        "max_row_norm_error": row_error,
        "optimizer_history": histories,
    }
    return best_factor, info
