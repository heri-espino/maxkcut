"""Full-rank GPU eigenprojection ADMM for the same convex Max-Cut SDP.

RESEARCH BACKEND: not a patched CVXPY/SCS solver and not the separate
nonconvex low-rank solver. No certified dual bound is computed.

Split X=Z in max <L/4,X>, diag(X)=1, Z>=0 (PSD):
    X = Z-U+L/(4*rho); diag(X)=1
    Z = Proj_PSD(X+U) (full torch.linalg.eigh)
    U = U+X-Z.
"""
from __future__ import annotations

import time
import numpy as np
from scipy import sparse


def full_sdp_admm_torch(
    adjacency: sparse.spmatrix, *, device: str = "cuda",
    precision: str = "float32", rho: float = 1.0,
    max_iters: int = 120, eps_abs: float = 1e-4,
    eps_rel: float = 1e-3, log_every: int = 10,
) -> tuple[np.ndarray, dict]:
    """Return an approximate full-size PSD SDP matrix and convergence diagnostics.

    Same full convex feasible set as SCS/SDPA, but a different first-order
    algorithm, and float32 can lose precision. Never label an
    unconverged ADMM matrix a certified SDP optimum.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required") from exc
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; never silently fall back to CPU")
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be float32 or float64")
    if rho <= 0 or max_iters < 1 or eps_abs <= 0 or eps_rel <= 0 or log_every < 1:
        raise ValueError("Invalid ADMM parameters")
    a = sparse.csr_matrix(adjacency, dtype=np.float64)
    n = a.shape[0]
    if a.shape != (n, n) or n < 2 or a.diagonal().any() or (a != a.T).nnz:
        raise ValueError("Requires symmetric loop-free graph")
    if not np.isfinite(a.data).all():
        raise ValueError("Nonfinite graph weights")
    dtype = getattr(torch, precision)
    dev = torch.device(device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(dev)
    lap = sparse.diags(np.asarray(a.sum(axis=1)).ravel()) - a
    c = torch.as_tensor((lap.toarray() / 4).copy(), dtype=dtype, device=dev)
    z = torch.eye(n, dtype=dtype, device=dev)
    u = torch.zeros_like(z)
    history = []
    converged = False
    current = None
    t0 = time.perf_counter()
    for iteration in range(1, max_iters + 1):
        x = z - u + c / rho
        x.fill_diagonal_(1.)
        m = x + u
        m = (m + m.mT) * 0.5
        eigenvalues, q = torch.linalg.eigh(m)
        old_z = z
        z = (q * eigenvalues.clamp_min(0).unsqueeze(0)) @ q.mT
        z = (z + z.mT) * 0.5
        u = u + x - z
        if iteration == 1 or iteration % log_every == 0 or iteration == max_iters:
            r = float(torch.linalg.vector_norm(x - z).item())
            d = float((rho * torch.linalg.vector_norm(z - old_z)).item())
            nx = float(torch.linalg.vector_norm(x).item())
            nz = float(torch.linalg.vector_norm(z).item())
            nu = float(torch.linalg.vector_norm(u).item())
            primal_tol = n * eps_abs + eps_rel * max(nx, nz)
            dual_tol = n * eps_abs + eps_rel * rho * nu
            current = {
                "iteration": iteration, "primal_residual": r,
                "dual_residual": d, "primal_tolerance": primal_tol,
                "dual_tolerance": dual_tol,
                "min_unprojected_eigenvalue": float(eigenvalues.min().item()),
            }
            history.append(current)
            print(
                f"ADMM {device}/{precision} iter={iteration} "
                f"primal={r:.4e}/{primal_tol:.4e} "
                f"dual={d:.4e}/{dual_tol:.4e}", flush=True,
            )
            if r <= primal_tol and d <= dual_tol:
                converged = True
                break
    if device == "cuda":
        torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0
    diagonal = z.diagonal()
    if float(diagonal.min().item()) <= 1e-8:
        raise RuntimeError("PSD projection produced near-zero diagonal")
    scale = torch.rsqrt(diagonal)
    candidate = (z * scale.unsqueeze(0)) * scale.unsqueeze(1)
    candidate = (candidate + candidate.mT) * 0.5
    candidate.fill_diagonal_(1.)
    objective = float(torch.sum(c * candidate).item())
    matrix = candidate.detach().cpu().numpy().astype(np.float64)
    meta = {
        "solver": "full-psd-admm", "optimization_type": "full_convex_psd_admm",
        "device": device, "precision": precision,
        "rho": rho, "eps_abs": eps_abs, "eps_rel": eps_rel,
        "max_iters": max_iters, "iterations": iteration,
        "converged": converged,
        "solver_status": "admm_residual_converged" if converged
                         else "admm_iteration_limit",
        "residuals": current, "history": history,
        "sdp_objective_primal_uncertified": objective,
        "sdp_seconds": elapsed,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(dev)) if device == "cuda" else None
        ),
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(dev) if device == "cuda" else None,
    }
    return matrix, meta


def psd_projection_eigh(
    matrix: np.ndarray, *, device: str = "cpu",
    precision: str = "float64",
) -> tuple[np.ndarray, dict]:
    """Time one full PSD projector with device transfers and kernel separately.

    This is a kernel-level diagnostic, NOT an SCS GPU solver backend.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch required") from exc
    if device not in ("cpu", "cuda") or precision not in ("float32", "float64"):
        raise ValueError("Invalid backend or precision")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.allclose(a, a.T):
        raise ValueError("Expected symmetric square matrix")
    target = torch.device(device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(target)
    start = time.perf_counter()
    t = torch.as_tensor(a.copy(), device=target, dtype=getattr(torch, precision))
    if device == "cuda":
        torch.cuda.synchronize(target)
    start_kernel = time.perf_counter()
    values, vectors = torch.linalg.eigh(t)
    result = (vectors * values.clamp_min(0).unsqueeze(0)) @ vectors.mT
    if device == "cuda":
        torch.cuda.synchronize(target)
    kernel_seconds = time.perf_counter() - start_kernel
    host = result.cpu().numpy().astype(np.float64)
    return host, {
        "device": device, "precision": precision,
        "kernel_seconds": kernel_seconds,
        "end_to_end_seconds": time.perf_counter() - start,
        "min_input_eigenvalue": float(values.min().item()),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(target)) if device == "cuda" else None
        ),
    }
