"""Full-rank GPU eigenprojection ADMM for the same convex Max-Cut SDP.

RESEARCH BACKEND: not a patched CVXPY/SCS solver and not the separate
nonconvex low-rank solver. No certified dual bound is computed.

Split X=Z in max <L/4,X>, diag(X)=1, Z>=0 (PSD):
    X = Z-U+L/(4*rho); diag(X)=1
    Z = Proj_PSD(X+U) (full torch.linalg.eigh)
    U = U+X-Z.
"""
from __future__ import annotations

import math
import time
import numpy as np
from scipy import sparse


def balanced_penalty_update(
    rho: float, primal: float, dual: float, *,
    mu: float = 10.0, scale: float = 2.0,
    rho_min: float = 0.0625, rho_max: float = 16.0,
) -> tuple[float, str | None]:
    """Residual-balancing rho update (unscaled ADMM primal/dual residuals).

    A pure scalar function, testable without importing torch. The caller
    MUST rescale its *scaled* dual U by rho_old / rho_new whenever rho
    changes, keeping the underlying (unscaled) dual Y = rho * U fixed.
    """
    values = (rho, primal, dual, mu, scale, rho_min, rho_max)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("ADMM penalty parameters and residuals must be finite")
    if (rho <= 0 or primal < 0 or dual < 0 or mu <= 1 or scale <= 1
            or rho_min <= 0 or rho_max < rho_min
            or not rho_min <= rho <= rho_max):
        raise ValueError("Invalid ADMM penalty balancing parameters")
    # Avoid the pathological 0/0 case and unnecessary changes at equality.
    if primal > mu * dual:
        updated = min(rho * scale, rho_max)
        return (updated, "increase") if updated > rho else (rho, None)
    if dual > mu * primal:
        updated = max(rho / scale, rho_min)
        return (updated, "decrease") if updated < rho else (rho, None)
    return rho, None


def full_sdp_admm_torch(
    adjacency: sparse.spmatrix, *, device: str = "cuda",
    precision: str = "float32", rho: float = 1.0,
    max_iters: int = 120, eps_abs: float = 1e-4,
    eps_rel: float = 1e-3, log_every: int = 10,
    adaptive_rho: bool = False, rho_update_interval: int = 25,
    rho_balance_mu: float = 10.0, rho_scale: float = 2.0,
    rho_min: float = 0.0625, rho_max: float = 16.0,
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
    if (not math.isfinite(rho) or not math.isfinite(eps_abs)
            or not math.isfinite(eps_rel) or rho <= 0
            or max_iters < 1 or eps_abs <= 0 or eps_rel <= 0
            or log_every < 1 or not isinstance(adaptive_rho, bool)
            or not isinstance(rho_update_interval, int)
            or isinstance(rho_update_interval, bool)
            or rho_update_interval < 1):
        raise ValueError("Invalid ADMM parameters")
    # Validate balancing controls without restricting legacy fixed rho:
    # a fixed-rho experiment may intentionally use values outside
    # the adaptive clipping bounds, preserving its original behavior.
    checked_rho = rho if adaptive_rho else min(max(rho, rho_min), rho_max)
    balanced_penalty_update(
        checked_rho, 0.0, 0.0, mu=rho_balance_mu, scale=rho_scale,
        rho_min=rho_min, rho_max=rho_max,
    )
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
    rho_events = []
    initial_rho = float(rho)
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
        check_residuals = (
            iteration == 1 or iteration % log_every == 0
            or iteration == max_iters
            or (adaptive_rho and iteration % rho_update_interval == 0)
        )
        if check_residuals:
            r = float(torch.linalg.vector_norm(x - z).item())
            d = float((rho * torch.linalg.vector_norm(z - old_z)).item())
            nx = float(torch.linalg.vector_norm(x).item())
            nz = float(torch.linalg.vector_norm(z).item())
            nu = float(torch.linalg.vector_norm(u).item())
            primal_tol = n * eps_abs + eps_rel * max(nx, nz)
            dual_tol = n * eps_abs + eps_rel * rho * nu
            current = {
                "iteration": iteration, "rho": float(rho),
                "primal_residual": r,
                "dual_residual": d, "primal_tolerance": primal_tol,
                "dual_tolerance": dual_tol,
                "min_unprojected_eigenvalue": float(eigenvalues.min().item()),
            }
            history.append(current)
            if (iteration == 1 or iteration % log_every == 0
                    or iteration == max_iters):
                print(
                    f"ADMM {device}/{precision} iter={iteration} "
                    f"rho={rho:.6g} "
                    f"primal={r:.4e}/{primal_tol:.4e} "
                    f"dual={d:.4e}/{dual_tol:.4e}", flush=True,
                )
            if r <= primal_tol and d <= dual_tol:
                converged = True
                break
            # Only adjust after evaluating the stopping test for the
            # current rho. The update changes *future* iterations; this
            # preserves correct residual units for the reported step.
            if (adaptive_rho and iteration < max_iters
                    and iteration % rho_update_interval == 0):
                new_rho, reason = balanced_penalty_update(
                    rho, r, d, mu=rho_balance_mu, scale=rho_scale,
                    rho_min=rho_min, rho_max=rho_max,
                )
                if reason is not None:
                    old_rho = rho
                    # Scaled dual: U=Y/rho. Keep Y constant under the
                    # penalty update or convergence can be invalidated.
                    u.mul_(old_rho / new_rho)
                    rho = new_rho
                    event = {
                        "iteration": iteration, "previous_rho": old_rho,
                        "new_rho": rho, "reason": reason,
                        "primal_residual": r, "dual_residual": d,
                        "scaled_dual_multiplier": old_rho / rho,
                    }
                    rho_events.append(event)
                    print(
                        f"ADMM rho update iter={iteration}: "
                        f"{old_rho:.6g} -> {rho:.6g} ({reason}; "
                        f"primal={r:.4e}, dual={d:.4e})", flush=True,
                    )
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
        "rho": initial_rho, "rho_final": float(rho),
        "adaptive_rho": adaptive_rho,
        "rho_update_interval": rho_update_interval,
        "rho_balance_mu": rho_balance_mu, "rho_scale": rho_scale,
        "rho_min": rho_min, "rho_max": rho_max,
        "rho_update_count": len(rho_events), "rho_events": rho_events,
        "eps_abs": eps_abs, "eps_rel": eps_rel,
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
