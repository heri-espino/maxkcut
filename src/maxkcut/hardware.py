"""CPU/BLAS execution controls and optional hardware inventory.

Controlling BLAS thread pools is not equivalent to making the entire SCS
solver parallel. SCS can use different native libraries for linear systems and
semidefinite cone projection; benchmark and inspect solver logs.
"""
from __future__ import annotations

from contextlib import nullcontext
import os
import shutil
import subprocess

from threadpoolctl import threadpool_info, threadpool_limits


def blas_limit(threads: int | None):
    """Temporarily limit recognized BLAS thread pools; None leaves defaults."""
    if threads is None:
        return nullcontext()
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    return threadpool_limits(limits=threads, user_api="blas")


def hardware_info() -> dict:
    """Report detected CPU count, BLAS backends and optional NVIDIA GPUs.

    The GPU query is informational; it never initializes CUDA or assumes that
    SCS was compiled with the NVIDIA cuDSS backend.
    """
    pools = []
    for pool in threadpool_info():
        if pool.get("user_api") == "blas":
            pools.append({
                "internal_api": pool.get("internal_api"),
                "num_threads": pool.get("num_threads"),
                "version": pool.get("version"),
                "threading_layer": pool.get("threading_layer"),
            })
    gpu_command = shutil.which("nvidia-smi")
    gpus = []
    gpu_note = "nvidia-smi not found; this does not establish GPU availability"
    if gpu_command:
        try:
            run = subprocess.run(
                [
                    gpu_command,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5, check=True,
            )
            gpus = [
                {"description": line.strip()}
                for line in run.stdout.splitlines() if line.strip()
            ]
            gpu_note = "NVIDIA driver query succeeded; SCS GPU support is unverified"
        except (subprocess.SubprocessError, OSError) as exc:
            gpu_note = f"nvidia-smi query failed: {type(exc).__name__}"
    return {
        "logical_cpus": os.cpu_count(),
        "blas_pools": pools,
        "nvidia_gpus": gpus,
        "gpu_note": gpu_note,
    }
