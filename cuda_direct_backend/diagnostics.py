"""
System diagnostics for cuda_direct backend.

Detects OS, GPU topology, P2P capability, and recommends the optimal
distributed backend for the current machine.
"""

import os
import sys
import platform
import logging

logger = logging.getLogger("cuda_direct")


def _is_windows() -> bool:
    return os.name == "nt"


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _get_gpu_info() -> list[dict]:
    """Return list of GPU dicts with name, memory, device index."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory_mb": props.total_mem // (1024 * 1024),
                "major": props.major,
                "minor": props.minor,
            })
        return gpus
    except Exception:
        return []


def _get_p2p_matrix(gpu_count: int) -> list[list[bool]]:
    """Build NxN matrix of P2P accessibility between GPUs."""
    from . import cuda_ipc
    matrix = [[False] * gpu_count for _ in range(gpu_count)]
    for i in range(gpu_count):
        matrix[i][i] = True  # self-access always works
        for j in range(i + 1, gpu_count):
            try:
                can = cuda_ipc.can_access_peer(i, j)
                matrix[i][j] = can
                matrix[j][i] = can
            except RuntimeError:
                matrix[i][j] = False
                matrix[j][i] = False
    return matrix


def _check_real_nccl() -> bool:
    """Check if real NCCL (not our spoof) is available."""
    try:
        import torch.distributed as dist
        # If we already patched, this returns True but it's our spoof
        from cuda_direct_backend.patch_torch import _PATCHED
        if _PATCHED:
            return False
        return dist.is_nccl_available()
    except Exception:
        return False


def diagnose(verbose: bool = True) -> dict:
    """
    Run full system diagnostics and return a capabilities dict.

    Returns:
        dict with keys:
            os: str ('windows', 'linux', 'other')
            gpu_count: int
            gpus: list[dict]
            cuda_available: bool
            nccl_native: bool  (real NCCL, not spoofed)
            p2p_matrix: list[list[bool]]
            p2p_full: bool  (all GPU pairs can P2P)
            p2p_partial: bool  (some but not all pairs can P2P)
            recommendation: str ('cuda_direct', 'nccl', 'gloo', 'single_gpu')
            reason: str
    """
    import torch

    result = {
        "os": "windows" if _is_windows() else ("linux" if _is_linux() else "other"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
        "nccl_native": False,
        "p2p_matrix": [],
        "p2p_full": False,
        "p2p_partial": False,
        "recommendation": "gloo",
        "reason": "",
    }

    # GPU info
    result["gpus"] = _get_gpu_info()

    # Real NCCL check
    result["nccl_native"] = _check_real_nccl()

    # Single GPU or no GPU — nothing to do
    if result["gpu_count"] < 2:
        result["recommendation"] = "single_gpu" if result["gpu_count"] == 1 else "gloo"
        result["reason"] = (
            "Single GPU detected — distributed backend not needed"
            if result["gpu_count"] == 1
            else "No CUDA GPUs detected — falling back to Gloo (CPU)"
        )
        if verbose:
            _print_report(result)
        return result

    # P2P matrix
    result["p2p_matrix"] = _get_p2p_matrix(result["gpu_count"])
    all_p2p = all(
        result["p2p_matrix"][i][j]
        for i in range(result["gpu_count"])
        for j in range(result["gpu_count"])
    )
    any_p2p = any(
        result["p2p_matrix"][i][j]
        for i in range(result["gpu_count"])
        for j in range(result["gpu_count"])
        if i != j
    )
    result["p2p_full"] = all_p2p
    result["p2p_partial"] = any_p2p and not all_p2p

    # Recommendation
    if result["nccl_native"] and _is_linux():
        result["recommendation"] = "nccl"
        result["reason"] = "Linux with native NCCL available — use NCCL for best performance"
    elif _is_windows() and all_p2p:
        result["recommendation"] = "cuda_direct"
        result["reason"] = (
            f"Windows with {result['gpu_count']} GPUs, all P2P accessible — "
            "cuda_direct uses direct GPU-GPU DMA (~22-25 GB/s). "
            "Gloo calls are not redirected."
        )
    elif _is_windows() and any_p2p:
        result["recommendation"] = "cuda_direct"
        result["reason"] = (
            f"Windows with {result['gpu_count']} GPUs, partial P2P — "
            "cuda_direct uses P2P for accessible pairs, host-staged Zero-Copy "
            "SHM for others (~22-28 GB/s). Gloo calls are not redirected."
        )
    elif _is_windows() and not any_p2p:
        result["recommendation"] = "cuda_direct"
        result["reason"] = (
            f"Windows with {result['gpu_count']} GPUs, no P2P (consumer GPU) — "
            "cuda_direct uses host-staged Zero-Copy SHM transport (~22-28 GB/s, "
            "3x faster than Gloo due to GPU-side reduction)."
        )
    else:
        result["recommendation"] = "gloo"
        result["reason"] = "No better backend available"

    if verbose:
        _print_report(result)
    return result


def _print_report(result: dict):
    """Print a human-readable diagnostic report."""
    print("=" * 60)
    print("  cuda_direct backend — System Diagnostics")
    print("=" * 60)
    print(f"  OS:              {result['platform']}")
    print(f"  Python:          {result['python']}")
    print(f"  PyTorch:         {result['torch']}")
    print(f"  CUDA available:  {result['cuda_available']}")
    print(f"  GPU count:       {result['gpu_count']}")
    print(f"  Native NCCL:     {result['nccl_native']}")

    if result["gpus"]:
        print()
        print("  GPUs:")
        for gpu in result["gpus"]:
            print(f"    [{gpu['index']}] {gpu['name']} "
                  f"({gpu['total_memory_mb']} MB, SM {gpu['major']}.{gpu['minor']})")

    if result["p2p_matrix"]:
        n = result["gpu_count"]
        print()
        print("  P2P Access Matrix:")
        header = "       " + "  ".join(f"GPU{j}" for j in range(n))
        print(header)
        for i in range(n):
            row = f"  GPU{i}  " + "  ".join(
                " OK " if result["p2p_matrix"][i][j] else " -- "
                for j in range(n)
            )
            print(row)

    print()
    rec = result["recommendation"].upper()
    transport = ""
    if rec == "CUDA_DIRECT":
        if result.get("p2p_full"):
            transport = "  [transport: P2P direct ~22-25 GB/s]"
        elif result.get("p2p_partial"):
            transport = "  [transport: hybrid P2P+SHM]"
        else:
            transport = "  [transport: SHM Zero-Copy ~22-28 GB/s]"
    print(f"  Recommendation:  {rec}{transport}")
    print(f"  Reason:          {result['reason']}")
    print("=" * 60)


if __name__ == "__main__":
    diagnose()
