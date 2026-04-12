"""
Smart auto-activation for the cuda_direct backend.

Provides a single entry point that detects the system configuration
and activates the appropriate level of backend integration.

Usage:
    # Auto-detect everything (recommended)
    from cuda_direct_backend.auto import activate
    activate()

    # Explicit mode selection
    activate(mode="full")         # Register + NCCL spoof + collective dispatch
    activate(mode="backend_only") # Register as "cuda_direct" backend, no spoofing
    activate(mode="diagnose")     # Print diagnostics, don't activate anything

    # Programmatic check before activation
    from cuda_direct_backend.auto import should_activate
    if should_activate():
        activate()
"""

import os
import logging

logger = logging.getLogger("cuda_direct")

_ACTIVATED = False
_ACTIVATION_MODE = None


def should_activate() -> bool:
    """
    Check if cuda_direct should be activated on this system.

    Returns True when ALL of these are true:
      - Running on Windows
      - CUDA is available
      - 2+ GPUs detected
      - At least one GPU pair has P2P access
      - Real NCCL is NOT available (otherwise use that)
    """
    from .diagnostics import diagnose
    result = diagnose(verbose=False)
    return result["recommendation"] == "cuda_direct"


def activate(mode: str = "auto", verbose: bool = True):
    """
    Activate the cuda_direct backend with the appropriate level of integration.

    Args:
        mode: One of:
            "auto"         - Detect system and pick the right mode automatically.
                             On Windows with 2+ P2P GPUs: activates "full".
                             On Linux with NCCL: does nothing.
                             On single GPU: does nothing.
            "full"         - Register backend + NCCL spoof + collective dispatch.
                             Use this for FSDP or when code hardcodes backend="nccl".
            "backend_only" - Register "cuda_direct" as a backend option.
                             No monkey-patching. User must explicitly pass
                             backend="cuda_direct" to init_process_group().
            "diagnose"     - Print system diagnostics and exit. No activation.
        verbose: Print status messages.

    Returns:
        str: The activation mode that was applied ("full", "backend_only",
             "skipped", or "diagnose").
    """
    global _ACTIVATED, _ACTIVATION_MODE

    if mode == "diagnose":
        from .diagnostics import diagnose
        diagnose(verbose=True)
        return "diagnose"

    if _ACTIVATED:
        if verbose:
            logger.info(f"cuda_direct already activated in '{_ACTIVATION_MODE}' mode.")
        return _ACTIVATION_MODE

    if mode == "auto":
        mode = _auto_detect_mode(verbose)

    if mode == "skipped":
        if verbose:
            logger.info("cuda_direct: Activation skipped (not needed on this system).")
        return "skipped"

    if mode == "backend_only":
        from .diagnostics import _is_windows
        if not _is_windows():
            logger.warning(
                "cuda_direct: activate(mode='backend_only') has no effect on non-Windows "
                "systems. NCCL is available and should be used instead. Skipping."
            )
            return "skipped"
        _activate_backend_only(verbose)
        _ACTIVATED = True
        _ACTIVATION_MODE = "backend_only"
        return "backend_only"

    if mode == "full":
        from .diagnostics import _is_windows
        if not _is_windows():
            logger.warning(
                "cuda_direct: activate(mode='full') has no effect on non-Windows systems. "
                "NCCL is available and will be used instead. Skipping activation."
            )
            return "skipped"
        _activate_full(verbose)
        _ACTIVATED = True
        _ACTIVATION_MODE = "full"
        return "full"

    raise ValueError(
        f"Unknown activation mode: '{mode}'. "
        "Use 'auto', 'full', 'backend_only', or 'diagnose'."
    )


def _auto_detect_mode(verbose: bool) -> str:
    """Determine the right activation mode based on system diagnostics."""
    from .diagnostics import diagnose, _is_windows, _is_linux

    result = diagnose(verbose=verbose)

    if result["recommendation"] == "single_gpu":
        return "skipped"

    if result["recommendation"] == "nccl":
        # Linux with real NCCL — don't interfere
        return "skipped"

    if result["recommendation"] == "gloo" and not _is_windows():
        # Non-Windows without NCCL — can't help
        return "skipped"

    if result["recommendation"] == "gloo" and _is_windows():
        # Windows but no P2P — can't do direct GPU transfers
        if verbose:
            logger.warning(
                "cuda_direct: No P2P access between GPUs. "
                "Falling back to Gloo. Training will work but GPU-GPU "
                "transfers will go through system RAM."
            )
        return "skipped"

    if result["recommendation"] == "cuda_direct":
        # Windows + multi-GPU + P2P available
        return "full"

    return "skipped"


def _activate_backend_only(verbose: bool):
    """Register cuda_direct as an available backend without any monkey-patching."""
    from .process_group import register_backend
    try:
        register_backend()
        if verbose:
            logger.info(
                "cuda_direct: Registered as backend. "
                "Use dist.init_process_group(backend='cuda_direct') to activate."
            )
    except RuntimeError:
        # Already registered
        if verbose:
            logger.info("cuda_direct: Backend already registered.")


def _activate_full(verbose: bool):
    """Register backend + apply NCCL spoof + collective dispatch patches."""
    from .patch_torch import patch_torch_distributed
    patch_torch_distributed()
    if verbose:
        logger.info(
            "cuda_direct: Full activation complete. "
            "NCCL calls will be transparently redirected to cuda_direct. "
            "Gloo calls are not affected."
        )
