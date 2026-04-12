"""
HybridTransport: auto-selects P2P or SHM transport based on GPU topology.

If ALL GPU pairs in the process group have P2P access → CudaPeerTransport (fast path).
If ANY pair lacks P2P (typical for consumer GPUs) → ShmTransport (host-staged path).

Both transports expose the same interface:
  publish_tensor(tensor, sync)
  fetch_chunk(peer, dst, src_offset_bytes, size_bytes, sync)
  wait_for_peer(peer)
  invalidate_cache()
  sync_peer_stream(peer)
  enable_peer_access(devices)
  set_session(session_id)   [ShmTransport only, no-op for P2P]
  is_shm: bool
  streams: dict[int, Stream]
"""

import logging

import torch

from . import cuda_ipc
from .transport import CudaPeerTransport
from .shm_transport import ShmTransport

logger = logging.getLogger("cuda_direct.hybrid_transport")


def _all_pairs_have_p2p(device_count: int) -> bool:
    """Return True iff every ordered pair of CUDA devices can do P2P."""
    current = torch.cuda.current_device()
    for i in range(device_count):
        for j in range(device_count):
            if i == j:
                continue
            try:
                if not cuda_ipc.can_access_peer(i, j):
                    logger.info("No P2P between device %d and device %d — using SHM transport", i, j)
                    return False
            except RuntimeError:
                logger.info("P2P check failed for device %d↔%d — using SHM transport", i, j)
                return False
    return True


class HybridTransport:
    """
    Wrapper that delegates to either CudaPeerTransport or ShmTransport.

    Instantiate via HybridTransport.create() which probes P2P availability.
    All attribute accesses are forwarded to the inner transport so callers
    do not need to know which backend is active.
    """

    def __init__(self, inner: CudaPeerTransport | ShmTransport):
        self._inner = inner

    @classmethod
    def create(cls, rank: int, world_size: int) -> "HybridTransport":
        """
        Probe GPU topology and return a HybridTransport wrapping the best transport.

        Called once per process during ProcessGroupCudaDirect.__init__.
        """
        device_count = torch.cuda.device_count()
        has_p2p = _all_pairs_have_p2p(device_count)

        if has_p2p:
            inner = CudaPeerTransport(rank, world_size)
            logger.info(
                "HybridTransport: P2P available — using CudaPeerTransport  "
                "rank=%d  world=%d", rank, world_size
            )
        else:
            inner = ShmTransport(rank, world_size)
            logger.info(
                "HybridTransport: no P2P — using ShmTransport (host-staged)  "
                "rank=%d  world=%d", rank, world_size
            )
        return cls(inner)

    # ---------------------------------------------------------------
    #  Transparent delegation
    # ---------------------------------------------------------------

    def __getattr__(self, name: str):
        # Called only when normal attribute lookup fails, i.e. attr is on inner
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value):
        if name == "_inner":
            super().__setattr__(name, value)
        else:
            setattr(self._inner, name, value)
