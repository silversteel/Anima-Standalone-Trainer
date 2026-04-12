"""
Monkey-patching utility to spoof NCCL presence on Windows and seamlessly
redirect PyTorch FSDP communication to the `cuda_direct` backend.
"""

import sys
import torch
import torch.distributed as dist
import logging

try:
    from torch.distributed.device_mesh import DeviceMesh
    _HAS_DEVICE_MESH = True
except ImportError:
    _HAS_DEVICE_MESH = False

logger = logging.getLogger("cuda_direct_patch")

_PATCHED = False

def patch_torch_distributed():
    global _PATCHED
    if _PATCHED:
        return

    import os
    if os.name != "nt":
        logger.warning(
            "cuda_direct: patch_torch_distributed() called on a non-Windows system. "
            "NCCL is available and should be used instead. Skipping patch."
        )
        return

    logger.info("Hijacking torch.distributed to enable cuda_direct FSDP...")

    # 1. Spoof NCCL availability to bypass global asserts (esp. in Accelerate/DeviceMesh)
    dist.is_nccl_available = lambda: True
    if hasattr(dist, "distributed_c10d"):
        dist.distributed_c10d._NCCL_AVAILABLE = True
        dist.distributed_c10d.is_nccl_available = lambda: True

    # Register the backend immediately
    try:
        from cuda_direct_backend import process_group
        process_group.register_backend()
    except Exception as e:
        logger.warning(f"Failed to register cuda_direct directly: {e}")

    # 2. Redirect init_process_group
    _orig_init_process_group = dist.init_process_group

    def _mock_init_process_group(backend=None, *args, **kwargs):
        import os
        import torch
        _is_win_multigpu = (
            os.name == "nt"
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
        )

        if (backend == "nccl" or backend == dist.Backend.NCCL) and _is_win_multigpu:
            logger.info("Intercepted init_process_group(backend='nccl'). Redirecting to cuda_direct...")
            backend = "cuda_direct"

        return _orig_init_process_group(backend, *args, **kwargs)

    dist.init_process_group = _mock_init_process_group
    if hasattr(dist, "distributed_c10d"):
        dist.distributed_c10d.init_process_group = _mock_init_process_group

    # 3. Patch DeviceMesh — force backend to cuda_direct when CUDA is requested
    if _HAS_DEVICE_MESH:
        try:
            from torch.distributed.device_mesh import init_device_mesh
            _orig_init_device_mesh = init_device_mesh

            def _mock_init_device_mesh(device_type, mesh_shape, *args, **kwargs):
                # DeviceMesh defaults to NCCL for CUDA — our init_process_group
                # intercept handles the redirect, but some code paths check
                # backend name directly. Ensure "nccl" → "cuda_direct" mapping
                # is active before DeviceMesh creates sub-process-groups.
                return _orig_init_device_mesh(device_type, mesh_shape, *args, **kwargs)

            import torch.distributed.device_mesh
            torch.distributed.device_mesh.init_device_mesh = _mock_init_device_mesh
        except Exception as e:
            logger.warning(f"Could not patch DeviceMesh: {e}")

    # 4. Patch dist._all_gather_base / reduce_scatter_base / allreduce_coalesced
    # to dispatch to our custom ProcessGroup methods
    _orig_all_gather_base = getattr(dist.distributed_c10d, "_all_gather_base", None)
    if _orig_all_gather_base:
        def _mock_all_gather_base(output_tensor, input_tensor, group=None, async_op=False):
            if group is None:
                group = dist.distributed_c10d._get_default_group()
            if hasattr(group, "_allgather_base"):
                return group._allgather_base(output_tensor, input_tensor)
            return _orig_all_gather_base(output_tensor, input_tensor, group=group, async_op=async_op)

        dist.distributed_c10d._all_gather_base = _mock_all_gather_base
        dist._all_gather_base = _mock_all_gather_base

    _orig_reduce_scatter_base = getattr(dist.distributed_c10d, "_reduce_scatter_base", None)
    if _orig_reduce_scatter_base:
        def _mock_reduce_scatter_base(output_tensor, input_tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
            if group is None:
                group = dist.distributed_c10d._get_default_group()
            if hasattr(group, "_reduce_scatter_base"):
                class _Opt:
                    pass
                opts = _Opt()
                opts.reduceOp = op
                return group._reduce_scatter_base(output_tensor, input_tensor, opts)
            return _orig_reduce_scatter_base(output_tensor, input_tensor, op=op, group=group, async_op=async_op)

        dist.distributed_c10d._reduce_scatter_base = _mock_reduce_scatter_base
        dist._reduce_scatter_base = _mock_reduce_scatter_base

    # 5. Patch allreduce_coalesced for DDP bucketed gradient sync
    _orig_allreduce_coalesced = getattr(dist.distributed_c10d, "all_reduce_coalesced", None)
    if _orig_allreduce_coalesced:
        def _mock_allreduce_coalesced(tensors, op=dist.ReduceOp.SUM, group=None, async_op=False):
            if group is None:
                group = dist.distributed_c10d._get_default_group()
            if hasattr(group, "allreduce_coalesced"):
                class _Opt:
                    pass
                opts = _Opt()
                opts.reduceOp = op
                return group.allreduce_coalesced(tensors, opts)
            return _orig_allreduce_coalesced(tensors, op=op, group=group, async_op=async_op)

        dist.distributed_c10d.all_reduce_coalesced = _mock_allreduce_coalesced
        if hasattr(dist, "all_reduce_coalesced"):
            dist.all_reduce_coalesced = _mock_allreduce_coalesced

    _PATCHED = True
    logger.info("Successfully hijacked torch.distributed.")

if __name__ == "__main__":
    patch_torch_distributed()
    print(f"NCCL Available after patch: {dist.is_nccl_available()}")
