import json
import logging
import torch
import torch.distributed as dist
from torch.futures import Future
import uuid
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from torch._C._distributed_c10d import (
    _create_work_from_future,
    AllreduceOptions,
    BroadcastOptions,
    BarrierOptions,
    AllgatherOptions,
    ReduceScatterOptions
)
from .sync import SharedMemorySync
from .transport import CudaPeerTransport
from .hybrid_transport import HybridTransport
from .collectives import CollectivesImpl

logger = logging.getLogger("cuda_direct.pg")

def ret_work(ret):
    """Return a completed Work object (synchronous path)."""
    fut = Future()
    fut.set_result(ret)
    return _create_work_from_future(fut)

def _async_work(pool, fn, result_val):
    """Submit fn to the thread pool and return an async Work object.

    Used for collectives whose input tensors are **freshly computed** on the
    caller's CUDA stream (e.g. allreduce on gradients, reduce-scatter after
    backward).  The worker CPU-blocks on caller_done.synchronize() so that
    gradient data is committed to device memory before any cross-rank IPC
    read or SHM publish.

    For collectives whose inputs are **stable** (not being written by the
    caller's current compute), use _allgather_work() instead — it skips the
    caller sync and lets the pool worker start immediately.

    Stream safety: two levels of synchronization are required.

    GPU-level (worker_stream.wait_stream): protects any GPU ops issued on
    worker_stream inside fn() — e.g. tensor.contiguous(), copy_() — from
    starting before caller_stream finishes. The CPU thread is not blocked.

    CPU-level (caller_done.synchronize): required for cross-rank operations
    that precede an inter-process CPU barrier. After the barrier, peer
    processes issue IPC DMA reads against our tensor pointers. Those DMAs
    have no visibility into our GPU stream ordering. We must guarantee our
    gradient computation is committed to device memory BEFORE we signal the
    barrier. worker_stream.wait_stream alone does not provide this: the CPU
    proceeds to exchange_handles+barrier immediately (both are CPU ops),
    potentially before caller_stream has drained on the GPU. Recording an
    event and synchronizing it on the worker thread blocks only the worker
    (not the main thread) until the gradients are ready.
    """
    # Capture now, on the calling thread, before the pool takes over.
    caller_stream = torch.cuda.current_stream()
    caller_device = torch.cuda.current_device()
    # Record caller's position so the worker can CPU-wait for it.
    caller_done = torch.cuda.Event()
    caller_done.record(caller_stream)

    fut = Future()
    def _worker():
        try:
            # The ThreadPoolExecutor thread starts on device 0 by default.
            # Set it to the caller's device so IPC opens, cudaMemcpyAsync, and
            # stream operations all run in the correct CUDA context.
            torch.cuda.set_device(caller_device)

            worker_stream = torch.cuda.current_stream()

            # GPU-level: protects GPU ops on worker_stream within fn().
            worker_stream.wait_stream(caller_stream)

            # CPU-level: block this worker thread until caller's compute
            # (e.g. backward pass) is committed to device memory. Required
            # before any cross-rank IPC read or SHM publish that follows a
            # CPU barrier. The main training thread is NOT blocked here.
            caller_done.synchronize()

            fn()
            fut.set_result(result_val)
        except Exception as e:
            fut.set_exception(e)
    pool.submit(_worker)
    return _create_work_from_future(fut)


def _allgather_work(pool, fn, result_val, caller_device):
    """Lightweight async dispatch for allgather — no caller stream dependency.

    FSDP allgather inputs are **stored sharded parameters**, not tensors
    being actively written by the caller's compute stream.  Skipping
    caller_done.synchronize() (and the worker_stream.wait_stream dependency
    on the caller) lets the pool worker run the allgather immediately
    instead of blocking until the caller's backward compute drains.

    This is the key enabler for FSDP communication-compute overlap: without
    it, the single-worker pool is occupied waiting for backward compute,
    which prevents a prefetched allgather from starting until the previous
    reduce-scatter (plus its caller sync) finishes.

    A final torch.cuda.current_stream().synchronize() before setting the
    Future ensures all GPU work from the collective (local shard copy, peer
    DMAs) is committed to device memory before work.wait() returns.
    """
    fut = Future()
    def _worker():
        try:
            torch.cuda.set_device(caller_device)
            fn()
            # Commit all GPU ops (local copy_, DMA) before signaling
            # completion.  The caller will use the output tensor on its own
            # stream after work.wait() — without this sync, the local
            # shard copy could still be in the GPU queue.
            torch.cuda.current_stream().synchronize()
            fut.set_result(result_val)
        except Exception as e:
            fut.set_exception(e)
    pool.submit(_worker)
    return _create_work_from_future(fut)

def _drain_pool(pool: ThreadPoolExecutor) -> None:
    """Block until all in-flight pool jobs complete.

    Called before any synchronous collective that calls invalidate_cache()
    on the main thread.  Without this, a synchronous collective can race with
    a still-running async worker from a previous _async_work/_allgather_work
    dispatch: both try to use or unregister the same SHM/IPC resources.
    The pool has max_workers=1, so submitting a no-op and waiting for it
    guarantees the previous job has finished.
    """
    pool.submit(lambda: None).result()


class ProcessGroupCudaDirect(dist.ProcessGroup):
    def __init__(self, store, rank, world_size, timeout):
        import os
        if os.name != "nt":
            raise RuntimeError(
                "cuda_direct backend is Windows-only. "
                "On Linux, use NCCL: dist.init_process_group(backend='nccl'). "
                "Do not call register_backend() or activate() on Linux."
            )
        super().__init__(rank, world_size)
        self.store = store
        self._rank = rank
        self._world_size = world_size

        logger.info("ProcessGroupCudaDirect init  rank=%d  world_size=%d  device=%d",
                    rank, world_size, torch.cuda.current_device())

        self.timeout_sec = float(timeout.total_seconds()) if hasattr(timeout, 'total_seconds') else float(timeout) if timeout else 300.0
        logger.info("ProcessGroupCudaDirect timeout configured to %.1f seconds", self.timeout_sec)

        # Proper initialization synchronization
        if rank == 0:
            session_id = str(uuid.uuid4().hex)
            store.set("cuda_direct_session", session_id.encode('utf-8'))
            self.sync = SharedMemorySync(rank, world_size, session_id, create=True, timeout=self.timeout_sec)
            store.set("cuda_direct_ready", b"1")
            logger.info("rank=0  shared memory created  session=%s", session_id)
        else:
            logger.debug("rank=%d  waiting for rank-0 shared memory init...", rank)
            store.wait(["cuda_direct_ready"])
            session_id = store.get("cuda_direct_session").decode('utf-8')
            self.sync = SharedMemorySync(rank, world_size, session_id, create=False, timeout=self.timeout_sec)
            logger.info("rank=%d  attached to shared memory  session=%s", rank, session_id)

        self.session_id = session_id

        # HybridTransport probes P2P and picks CudaPeerTransport or ShmTransport
        self.transport = HybridTransport.create(rank, world_size)

        # Give ShmTransport the session ID for SHM region naming
        self.transport.set_session(session_id)

        device_count = torch.cuda.device_count()
        logger.info("rank=%d  probing P2P access  device_count=%d", rank, device_count)
        self.transport.enable_peer_access(list(range(device_count)))

        # Topology discovery: rank 0 probes bandwidth and shares ring order
        ring_order = self._discover_ring(store, rank, world_size)

        self.collectives = CollectivesImpl(rank, world_size, self.transport, self.sync,
                                           ring_order=ring_order)

        # Async dispatch engine: allows FSDP to overlap communication with compute.
        # max_workers=1 ensures collectives execute in order (no race conditions).
        self._async_pool = ThreadPoolExecutor(max_workers=1,
                                              thread_name_prefix="cuda_direct_async")
        logger.info("rank=%d  async dispatch engine ready", rank)

    def _discover_ring(self, store, rank, world_size):
        """Discover optimal ring order. Rank 0 probes, others wait.

        Short-circuits when SHM transport is active: SHM uses a flat star
        topology (each rank publishes to its own slot, all peers read directly).
        Ring ordering has no effect on SHM bandwidth — every pair goes through
        the same PCIe bus regardless of adjacency — so probing is wasted time.
        """
        if world_size < 3:
            return list(range(world_size))

        if self.transport.is_shm:
            logger.info(
                "rank=%d  SHM transport active (no P2P) — skipping topology "
                "discovery, ring ordering has no effect on star topology", rank)
            return list(range(world_size))

        if rank == 0:
            try:
                from .topology import discover_topology
                ring_order, bw_matrix = discover_topology(world_size)
                store.set("cuda_direct_ring", json.dumps(ring_order).encode("utf-8"))
                logger.info(f"Topology ring computed: {ring_order}")
            except Exception as e:
                logger.warning(f"Topology discovery failed: {e}. Using default order.")
                ring_order = list(range(world_size))
                store.set("cuda_direct_ring", json.dumps(ring_order).encode("utf-8"))
        else:
            store.wait(["cuda_direct_ring"])
            ring_order = json.loads(store.get("cuda_direct_ring").decode("utf-8"))

        return ring_order

    def getBackendName(self):
        return "cuda_direct"

    def allreduce(self, tensor_list, opts=None):
        op = opts.reduceOp if opts is not None else dist.ReduceOp.SUM
        def _do_allreduce():
            # Invalidate inside the worker so it runs after the previous async
            # job completes (max_workers=1 ensures sequential execution).
            # Calling invalidate_cache() on the main thread races with the
            # previous worker's in-flight cudaMemcpyAsync / SHM access.
            self.transport.invalidate_cache()
            try:
                self.collectives.allreduce(tensor_list, op)
            except Exception as e:
                shapes = [t.shape for t in tensor_list]
                dtypes = [t.dtype for t in tensor_list]
                raise RuntimeError(
                    f"cuda_direct allreduce failed  rank={self._rank}  op={op}  "
                    f"shapes={shapes}  dtypes={dtypes}"
                ) from e
        return _async_work(self._async_pool, _do_allreduce, tensor_list)

    def allreduce_coalesced(self, tensor_list, opts=None):
        op = opts.reduceOp if opts is not None else dist.ReduceOp.SUM
        _drain_pool(self._async_pool)
        self.transport.invalidate_cache()
        try:
            self.collectives.allreduce_coalesced(tensor_list, op)
        except Exception as e:
            raise RuntimeError(
                f"cuda_direct allreduce_coalesced failed  rank={self._rank}  op={op}  "
                f"num_tensors={len(tensor_list)}"
            ) from e
        return ret_work(tensor_list)

    def broadcast(self, tensor_list, opts=None):
        root = opts.rootRank if opts is not None else 0
        _drain_pool(self._async_pool)
        self.transport.invalidate_cache()
        try:
            self.collectives.broadcast(tensor_list, root)
        except Exception as e:
            shapes = [t.shape for t in tensor_list]
            raise RuntimeError(
                f"cuda_direct broadcast failed  rank={self._rank}  root={root}  "
                f"shapes={shapes}"
            ) from e
        return ret_work(tensor_list)

    def barrier(self, opts=None):
        try:
            self.collectives.barrier()
        except TimeoutError as e:
            raise TimeoutError(
                f"cuda_direct barrier timed out  rank={self._rank}  "
                f"world_size={self._world_size}. "
                f"Check that all ranks are still alive and calling barrier."
            ) from e
        return ret_work(None)

    def allgather(self, output_tensors_list, input_tensor_list, opts=None):
        """AllGather with list-of-lists interface expected by DDP.

        Args:
            output_tensors_list: list[list[Tensor]] — one list per input tensor,
                each inner list has world_size tensors.
            input_tensor_list: list[Tensor] — tensors to gather.
        """
        if len(output_tensors_list) != len(input_tensor_list):
            raise ValueError(
                f"cuda_direct allgather: output_tensors_list length "
                f"({len(output_tensors_list)}) != input_tensor_list length "
                f"({len(input_tensor_list)})"
            )
        for output_tensors, input_tensor in zip(output_tensors_list, input_tensor_list):
            if len(output_tensors) != self._world_size:
                raise ValueError(
                    f"cuda_direct allgather: output_tensors has {len(output_tensors)} "
                    f"slots but world_size={self._world_size}"
                )
            chunk_size = input_tensor.numel()
            _drain_pool(self._async_pool)
            self.transport.invalidate_cache()
            flat_output = torch.empty(
                chunk_size * self._world_size,
                dtype=input_tensor.dtype,
                device=input_tensor.device
            )
            try:
                self.collectives._allgather_base(flat_output, input_tensor)
                self.collectives.barrier()
            except Exception as e:
                raise RuntimeError(
                    f"cuda_direct allgather failed  rank={self._rank}  "
                    f"input_shape={input_tensor.shape}  dtype={input_tensor.dtype}"
                ) from e

            for i, out_t in enumerate(output_tensors):
                out_t.copy_(flat_output[i * chunk_size:(i + 1) * chunk_size].view_as(out_t))
        return ret_work(output_tensors_list)

    def _allgather_base(self, output_tensor, input_tensor, opts=None):
        def _do_allgather():
            # Invalidate inside the worker (runs after previous job finishes).
            # See allreduce() comment for why this must NOT be on the main thread.
            self.transport.invalidate_cache()
            try:
                self.collectives._allgather_base(output_tensor, input_tensor)
                self.collectives.barrier()
            except Exception as e:
                raise RuntimeError(
                    f"cuda_direct _allgather_base failed  rank={self._rank}  "
                    f"input_shape={input_tensor.shape}  output_shape={output_tensor.shape}  "
                    f"dtype={input_tensor.dtype}"
                ) from e
        # Use lightweight dispatch: allgather inputs are stored FSDP shards,
        # not tensors being written by the caller's current compute.
        # Skipping caller_done.synchronize() unblocks the pool worker so
        # prefetched allgathers can start immediately instead of waiting
        # behind a reduce-scatter that is blocked on backward compute.
        return _allgather_work(self._async_pool, _do_allgather, [output_tensor],
                               torch.cuda.current_device())

    def alltoall_base(self, output_tensor, input_tensor,
                      output_split_sizes, input_split_sizes, opts=None):
        _drain_pool(self._async_pool)
        self.transport.invalidate_cache()
        out_splits = output_split_sizes if output_split_sizes else None
        in_splits = input_split_sizes if input_split_sizes else None
        try:
            self.collectives.all_to_all_single(
                output_tensor, input_tensor, out_splits, in_splits)
            self.collectives.barrier()
        except Exception as e:
            raise RuntimeError(
                f"cuda_direct alltoall_base failed  rank={self._rank}  "
                f"input_shape={input_tensor.shape}  output_shape={output_tensor.shape}  "
                f"dtype={input_tensor.dtype}"
            ) from e
        return ret_work([output_tensor])

    def _reduce_scatter_base(self, output_tensor, input_tensor, opts=None):
        op = opts.reduceOp if opts is not None else dist.ReduceOp.SUM
        def _do_reduce_scatter():
            # Invalidate inside the worker (runs after previous job finishes).
            # See allreduce() comment for why this must NOT be on the main thread.
            self.transport.invalidate_cache()
            try:
                self.collectives._reduce_scatter_base(output_tensor, input_tensor, op)
                self.collectives.barrier()
            except Exception as e:
                raise RuntimeError(
                    f"cuda_direct _reduce_scatter_base failed  rank={self._rank}  op={op}  "
                    f"input_shape={input_tensor.shape}  output_shape={output_tensor.shape}  "
                    f"dtype={input_tensor.dtype}"
                ) from e
        return _async_work(self._async_pool, _do_reduce_scatter, [output_tensor])

def _create_cuda_direct_pg(prefix_store, rank, world_size, timeout):
    return ProcessGroupCudaDirect(prefix_store, rank, world_size, timeout)

def register_backend():
    import os
    if os.name != "nt":
        logger.warning(
            "cuda_direct: register_backend() called on a non-Windows system. "
            "NCCL is available and should be used instead. Skipping registration."
        )
        return
    try:
        dist.Backend.register_backend("cuda_direct", _create_cuda_direct_pg, devices=["cuda"])
    except RuntimeError:
        pass  # already registered, harmless
