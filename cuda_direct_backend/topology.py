"""
Topology discovery and optimal ring computation.

Probes P2P bandwidth between all GPU pairs via timed cudaMemcpyAsync,
then finds a ring ordering that maximizes the minimum link bandwidth
(i.e. avoids putting slow links like x4 PCIe in the ring).
"""

import time
import itertools
import logging
import torch
from . import cuda_ipc

logger = logging.getLogger("cuda_direct")


def probe_p2p_bandwidth(
    dev_i: int, dev_j: int,
    size_mb: int = 32,
    warmup: int = 3,
    iterations: int = 8,
) -> float:
    """Measure effective unidirectional DMA bandwidth from dev_i → dev_j in GB/s.

    Uses torch tensor copy with synchronization for reliable timing.
    Returns 0.0 if P2P is not possible between the pair.
    """
    numel = (size_mb * 1024 * 1024) // 4  # float32

    try:
        with torch.cuda.device(dev_i):
            src = torch.empty(numel, dtype=torch.float32, device=f"cuda:{dev_i}")
        with torch.cuda.device(dev_j):
            dst = torch.empty(numel, dtype=torch.float32, device=f"cuda:{dev_j}")

        # Warmup
        for _ in range(warmup):
            dst.copy_(src, non_blocking=False)

        torch.cuda.synchronize(dev_i)
        torch.cuda.synchronize(dev_j)

        start = time.perf_counter()
        for _ in range(iterations):
            dst.copy_(src, non_blocking=False)
        end = time.perf_counter()

        elapsed = end - start
        total_bytes = numel * 4 * iterations
        bandwidth = total_bytes / elapsed / 1e9

        del src, dst
        torch.cuda.empty_cache()
        return bandwidth

    except Exception as e:
        logger.warning(f"Failed to probe bandwidth {dev_i}→{dev_j}: {e}")
        return 0.0


def build_bandwidth_matrix(gpu_count: int) -> list[list[float]]:
    """Probe all GPU pairs and return an NxN bandwidth matrix (GB/s).

    matrix[i][j] = bandwidth from GPU i to GPU j.
    Diagonal is set to infinity (self-transfer).
    """
    matrix = [[0.0] * gpu_count for _ in range(gpu_count)]

    for i in range(gpu_count):
        matrix[i][i] = float("inf")
        for j in range(gpu_count):
            if i != j:
                bw = probe_p2p_bandwidth(i, j)
                matrix[i][j] = bw
                try:
                    p2p = cuda_ipc.can_access_peer(i, j)
                except Exception:
                    p2p = False
                transport = "P2P" if p2p else "CPU-staged fallback"
                logger.info(f"  GPU {i} → GPU {j}: {bw:.1f} GB/s [{transport}]")

    return matrix


def _ring_min_bandwidth(ring: list[int], bw_matrix: list[list[float]]) -> float:
    """Return the minimum link bandwidth in a ring."""
    n = len(ring)
    min_bw = float("inf")
    for i in range(n):
        j = (i + 1) % n
        bw = bw_matrix[ring[i]][ring[j]]
        min_bw = min(min_bw, bw)
    return min_bw


def find_optimal_ring(bw_matrix: list[list[float]]) -> list[int]:
    """Find the ring ordering that maximizes the minimum link bandwidth.

    For N <= 8: brute-force all permutations (7! = 5040 at most).
    For N > 8: greedy nearest-neighbor heuristic starting from each node,
               pick the best.
    """
    n = len(bw_matrix)
    if n <= 1:
        return list(range(n))
    if n == 2:
        return [0, 1]

    if n <= 8:
        return _brute_force_ring(bw_matrix)
    else:
        return _greedy_ring(bw_matrix)


def _brute_force_ring(bw_matrix: list[list[float]]) -> list[int]:
    """Try all permutations, return ring with best min-bandwidth."""
    n = len(bw_matrix)
    best_ring = list(range(n))
    best_min_bw = _ring_min_bandwidth(best_ring, bw_matrix)

    # Fix node 0 to avoid rotational duplicates
    for perm in itertools.permutations(range(1, n)):
        ring = [0] + list(perm)
        min_bw = _ring_min_bandwidth(ring, bw_matrix)
        if min_bw > best_min_bw:
            best_min_bw = min_bw
            best_ring = ring

    return best_ring


def _greedy_ring(bw_matrix: list[list[float]]) -> list[int]:
    """Greedy nearest-neighbor: from each start node, always pick the
    unvisited neighbor with the highest bandwidth. Return the best ring."""
    n = len(bw_matrix)
    best_ring = None
    best_min_bw = -1.0

    for start in range(n):
        ring = [start]
        visited = {start}

        for _ in range(n - 1):
            current = ring[-1]
            # Pick unvisited neighbor with highest bandwidth
            best_next = -1
            best_bw = -1.0
            for candidate in range(n):
                if candidate not in visited and bw_matrix[current][candidate] > best_bw:
                    best_bw = bw_matrix[current][candidate]
                    best_next = candidate
            ring.append(best_next)
            visited.add(best_next)

        min_bw = _ring_min_bandwidth(ring, bw_matrix)
        if min_bw > best_min_bw:
            best_min_bw = min_bw
            best_ring = ring

    # Normalize so ring starts at 0
    idx = best_ring.index(0)
    best_ring = best_ring[idx:] + best_ring[:idx]
    return best_ring


def discover_topology(gpu_count: int) -> tuple[list[int], list[list[float]]]:
    """Full topology discovery: probe bandwidths and compute optimal ring.

    Returns:
        (ring_order, bw_matrix)
    """
    logger.info(f"Probing P2P bandwidth between {gpu_count} GPUs...")
    bw_matrix = build_bandwidth_matrix(gpu_count)

    ring = find_optimal_ring(bw_matrix)
    min_bw = _ring_min_bandwidth(ring, bw_matrix)
    logger.info(f"Optimal ring: {ring} (min link: {min_bw:.1f} GB/s)")

    return ring, bw_matrix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print("Running topology probe 10 times to average bandwidth...")
            total_bw = None
            for i in range(10):
                print(f"Pass {i+1}/10...")
                ring, bw_matrix = discover_topology(gpu_count)
                if total_bw is None:
                    total_bw = bw_matrix
                else:
                    for r in range(gpu_count):
                        for c in range(gpu_count):
                            if bw_matrix[r][c] != float('inf'):
                                total_bw[r][c] += bw_matrix[r][c]
            
            # Average
            for r in range(gpu_count):
                for c in range(gpu_count):
                    if total_bw[r][c] != float('inf'):
                        total_bw[r][c] /= 10
            
            final_ring = find_optimal_ring(total_bw)
            min_bw = _ring_min_bandwidth(final_ring, total_bw)
            print("\n" + "=" * 60)
            print("  FINAL AVERAGED TOPOLOGY")
            print("=" * 60)
            for i in range(gpu_count):
                for j in range(gpu_count):
                    if i != j:
                        print(f"  GPU {i} → GPU {j}: {total_bw[i][j]:.1f} GB/s")
            print("-" * 60)
            print(f"  Optimal ring: {final_ring} (average min link: {min_bw:.1f} GB/s)")
            print("=" * 60)
        else:
            print(f"Only {gpu_count} GPU(s) found. Need at least 2 for topology discovery.")
    else:
        print("CUDA not available.")
