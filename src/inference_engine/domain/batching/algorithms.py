
from collections.abc import Callable, Hashable
from typing import TypeVar

T = TypeVar("T")


def greedy_batch_collection(
    requests: list[T], target_size: int, max_wait_ms: int, current_age_ms: int
) -> tuple[list[T], bool]:
    """
    Greedy batching algorithm: collect up to target size or until timeout.

    Returns:
        Tuple of (collected_requests, should_wait_more)

    Args:
        requests: Pool of available requests
        target_size: Desired batch size
        max_wait_ms: Maximum age before forcing batch
        current_age_ms: Age of oldest request

    Returns:
        Tuple of collected requests and whether more requests are coming
    """
    if not requests:
        return [], False

    if current_age_ms >= max_wait_ms:
        # Timeout: collect what we have
        return requests[:target_size], False

    if len(requests) >= target_size:
        # Full batch ready
        return requests[:target_size], False

    # Need to wait for more requests
    return [], True


def affinity_batch_collection(
    requests: list[T], target_size: int, affinity_key: Callable[[T], Hashable]
) -> list[T]:
    """
    Affinity-based batching: group requests sharing an affinity key.

    Args:
        requests: Pool of available requests
        target_size: Maximum batch size
        affinity_key: Function to extract affinity key from request

    Returns:
        List of requests grouped by affinity
    """
    if not requests:
        return []

    # Group by affinity
    groups: dict[Hashable, list[T]] = {}
    for req in requests:
        key = affinity_key(req)
        if key not in groups:
            groups[key] = []
        groups[key].append(req)

    # Return largest group up to target_size
    largest_group = max(groups.values(), key=len)
    return largest_group[:target_size]


def throughput_optimized_batching(
    recent_latencies: list[float], current_batch_size: int, min_size: int, max_size: int
) -> int:
    """
    Adjust batch size to maximize throughput while maintaining latency.

    Uses exponential moving average of latency.

    Args:
        recent_latencies: Recent latency measurements
        current_batch_size: Current batch size
        min_size: Minimum allowed batch size
        max_size: Maximum allowed batch size

    Returns:
        Adjusted batch size
    """
    if not recent_latencies:
        return min_size

    # EMA of latency
    alpha = 0.3
    avg_latency = recent_latencies[0]
    for lat in recent_latencies[1:]:
        avg_latency = alpha * lat + (1 - alpha) * avg_latency

    # Adjust based on throughput/latency tradeoff
    # Higher latency → smaller batches
    # Lower latency → larger batches

    if avg_latency > 200:  # Too slow
        new_size = int(current_batch_size * 0.8)
    elif avg_latency < 50:  # Too fast, can handle more
        new_size = int(current_batch_size * 1.2)
    else:
        new_size = current_batch_size

    return max(min_size, min(max_size, new_size))
