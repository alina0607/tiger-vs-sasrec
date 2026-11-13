"""Ranking metrics and the registry that `evaluate` reads.

A metric scores one test case from the rank of the correct item (1 is the top
of the list, `None` means it is not in the list) and a cutoff `k`. Each metric
registers itself with `@metric(name)`, and `evaluate` computes every registered
metric, so adding one needs no change there.
"""

import math
from collections.abc import Callable
from types import MappingProxyType

Metric = Callable[[int | None, int], float]

_registry: dict[str, Metric] = {}


def metric(name: str) -> Callable[[Metric], Metric]:
    """Register the decorated function as the metric `name`.

    The function itself is returned unchanged.

    Raises:
        ValueError: If a metric with this name is already registered.
    """

    def register(func: Metric) -> Metric:
        if name in _registry:
            raise ValueError(f"metric {name!r} is already registered")
        _registry[name] = func
        return func

    return register


def registered() -> MappingProxyType[str, Metric]:
    """Return a read-only view of the registered metrics, by name."""
    return MappingProxyType(_registry)


@metric("recall")
def recall(rank: int | None, k: int) -> float:
    """1 if the correct item is in the top `k`, else 0."""
    return 1.0 if rank is not None and rank <= k else 0.0


@metric("ndcg")
def ndcg(rank: int | None, k: int) -> float:
    """`1 / log2(rank + 1)` if the correct item is in the top `k`, else 0.

    With one correct item per case the ideal ranking scores 1, so no further
    normalisation is needed.
    """
    return 1 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0
