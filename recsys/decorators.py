"""Decorators used across the package."""

import functools
import logging
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import ParamSpec, TypeVar

import torch

from recsys.runlog import current_run_log

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Record how long each call of `func` takes in the active `RunLog`.

    Calls that raise are not recorded.

    Args:
        func: The function to time.

    Returns:
        A wrapper with the same signature as `func`.
    """
    label = func.__qualname__

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = perf_counter()
        result = func(*args, **kwargs)
        seconds = perf_counter() - start
        log = current_run_log()
        if log is not None:
            log.add_timing(label, seconds)
        logger.debug("%s took %.3f s", label, seconds)
        return result

    return wrapper


@contextmanager
def seeded(seed: int) -> Iterator[None]:
    """Seed `random` and `torch` for a block, then restore their previous state.

    Works as `with seeded(67):` and, like every `contextmanager`, as a decorator
    `@seeded(67)`: each call of the decorated function gets the same numbers.
    `torch.manual_seed` also seeds MPS and CUDA; their states are restored too.
    `torch.random.fork_rng` does the torch half of this.

    Args:
        seed: The seed for both generators.
    """
    py_state = random.getstate()
    cpu_state = torch.get_rng_state()
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        torch.set_rng_state(cpu_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
