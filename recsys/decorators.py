"""Decorators used across the package."""

import functools
import logging
from collections.abc import Callable
from time import perf_counter
from typing import ParamSpec, TypeVar

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
