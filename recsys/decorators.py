"""Decorators used across the package."""

import functools
import hashlib
import inspect
import json
import logging
import os
import pickle
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, ParamSpec, TypeVar

import torch

from recsys.runlog import current_run_log

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)

# Where `disk_cache` stores results; read at call time, so tests can point it
# at a temporary directory.
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


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


def disk_cache(
    name: str, *, version: int = 1
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Keep each result of the decorated function in a pickle under `CACHE_DIR`.

    A call with the same arguments as an earlier one loads the stored result
    instead of running the function. The key is the SHA-256 of the function's
    full name, `version` and its arguments after `inspect.signature` binds
    them (so `f(1)` and `f(x=1)` are the same call, defaults included). The
    arguments are written as JSON with sorted keys, which gives the same text
    in every run; a `Path` becomes its absolute path (so `data/raw` and the
    same folder written in full are one key), and any other type that JSON
    cannot write raises `TypeError` instead of being guessed at.

    The cache cannot see a change inside the function: raise `version` when
    the code changes what it returns. Only load pickles this package wrote.

    Args:
        name: Subdirectory of `CACHE_DIR` for this function's results.
        version: Part of the key; raising it makes every stored result stale.

    Returns:
        A decorator that keeps the function's name, docstring and signature.
    """

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(func)
        label = f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            path = (
                CACHE_DIR / name / f"{cache_key(label, version, bound.arguments)}.pkl"
            )
            if path.exists():
                logger.debug("%s: loaded %s", label, path)
                with path.open("rb") as f:
                    cached: R = pickle.load(f)
                return cached
            result = func(*args, **kwargs)
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")
            try:
                with part.open("wb") as f:
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            except BaseException:
                part.unlink(missing_ok=True)
                raise
            os.replace(part, path)  # a crash never leaves a half-written result
            return result

        return wrapper

    return decorate


def cache_key(label: str, version: int, arguments: dict[str, Any]) -> str:
    """SHA-256 of a call, as hex: the same label, version and arguments, the same key.

    Raises:
        TypeError: If an argument is not a JSON type or a `Path`.
    """
    text = json.dumps(
        {"function": label, "version": version, "arguments": arguments},
        sort_keys=True,
        default=_path_to_str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _path_to_str(value: object) -> str:
    if isinstance(value, Path):
        return str(value.resolve())
    raise TypeError(
        f"disk_cache cannot build a key from {type(value).__name__}; "
        "pass JSON types (str, int, float, bool, None, list, tuple, dict) or Path"
    )
