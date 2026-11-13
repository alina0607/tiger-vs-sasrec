"""Per-run record: which model was built, how long each step took, the scores.

`with RunLog() as log:` makes `log` the active run log for the current thread
or asyncio task. It is kept in a `ContextVar`, so parallel threads never share
one.
"""

import json
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from recsys.config import ModelConfig


@dataclass
class RunLog:
    """What one run built, how long each step took, and what it scored.

    Attributes:
        models: Name and configuration of each model built, in build order.
        timings: Label (usually a qualified function name) -> seconds of each
            call, in call order.
        results: Part (`"valid"` or `"test"`) -> metric scores.
    """

    models: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, list[float]] = field(default_factory=dict)
    results: dict[str, dict[str, float]] = field(default_factory=dict)
    # Set while active; restores the previously active log on exit.
    _token: Token["RunLog | None"] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def add_model(self, name: str, config: ModelConfig) -> None:
        """Record a model as it is built.

        Args:
            name: The model's registered name.
            config: Its configuration.
        """
        self.models.append({"name": name, "config": asdict(config)})

    def add_timing(self, label: str, seconds: float) -> None:
        """Record one call.

        Args:
            label: What was timed, for example `SASRec.fit`.
            seconds: How long the call took.
        """
        self.timings.setdefault(label, []).append(seconds)

    def add_result(self, part: str, scores: dict[str, float]) -> None:
        """Record the scores of one evaluation.

        Args:
            part: Which cases were scored, for example `"test"`.
            scores: Metric name -> mean score.
        """
        self.results[part] = dict(scores)

    def to_dict(self) -> dict[str, Any]:
        """Return the log as JSON-ready data (copies, not the live lists)."""
        return {
            "models": [dict(m, config=dict(m["config"])) for m in self.models],
            "timings": {label: list(s) for label, s in self.timings.items()},
            "results": {part: dict(s) for part, s in self.results.items()},
        }

    def save(self, path: Path) -> None:
        """Write the log as JSON.

        Args:
            path: The file to write. Missing parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def __enter__(self) -> Self:
        if self._token is not None:
            raise RuntimeError("this RunLog is already active")
        self._token = _active.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._token is None:
            raise RuntimeError("this RunLog is not active")
        # reset, not set(None), so an enclosing run becomes active again.
        _active.reset(self._token)
        self._token = None


_active: ContextVar[RunLog | None] = ContextVar("recsys_active_run_log", default=None)


def current_run_log() -> RunLog | None:
    """Return the active `RunLog` of this thread or task, or `None`."""
    return _active.get()
