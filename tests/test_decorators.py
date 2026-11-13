import logging
import threading
import time

import pytest

import recsys.decorators
from recsys.decorators import timed
from recsys.runlog import RunLog


@pytest.fixture
def clock(monkeypatch):
    """Fake clock: every call of a timed function lasts exactly 2.5 s."""
    ticks = iter(range(0, 1000, 5))
    monkeypatch.setattr(recsys.decorators, "perf_counter", lambda: next(ticks) / 2)


@timed
def add(a, b=0):
    """Add two numbers."""
    return a + b


def test_wraps_keeps_the_name_and_docstring():
    assert add.__name__ == "add"
    assert add.__qualname__ == "add"
    assert add.__doc__ == "Add two numbers."
    assert add.__module__ == __name__


def test_uses_the_monotonic_perf_counter_clock():
    # The clock fixture replaces this name, so check the real one here.
    assert recsys.decorators.perf_counter is time.perf_counter


def test_wraps_keeps_a_reference_to_the_original_function():
    assert add.__wrapped__(1, 2) == 3  # type: ignore[attr-defined]


def test_arguments_and_return_value_pass_through():
    assert add(2, b=3) == 5


def test_each_call_is_recorded_under_the_qualified_name(clock):
    class Model:
        @timed
        def fit(self):
            return "fitted"

    with RunLog() as log:
        assert Model().fit() == "fitted"
        Model().fit()
    expected = "test_each_call_is_recorded_under_the_qualified_name.<locals>.Model.fit"
    assert log.timings == {expected: [2.5, 2.5]}


def test_nothing_is_recorded_outside_a_run(clock):
    assert add(1) == 1
    with RunLog() as log:
        pass
    assert log.timings == {}


def test_outside_a_run_the_time_is_logged_at_debug_level(clock, caplog):
    with caplog.at_level(logging.DEBUG, logger="recsys.decorators"):
        add(1)
    assert caplog.messages == ["add took 2.500 s"]


def test_a_call_that_raises_is_not_recorded(clock):
    @timed
    def fail():
        raise ValueError("boom")

    with RunLog() as log, pytest.raises(ValueError, match="boom"):
        fail()
    assert log.timings == {}


def test_calls_go_to_the_innermost_active_run(clock):
    with RunLog() as outer:
        add(1)
        with RunLog() as inner:
            add(1)
        add(1)
    assert outer.timings == {"add": [2.5, 2.5]}
    assert inner.timings == {"add": [2.5]}


def test_parallel_threads_record_into_their_own_logs():
    # Real clock: the fake one is not thread-safe. Same two barriers as in
    # test_runlog.py.
    both_active = threading.Barrier(2)
    both_done = threading.Barrier(2)
    logs: dict[str, RunLog] = {}

    def run(name, calls):
        with RunLog() as log:
            both_active.wait()
            for _ in range(calls):
                add(1)
            both_done.wait()
            logs[name] = log

    threads = [
        threading.Thread(target=run, args=("a", 3)),
        threading.Thread(target=run, args=("b", 5)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(logs["a"].timings["add"]) == 3
    assert len(logs["b"].timings["add"]) == 5
