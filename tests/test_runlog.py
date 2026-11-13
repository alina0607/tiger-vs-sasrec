import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from recsys.runlog import RunLog, current_run_log


def test_add_timing_keeps_one_entry_per_call_in_order():
    log = RunLog()
    log.add_timing("fit", 1.5)
    log.add_timing("fit", 0.5)
    log.add_timing("evaluate", 2.0)
    assert log.timings == {"fit": [1.5, 0.5], "evaluate": [2.0]}


def test_save_writes_json_that_reads_back(tmp_path):
    log = RunLog()
    log.add_timing("fit", 1.25)
    path = tmp_path / "runs" / "2026-11-12" / "log.json"  # parents do not exist yet
    log.save(path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"timings": {"fit": [1.25]}}


def test_no_log_is_active_outside_a_with_block():
    assert current_run_log() is None


def test_with_block_makes_the_log_active_and_clears_it_on_exit():
    with RunLog() as log:
        assert current_run_log() is log
    assert current_run_log() is None


def test_log_is_cleared_even_when_the_block_raises():
    with pytest.raises(ValueError, match="boom"), RunLog():
        raise ValueError("boom")
    assert current_run_log() is None


def test_nested_runs_restore_the_outer_log():
    with RunLog() as outer:
        with RunLog() as inner:
            assert current_run_log() is inner
        assert current_run_log() is outer


def test_the_same_log_cannot_be_entered_twice():
    log = RunLog()
    with log, pytest.raises(RuntimeError, match="already active"), log:
        pass


def test_exit_without_enter_is_an_error():
    with pytest.raises(RuntimeError, match="not active"):
        RunLog().__exit__(None, None, None)


def test_each_thread_sees_only_its_own_log():
    # The second barrier keeps both logs active until both threads have read.
    # Without it, a global passes by luck: the first thread to exit restores
    # the other thread's log.
    both_active = threading.Barrier(2)
    both_read = threading.Barrier(2)
    seen: dict[str, bool] = {}

    def run(name):
        with RunLog() as log:
            both_active.wait()
            seen[name] = current_run_log() is log
            both_read.wait()

    threads = [threading.Thread(target=run, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {"a": True, "b": True}


def test_a_thread_pool_worker_does_not_inherit_the_active_log():
    with RunLog(), ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(current_run_log).result() is None
