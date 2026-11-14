import os
import pickle
from pathlib import Path

import pytest

import recsys.decorators
from recsys.decorators import cache_key, disk_cache


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    """Every test gets its own empty cache directory."""
    monkeypatch.setattr(recsys.decorators, "CACHE_DIR", tmp_path)
    return tmp_path


def counted(version=1):
    """A cached function and a list that records each real call."""
    calls = []

    @disk_cache("square", version=version)
    def square(x, offset=0):
        """Square x and add offset."""
        calls.append((x, offset))
        return x * x + offset

    return square, calls


def test_second_call_with_the_same_arguments_is_loaded_not_run(cache_dir):
    square, calls = counted()
    assert square(3) == 9
    assert square(3) == 9
    assert calls == [(3, 0)]
    assert len(list((cache_dir / "square").glob("*.pkl"))) == 1


def test_different_arguments_run_again():
    square, calls = counted()
    square(3)
    square(4)
    square(3, offset=1)
    assert calls == [(3, 0), (4, 0), (3, 1)]


def test_positional_keyword_and_default_spellings_are_one_call():
    square, calls = counted()
    square(3)
    square(x=3)
    square(3, 0)
    square(3, offset=0)
    assert calls == [(3, 0)]


def test_result_survives_a_new_process():
    # A new definition stands for a later run: nothing is shared but the files.
    first, _ = counted()
    first(5)
    second, calls = counted()
    assert second(5) == 25
    assert calls == []


def test_raising_the_version_makes_old_results_stale():
    old, _ = counted(version=1)
    old(3)
    new, calls = counted(version=2)
    new(3)
    assert calls == [(3, 0)]


def test_key_does_not_depend_on_dict_order():
    a = cache_key("f", 1, {"x": 1, "y": {"b": 2, "a": 1}})
    b = cache_key("f", 1, {"y": {"a": 1, "b": 2}, "x": 1})
    assert a == b
    assert len(a) == 64


def test_key_includes_the_function_name():
    assert cache_key("f", 1, {"x": 1}) != cache_key("g", 1, {"x": 1})


def test_a_relative_and_an_absolute_path_to_one_folder_are_one_key(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    relative = cache_key("f", 1, {"p": Path("data/raw")})
    assert relative == cache_key("f", 1, {"p": tmp_path / "data" / "raw"})
    assert relative == cache_key("f", 1, {"p": str(tmp_path.resolve() / "data/raw")})
    assert relative != cache_key("f", 1, {"p": Path("data/other")})


def test_an_argument_json_cannot_write_is_an_error():
    @disk_cache("sets")
    def size(items):
        return len(items)

    with pytest.raises(TypeError, match="set"):
        size({1, 2})


def test_a_failed_call_stores_nothing(cache_dir):
    @disk_cache("fails")
    def fail(x):
        raise ValueError(x)

    with pytest.raises(ValueError, match="3"):
        fail(3)
    assert not (cache_dir / "fails").exists()


def test_a_result_pickle_cannot_store_leaves_no_file(cache_dir):
    @disk_cache("lambda")
    def make(x):
        return lambda: x  # lambdas cannot be pickled

    with pytest.raises((pickle.PicklingError, AttributeError)):
        make(1)
    assert os.listdir(cache_dir / "lambda") == []


def test_wraps_keeps_the_name_docstring_and_original():
    square, calls = counted()
    assert square.__name__ == "square"
    assert square.__doc__ == "Square x and add offset."
    assert square.__wrapped__(2) == 4  # type: ignore[attr-defined]
    assert calls == [(2, 0)]
