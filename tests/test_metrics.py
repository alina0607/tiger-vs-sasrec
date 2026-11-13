import math

import pytest

import recsys.metrics
from recsys.metrics import metric, ndcg, recall, registered


@pytest.mark.parametrize(
    ("rank", "k", "expected"),
    [
        (1, 10, 1.0),
        (10, 10, 1.0),
        (11, 10, 0.0),
        (None, 10, 0.0),
        (5, 5, 1.0),
        (6, 5, 0.0),
    ],
)
def test_recall(rank, k, expected):
    assert recall(rank, k) == expected


@pytest.mark.parametrize(
    ("rank", "k", "expected"),
    [
        (1, 10, 1.0),
        (3, 10, 0.5),  # 1 / log2(4)
        (10, 10, 1 / math.log2(11)),
        (11, 10, 0.0),
        (None, 10, 0.0),
        (6, 5, 0.0),
    ],
)
def test_ndcg(rank, k, expected):
    assert ndcg(rank, k) == pytest.approx(expected)


def test_five_users_from_the_notes():
    ranks = [1, 3, 10, 25, None]
    assert sum(recall(r, 10) for r in ranks) / 5 == pytest.approx(0.6)
    assert sum(ndcg(r, 10) for r in ranks) / 5 == pytest.approx(0.3578, abs=1e-4)


def test_decorator_registers_and_returns_the_function_unchanged():
    assert registered()["recall"] is recall
    assert registered()["ndcg"] is ndcg
    assert recall.__name__ == "recall"


def test_duplicate_name_fails_at_definition_and_keeps_the_original(monkeypatch):
    monkeypatch.setattr(recsys.metrics, "_registry", dict(registered()))
    with pytest.raises(ValueError, match="'recall' is already registered"):

        @metric("recall")
        def other(_rank, _k):
            return 0.0

    assert registered()["recall"] is recall


def test_registry_view_is_read_only():
    with pytest.raises(TypeError):
        registered()["x"] = recall  # type: ignore[index]
