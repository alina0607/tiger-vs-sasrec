import math
from dataclasses import dataclass

import pytest
import torch

import recsys.metrics
from recsys.base import ScoringRecommender
from recsys.config import ModelConfig
from recsys.data import Split
from recsys.evaluate import evaluate
from recsys.metrics import metric, registered
from recsys.runlog import RunLog

N_ITEMS = 15

# Three users. The fake model prefers lower item ids, so after the history is
# masked its list is the unseen ids in ascending order.
#   test:  A (5, 6) -> 0 is 1st;  B (0, 1) -> 4 is 3rd;  C (0, 1, 2) -> 14 is 12th
#   valid: A (5,) -> 6 is 6th;    B (0,) -> 1 is 1st;    C (0, 1) -> 2 is 1st
SPLIT = Split(
    sequences=((5, 6, 0), (0, 1, 4), (0, 1, 2, 14)),
    users=("A", "B", "C"),
    items=tuple(str(i) for i in range(N_ITEMS)),
)


@dataclass(frozen=True)
class FakeConfig(ModelConfig):
    pass


def make_model_class():
    class LowestIdFirst(ScoringRecommender):
        name = "lowest-id-first"
        config_cls = FakeConfig

        def __init__(self):
            self.config = FakeConfig()
            self.calls = 0

        def fit(self, split):
            pass

        def score_all(self, histories):
            self.calls += 1
            return -torch.arange(N_ITEMS, dtype=torch.float32).repeat(len(histories), 1)

    return LowestIdFirst


def make_model():
    return make_model_class()()


def test_test_part_matches_hand_computed_ranks():
    scores = evaluate(make_model(), SPLIT)
    assert scores == pytest.approx(
        {"ndcg@5": 0.5, "ndcg@10": 0.5, "recall@5": 2 / 3, "recall@10": 2 / 3}
    )


def test_valid_part_uses_the_second_to_last_item():
    scores = evaluate(make_model(), SPLIT, part="valid")
    sixth = 1 / math.log2(7)
    assert scores == pytest.approx(
        {
            "ndcg@5": 2 / 3,
            "ndcg@10": (sixth + 2) / 3,
            "recall@5": 2 / 3,
            "recall@10": 1.0,
        }
    )


def test_batching_does_not_change_the_result():
    cls = make_model_class()
    model = cls()
    one_by_one = evaluate(model, SPLIT, batch_size=1)
    assert model.calls == 3
    assert one_by_one == evaluate(cls(), SPLIT)


def test_a_new_metric_is_picked_up_without_changing_evaluate(monkeypatch):
    monkeypatch.setattr(recsys.metrics, "_registry", dict(registered()))

    @metric("first")
    def first(rank, _k):
        return 1.0 if rank == 1 else 0.0

    scores = evaluate(make_model(), SPLIT, ks=(10,))
    assert set(scores) == {"first@10", "ndcg@10", "recall@10"}
    assert scores["first@10"] == pytest.approx(1 / 3)


def test_unknown_part_is_rejected():
    with pytest.raises(ValueError, match="part must be"):
        evaluate(make_model(), SPLIT, part="train")  # type: ignore[arg-type]


def test_a_model_returning_too_few_items_is_rejected():
    model = make_model()
    model.recommend = lambda histories, k: [[0]] * len(histories)  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="returned 1 items, expected 10"):
        evaluate(model, SPLIT)


def test_evaluate_is_timed():
    with RunLog() as log:
        evaluate(make_model(), SPLIT)
    assert list(log.timings) == ["evaluate"]
