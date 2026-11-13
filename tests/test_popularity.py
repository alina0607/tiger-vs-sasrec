from pathlib import Path

import pytest

from recsys.config import PopularityConfig
from recsys.data import REVIEWS, Split, load_beauty
from recsys.evaluate import evaluate
from recsys.models.popularity import PopularityRecommender
from recsys.runlog import RunLog

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Training parts (all but the last two items): (2, 1), (2, 0), (2, 3, 0).
# Counts: 2 -> 3, 0 -> 2, 1 -> 1, 3 -> 1, 4 -> 0, 5 -> 0. Item 5 appears four
# times, but only in the held-out positions, so it must not count.
SPLIT = Split(
    sequences=(
        (2, 1, 5, 5),
        (2, 0, 5, 5),
        (2, 3, 0, 1, 4),
    ),
    users=("a", "b", "c"),
    items=tuple("abcdef"),
)


@pytest.fixture
def model():
    m = PopularityRecommender()
    m.fit(SPLIT)
    return m


def test_ranks_by_training_count_and_breaks_ties_by_lower_id(model):
    # 1 and 3 tie at 1, 4 and 5 tie at 0: the lower id goes first.
    assert model.recommend([[]], 6) == [[2, 0, 1, 3, 4, 5]]


def test_held_out_items_are_not_counted(model):
    scores = model.score_all([[]])[0]
    assert scores[5] < scores[3]  # 5 is held out four times, 3 trained once


def test_recommend_skips_the_history(model):
    assert model.recommend([[2, 0]], 3) == [[1, 3, 4]]


def test_every_user_gets_the_same_scores(model):
    scores = model.score_all([[0], [1, 2], []])
    assert scores.shape == (3, 6)
    assert (scores == scores[0]).all()


def test_scoring_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="call fit"):
        PopularityRecommender().score_all([[]])


def test_default_config_and_registered_name():
    assert PopularityRecommender().config == PopularityConfig()
    assert PopularityRecommender.name == "popularity"


def test_fit_is_timed_and_the_model_is_logged():
    with RunLog() as log:
        PopularityRecommender().fit(SPLIT)
    assert log.models == [{"name": "popularity", "config": {"seed": 42}}]
    assert list(log.timings) == ["PopularityRecommender.fit"]


@pytest.mark.skipif(
    not (DATA_DIR / REVIEWS.name).exists(), reason="Beauty data not downloaded"
)
def test_beauty_scores():
    split = load_beauty(DATA_DIR)
    model = PopularityRecommender()
    model.fit(split)
    scores = evaluate(model, split)
    assert scores["recall@10"] == pytest.approx(0.011537, abs=1e-6)
    assert scores["ndcg@10"] == pytest.approx(0.005456, abs=1e-6)
