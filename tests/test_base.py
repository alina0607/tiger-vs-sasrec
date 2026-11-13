import pytest
import torch

from recsys.base import Recommender, ScoringRecommender
from recsys.config import ModelConfig

# Two users, five items. Item 0 is user 0's best, item 4 is user 1's.
SCORES = torch.tensor(
    [
        [0.9, 0.1, 0.5, 0.7, 0.3],
        [0.2, 0.6, 0.4, 0.1, 0.8],
    ]
)


@pytest.fixture
def make_model():
    """Return a factory for a model that serves a fixed score matrix.

    With the scores fixed, the ranking logic of ScoringRecommender is tested
    on its own.
    """

    class FixedScores(ScoringRecommender):
        name = "fixed-scores"
        config_cls = ModelConfig

        def __init__(self, config, scores):
            self.config = config
            self.scores = scores

        def fit(self, split):
            pass

        def score_all(self, histories):
            return self.scores[: len(histories)]

    def make(scores=None):
        # Each model gets its own copy: a bug that writes into the scores must
        # not leak into the shared SCORES and hide itself from later tests.
        return FixedScores(ModelConfig(), SCORES.clone() if scores is None else scores)

    return make


@pytest.mark.parametrize("cls", [Recommender, ScoringRecommender])
def test_base_classes_cannot_be_instantiated(cls):
    with pytest.raises(TypeError, match="abstract"):
        cls()


def test_scoring_recommender_only_asks_for_fit_and_score_all():
    assert ScoringRecommender.__abstractmethods__ == {"fit", "score_all"}


def test_returns_top_k_best_first(make_model):
    assert make_model().recommend([[], []], k=3) == [[0, 3, 2], [4, 1, 2]]


def test_items_in_the_history_are_not_recommended(make_model):
    assert make_model().recommend([[0, 3], [4]], k=2) == [[2, 4], [1, 2]]


def test_model_scores_are_not_modified(make_model):
    scores = SCORES.clone()
    make_model(scores).recommend([[0, 3], [4]], k=2)
    assert torch.isfinite(scores).all()
    assert torch.equal(scores, SCORES)


def test_integer_scores_are_accepted(make_model):
    model = make_model(torch.tensor([[1, 5, 3]]))
    assert model.recommend([[1]], k=2) == [[2, 0]]


@pytest.mark.parametrize("k", [0, 6])
def test_k_out_of_range_is_rejected(make_model, k):
    with pytest.raises(ValueError, match="k must be between 1 and 5"):
        make_model().recommend([[], []], k=k)


def test_too_few_unseen_items_is_rejected(make_model):
    with pytest.raises(ValueError, match="fewer than k=3"):
        make_model().recommend([[0, 1, 2], []], k=3)


def test_wrong_score_shape_is_rejected(make_model, monkeypatch):
    model = make_model()
    monkeypatch.setattr(model, "score_all", lambda histories: torch.zeros(5))
    with pytest.raises(ValueError, match=r"shape \(5,\)"):
        model.recommend([[], []], k=1)
