import unittest
from unittest import mock

import torch

from recsys.base import Recommender, ScoringRecommender
from recsys.config import ModelConfig
from recsys.registry import RecommenderMeta

# A model class registers itself when defined; keep this test double out of
# the real registry.
with mock.patch.dict(RecommenderMeta._registry):

    class FixedScores(ScoringRecommender):
        """Returns a fixed score matrix, so the ranking logic can be tested alone."""

        name = "fixed-scores"
        config_cls = ModelConfig

        def __init__(self, config, scores):
            self.config = config
            self.scores = scores

        def fit(self, split):
            pass

        def score_all(self, histories):
            return self.scores[: len(histories)]


class TestAbstractClasses(unittest.TestCase):
    def test_base_classes_cannot_be_instantiated(self):
        for cls in (Recommender, ScoringRecommender):
            with (
                self.subTest(cls=cls.__name__),
                self.assertRaisesRegex(TypeError, "abstract"),
            ):
                cls()

    def test_scoring_recommender_only_asks_for_fit_and_score_all(self):
        self.assertEqual(ScoringRecommender.__abstractmethods__, {"fit", "score_all"})


class TestScoringRecommender(unittest.TestCase):
    def setUp(self):
        # Two users, five items. Item 0 is user 0's best, item 4 is user 1's.
        self.scores = torch.tensor(
            [
                [0.9, 0.1, 0.5, 0.7, 0.3],
                [0.2, 0.6, 0.4, 0.1, 0.8],
            ]
        )
        self.model = FixedScores(ModelConfig(), self.scores)

    def test_returns_top_k_best_first(self):
        self.assertEqual(self.model.recommend([[], []], k=3), [[0, 3, 2], [4, 1, 2]])

    def test_items_in_the_history_are_not_recommended(self):
        top = self.model.recommend([[0, 3], [4]], k=2)
        self.assertEqual(top, [[2, 4], [1, 2]])

    def test_model_scores_are_not_modified(self):
        before = self.scores.clone()
        self.model.recommend([[0, 3], [4]], k=2)
        self.assertTrue(torch.equal(self.scores, before))

    def test_integer_scores_are_accepted(self):
        model = FixedScores(ModelConfig(), torch.tensor([[1, 5, 3]]))
        self.assertEqual(model.recommend([[1]], k=2), [[2, 0]])

    def test_k_out_of_range_is_rejected(self):
        for k in (0, 6):
            with (
                self.subTest(k=k),
                self.assertRaisesRegex(ValueError, "k must be between 1 and 5"),
            ):
                self.model.recommend([[], []], k=k)

    def test_too_few_unseen_items_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fewer than k=3"):
            self.model.recommend([[0, 1, 2], []], k=3)

    def test_wrong_score_shape_is_rejected(self):
        with (
            mock.patch.object(self.model, "score_all", return_value=torch.zeros(5)),
            self.assertRaisesRegex(ValueError, r"shape \(5,\)"),
        ):
            self.model.recommend([[], []], k=1)


if __name__ == "__main__":
    unittest.main()
