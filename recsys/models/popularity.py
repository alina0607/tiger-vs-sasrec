"""The popularity baseline: recommend the items bought most often."""

from collections import Counter
from collections.abc import Sequence

import torch

from recsys.base import History, ScoringRecommender
from recsys.config import PopularityConfig
from recsys.data import Split
from recsys.decorators import timed


class PopularityRecommender(ScoringRecommender):
    """Gives every user the same ranking: items by how often they were bought.

    Counts come from the training sequences only, the data SASRec trains on.
    Ties go to the lower item id, so the ranking does not depend on how
    `torch.topk` orders equal scores.
    """

    name = "popularity"
    config_cls = PopularityConfig

    def __init__(self, config: PopularityConfig | None = None) -> None:
        self.config = config or PopularityConfig()
        self._scores: torch.Tensor | None = None

    @timed
    def fit(self, split: Split) -> None:
        """Count how often each item appears in `split.train()`."""
        counts = Counter(item for seq in split.train() for item in seq)
        order = sorted(range(split.n_items), key=lambda item: (-counts[item], item))
        # Score = how many items rank below it, so every score is distinct.
        scores = torch.empty(split.n_items)
        scores[order] = torch.arange(split.n_items - 1, -1, -1, dtype=torch.float32)
        self._scores = scores

    def score_all(self, histories: Sequence[History]) -> torch.Tensor:
        """Return the same popularity scores for every history.

        Raises:
            RuntimeError: If `fit` has not been called.
        """
        if self._scores is None:
            raise RuntimeError("call fit() before scoring")
        return self._scores.expand(len(histories), -1)
