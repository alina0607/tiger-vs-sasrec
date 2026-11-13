"""Abstract base classes for the recommenders.

The hierarchy branches where the TIGER paper separates two ways of retrieving
items. `ScoringRecommender` scores every item and returns the top k
(popularity, SASRec). A generative branch that decodes Semantic IDs is added
together with TIGER.

Item ids are column indices: column `j` of a score matrix is item `j`.
"""

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import torch

from recsys.config import ModelConfig
from recsys.data import Split
from recsys.registry import RecommenderMeta

# One user's interaction history: item ids, oldest first.
History = Sequence[int]


class Recommender(metaclass=RecommenderMeta):
    """Base class of every model.

    A concrete subclass must set the class attributes `name` and
    `config_cls`, and set `self.config` in `__init__`; the metaclass
    checks all three.
    """

    name: ClassVar[str]
    config_cls: ClassVar[type[ModelConfig]]
    config: ModelConfig

    @abstractmethod
    def fit(self, split: Split) -> None:
        """Train on the training sequences of `split`.

        Args:
            split: The dataset split to train on.
        """

    @abstractmethod
    def recommend(self, histories: Sequence[History], k: int) -> list[list[int]]:
        """Return the top `k` item ids for each history, best first.

        Args:
            histories: One item-id sequence per user, oldest first.
            k: How many items to return per user.

        Returns:
            One list of `k` item ids per history.
        """


class ScoringRecommender(Recommender):
    """Recommender that scores every item and returns the highest-scoring ones.

    Subclasses implement `score_all`; ranking and the removal of items the
    user already has are shared here.
    """

    @abstractmethod
    def score_all(self, histories: Sequence[History]) -> torch.Tensor:
        """Score every item for every history.

        Returns:
            A `(len(histories), n_items)` tensor; higher is better.
        """

    def recommend(self, histories: Sequence[History], k: int) -> list[list[int]]:
        """Return the `k` highest-scoring items not already in each history.

        Raises:
            ValueError: If `score_all` returns the wrong shape, `k` is out of
                range, or a history leaves fewer than `k` unseen items.
        """
        with torch.no_grad():
            scores = self.score_all(histories)
            if scores.dim() != 2 or scores.shape[0] != len(histories):
                raise ValueError(
                    f"score_all returned shape {tuple(scores.shape)}; "
                    f"expected ({len(histories)}, n_items)"
                )
            n_items = scores.shape[1]
            if not 1 <= k <= n_items:
                raise ValueError(f"k must be between 1 and {n_items}, got {k}")
            # Work on a float copy: the masking below must not change a tensor
            # the model may still hold, and -inf needs a floating-point dtype.
            scores = scores.to(torch.float32, copy=True)
            rows = [row for row, history in enumerate(histories) for _ in history]
            cols = [item for history in histories for item in history]
            scores[rows, cols] = float("-inf")
            if len(histories) and int(torch.isfinite(scores).sum(dim=1).min()) < k:
                raise ValueError(f"a history leaves fewer than k={k} unseen items")
            return scores.topk(k, dim=1).indices.tolist()
