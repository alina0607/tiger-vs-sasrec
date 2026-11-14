"""Semantic IDs: a few codes per item, computed from its text vector.

TIGER needs items whose text is alike to share their first codes (step 8
showed that nearest text vectors share a category 64.6% of the time). Each
tokenizer turns the 12,101 item vectors into `levels` codes, coarse to fine,
and adds one more code that tells apart items whose codes collide.

Shells until step 9: the methods below are declared but not written, so
none of these classes can be instantiated yet.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from recsys.config import RQVAEConfig, SemanticIdConfig


class SemanticIdTokenizer(ABC):
    """Gives every item a Semantic ID and finds the item behind an ID.

    Attributes:
        config: Number of levels and codes per level.
    """

    def __init__(self, config: SemanticIdConfig | None = None) -> None:
        self.config = config or SemanticIdConfig()

    @abstractmethod
    def fit(self, embeddings: torch.Tensor) -> None:
        """Learn the codebooks from the item vectors and give each item its ID.

        Args:
            embeddings: One row per item, `(n_items, 768)` from step 8.
        """

    @abstractmethod
    def encode(self, items: Sequence[int]) -> list[tuple[int, ...]]:
        """Return each item's Semantic ID: `levels` codes plus the collision code."""

    @abstractmethod
    def decode(self, semantic_id: Sequence[int]) -> int | None:
        """Return the item with this Semantic ID, or `None` if no item has it.

        TIGER can write IDs that belong to no item; the paper reports about
        0.1-1.6% of the top 10 across its datasets (Section 4.5).
        """


class ResidualKMeansTokenizer(SemanticIdTokenizer):
    """Residual k-means: cluster the vectors, then cluster what each cluster missed.

    Level 1 runs k-means on the item vectors; level 2 runs it on each vector
    minus its level-1 centre, and so on. Cheaper than the paper's RQ-VAE,
    which trains a network around the same idea; step 9 writes this one first.
    """


class RQVAETokenizer(SemanticIdTokenizer):
    """The paper's RQ-VAE: residual quantization inside a trained autoencoder.

    The encoder shrinks each 768-wide vector to 32 numbers, residual
    quantization picks one code per level, and a decoder must rebuild the
    vector from the codes (Section 3.1). Planned after step 11.

    Attributes:
        config: The layer widths and training settings of the paper.
    """

    config: RQVAEConfig

    def __init__(self, config: RQVAEConfig | None = None) -> None:
        super().__init__(config or RQVAEConfig())
