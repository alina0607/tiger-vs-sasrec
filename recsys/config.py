"""Configuration objects for the recommenders.

Every configuration is a frozen dataclass: once built it cannot be changed, so
the values recorded for a run are the values the model actually used.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Fields shared by every model configuration.

    Each model declares its own subclass and names it in `config_cls`.

    Attributes:
        seed: Seed for every random number generator in a run.
    """

    seed: int = 67


@dataclass(frozen=True)
class PopularityConfig(ModelConfig):
    """Configuration of `PopularityRecommender`; it has nothing to tune."""


@dataclass(frozen=True)
class SASRecConfig(ModelConfig):
    """Configuration of `SASRec`; defaults are the paper's settings for Beauty.

    Attributes:
        hidden: Width `d` of the embeddings and every layer.
        blocks: Number of self-attention blocks.
        heads: Attention heads per block.
        max_len: Longest history the model reads; older items are dropped.
        dropout: Dropout rate (the paper uses 0.5 on the sparse Amazon data).
        lr: Adam learning rate.
        batch_size: Users per training step.
        max_epochs: Upper bound on training epochs.
        eval_every: Epochs between two validation scores.
        patience: Validation checks without improvement before stopping.
    """

    hidden: int = 50
    blocks: int = 2
    heads: int = 1
    max_len: int = 50
    dropout: float = 0.5
    lr: float = 0.001
    batch_size: int = 128
    max_epochs: int = 200
    eval_every: int = 5
    patience: int = 10
