"""Configuration objects for the recommenders.

Every configuration is a frozen dataclass: once built it cannot be changed, so
the values recorded for a run are the values the model actually used.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Fields shared by every model configuration.

    Each model declares its own subclass and names it in ``config_cls``.
    """

    seed: int = 42
