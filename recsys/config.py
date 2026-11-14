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


@dataclass(frozen=True)
class TIGERConfig(ModelConfig):
    """Configuration of `TIGER`; defaults are the settings the paper reports (p6).

    TIGER is not written yet (step 11). These are the values the paper
    reports, kept here so the shell already says what will be built.

    Attributes:
        encoder_layers: Transformer layers that read the history.
        decoder_layers: Transformer layers that write the next Semantic ID.
        heads: Attention heads per layer.
        head_dim: Width of each head; 6 x 64 = 384 is wider than `d_model`,
            as T5 allows.
        d_model: Width of the token embeddings.
        d_ff: Width of the feed-forward layer.
        dropout: Dropout rate.
        user_tokens: Users are hashed into this many tokens.
        max_history: Most recent items the encoder reads.
        steps: Training steps on Beauty.
        batch_size: Sequences per training step.
        lr: Learning rate for the first `warmup_steps`, then inverse
            square-root decay.
        warmup_steps: Steps at the constant learning rate.
    """

    encoder_layers: int = 4
    decoder_layers: int = 4
    heads: int = 6
    head_dim: int = 64
    d_model: int = 128
    d_ff: int = 1024
    dropout: float = 0.1
    user_tokens: int = 2000
    max_history: int = 20
    steps: int = 200_000
    batch_size: int = 256
    lr: float = 0.01
    warmup_steps: int = 10_000


@dataclass(frozen=True)
class SemanticIdConfig:
    """How long a Semantic ID is: the paper's main setting is 3 levels of 256 codes.

    Not a `ModelConfig`: a tokenizer is part of TIGER, not a model of its own.

    Attributes:
        levels: Codes computed from the item's vector, coarse to fine.
        codebook_size: Possible values of each code.
        seed: Seed for the clustering.
    """

    levels: int = 3
    codebook_size: int = 256
    seed: int = 67


@dataclass(frozen=True)
class RQVAEConfig(SemanticIdConfig):
    """The paper's RQ-VAE settings: an autoencoder whose middle is quantized.

    Attributes:
        hidden: Encoder layer widths from the 768-wide input down.
        latent: Width of the vector that is quantized.
        beta: Weight of the commitment loss.
        epochs: Training epochs.
        lr: Adagrad learning rate.
        batch_size: Items per training step.
    """

    hidden: tuple[int, ...] = (512, 256, 128)
    latent: int = 32
    beta: float = 0.25
    epochs: int = 20_000
    lr: float = 0.4
    batch_size: int = 1024
