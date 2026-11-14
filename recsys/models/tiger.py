"""TIGER (Rajput et al., NeurIPS 2023): generative retrieval with Semantic IDs.

A shell until step 11. The classes and their relations are final: `TIGER`
*is a* `GenerativeRecommender` and *has a* `SemanticIdTokenizer` and a
`TIGERNet`, the way `SASRec` has a `SASRecNet`. The methods that are not
written yet stay abstract, so `TIGER()` raises `TypeError` and lists them,
and the model is not registered (`scripts/run.py --model` does not offer it).
"""

import torch
from torch import nn

from recsys.base import GenerativeRecommender
from recsys.config import TIGERConfig
from recsys.semantic_ids import ResidualKMeansTokenizer, SemanticIdTokenizer


class TIGERNet(nn.Module):
    """The encoder-decoder transformer (T5 style) of the paper.

    The encoder reads a user token and the Semantic IDs of the history in
    both directions; the decoder writes the next item's ID one code at a
    time, looking only at the codes it already wrote and back at the encoder.
    With 6 heads of 64 the attention is 384 wide while the tokens are 128
    wide, which `nn.Transformer` (attention as wide as the tokens) cannot
    express; step 11 decides how to build it.
    """

    def __init__(self, config: TIGERConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, history: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Scores of the next code at each decoder position.

        Raises:
            NotImplementedError: Always, until step 11.
        """
        raise NotImplementedError("TIGERNet is written in step 11")


class TIGER(GenerativeRecommender):
    """TIGER: reads the history as Semantic IDs and writes the next item's ID.

    Still abstract: `fit`, `generate` and `recommend` come in step 11.

    Attributes:
        tokenizer: Turns items into Semantic IDs and back.
        net: The encoder-decoder.
    """

    name = "tiger"
    config_cls = TIGERConfig
    config: TIGERConfig

    def __init__(
        self,
        config: TIGERConfig | None = None,
        tokenizer: SemanticIdTokenizer | None = None,
    ) -> None:
        self.config = config or TIGERConfig()
        # The default tokenizer is still abstract; step 9 writes it and removes
        # this ignore. TIGER itself is abstract, so this line never runs yet.
        self.tokenizer = tokenizer or ResidualKMeansTokenizer()  # type: ignore[abstract]
        self.net = TIGERNet(self.config)
