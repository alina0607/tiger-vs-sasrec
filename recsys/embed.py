"""Item text to Sentence-T5 vectors, the input of TIGER's Semantic IDs.

Needs the `embed` extra (`uv sync --extra embed`), which installs
sentence-transformers; the rest of the package does not import it.
"""

import logging
from typing import TYPE_CHECKING

import torch

from recsys.decorators import disk_cache, timed

if TYPE_CHECKING:  # only for the annotation; importing it is slow and optional
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL = "sentence-transformers/sentence-t5-base"
# The model files at this commit of its Hugging Face repository, so a later
# upload cannot change the vectors (or the cache key) behind our back.
REVISION = "fc5d4628481afbbaaacd7af6bb07cf9d3865f781"
_BATCH_SIZE = 64


@timed
@disk_cache("item_embeddings")
def embed_texts(
    texts: list[str], model_name: str = MODEL, revision: str | None = REVISION
) -> torch.Tensor:
    """Encode each text with a pretrained sentence encoder.

    The encoder is only used, never trained. It runs on the GPU when there is
    one (MPS or CUDA), which can change the last digits of the result; the
    cache stores whichever came first.

    Args:
        texts: One sentence per item (`recsys.items.load_item_texts`).
        model_name: A Sentence-Transformers model on the Hugging Face Hub.
        revision: Commit of the model repository; `None` takes the latest.

    Returns:
        A float32 CPU tensor of shape `(len(texts), dimension)`; 768 for
        Sentence-T5.
    """
    model = _load_model(model_name, revision)
    logger.info("encoding %d texts with %s on %s", len(texts), model_name, model.device)
    vectors = model.encode(
        texts,
        batch_size=_BATCH_SIZE,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    return torch.as_tensor(vectors).float().cpu()


def _load_model(model_name: str, revision: str | None) -> "SentenceTransformer":
    """Load (downloading on first use) a Sentence-Transformers model."""
    from sentence_transformers import SentenceTransformer  # the `embed` extra

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return SentenceTransformer(model_name, device=device, revision=revision)
