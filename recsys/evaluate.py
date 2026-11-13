"""Full-ranking evaluation, the protocol of TIGER Table 1.

Every case ranks its correct item against all items (not a sample of 100), and
items already in the history are excluded by the model's `recommend`.
"""

from collections.abc import Sequence
from typing import Literal

from recsys.base import Recommender
from recsys.data import Split
from recsys.decorators import timed
from recsys.metrics import registered


@timed
def evaluate(
    model: Recommender,
    split: Split,
    *,
    part: Literal["valid", "test"] = "test",
    ks: Sequence[int] = (5, 10),
    batch_size: int = 1024,
) -> dict[str, float]:
    """Average every registered metric at every cutoff in `ks`.

    Args:
        model: A fitted recommender.
        split: The data; its `valid()` or `test()` cases are scored.
        part: Which cases to score.
        ks: Cutoffs; the model is asked for its top `max(ks)` items.
        batch_size: Users per `recommend` call. Bounds memory: a scoring model
            builds a `batch_size x n_items` score matrix.

    Returns:
        Mean score per `"name@k"`, for example `{"ndcg@10": 0.03, ...}`.

    Raises:
        ValueError: If `part` is unknown or the model returns too few items.
    """
    ranks = answer_ranks(model, split, part=part, top=max(ks), batch_size=batch_size)
    return {
        f"{name}@{k}": sum(score(rank, k) for rank in ranks) / len(ranks)
        for name, score in sorted(registered().items())
        for k in ks
    }


def answer_ranks(
    model: Recommender,
    split: Split,
    *,
    part: Literal["valid", "test"],
    top: int,
    batch_size: int,
) -> list[int | None]:
    """Rank of each case's correct item in the model's top `top` (1 = first).

    `None` means the item is not in the list.
    """
    if part == "test":
        histories, answers = split.test()
    elif part == "valid":
        histories, answers = split.valid()
    else:
        raise ValueError(f"part must be 'valid' or 'test', got {part!r}")
    ranks: list[int | None] = []
    for start in range(0, len(answers), batch_size):
        stop = start + batch_size
        lists = model.recommend(histories[start:stop], top)
        for items, answer in zip(lists, answers[start:stop], strict=True):
            if len(items) < top:
                raise ValueError(
                    f"recommend returned {len(items)} items, expected {top}"
                )
            head = items[:top]
            ranks.append(head.index(answer) + 1 if answer in head else None)
    return ranks
