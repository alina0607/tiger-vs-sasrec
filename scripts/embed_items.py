"""Turn every Beauty item's text into a Sentence-T5 vector and cache the result.

Run `uv sync --extra embed`, then `uv run python scripts/embed_items.py`. The
first run downloads the model and encodes the 12,101 items; later runs load
both results from `cache/`.
"""

import argparse
import logging
from pathlib import Path

import torch

from recsys.data import load_beauty
from recsys.embed import MODEL, embed_texts
from recsys.items import load_item_texts
from recsys.runlog import RunLog

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Embed the items, print the shape, timings and one item's neighbours.

    Args:
        argv: Command-line arguments; `None` reads `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--item", type=int, default=0, help="item to show neighbours of"
    )
    args = parser.parse_args(argv)
    # INFO for this package only: the download libraries log every HTTP request.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("recsys").setLevel(logging.INFO)

    with RunLog() as log:
        split = load_beauty(args.data_dir)
        texts = load_item_texts(args.data_dir, list(split.items))
        vectors = embed_texts(texts, args.model)

    print(f"vectors  {tuple(vectors.shape)} {vectors.dtype}")
    for label, seconds in log.timings.items():
        print(f"{label:16} {seconds[0]:.2f} s")
    print(f"item {args.item}: {texts[args.item]}")
    unit = torch.nn.functional.normalize(vectors, dim=1)
    similarity = unit @ unit[args.item]
    similarity[args.item] = -1.0
    best = similarity.topk(5)
    for score, j in zip(best.values.tolist(), best.indices.tolist(), strict=True):
        print(f"  {score:.3f}  {j:5}  {texts[j][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
