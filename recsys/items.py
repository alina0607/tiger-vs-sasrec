"""One sentence of text per item, from the Beauty metadata file.

TIGER feeds each item's title, price, brand and categories to Sentence-T5
(Section 4) but does not give the exact sentence; this module writes the
fields in that order as `Name: value`, separated by `; `, and skips the ones
an item lacks.
"""

import ast
import gzip
import html
import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from recsys.data import META, download
from recsys.decorators import disk_cache, timed

# Every metadata line starts with the item id; reading it first lets
# read_item_meta parse only the lines it needs (12,101 of 259,204 for Beauty).
_ASIN = re.compile(r"\{'asin': '([^']+)'")


def read_item_meta(path: Path, asins: Collection[str]) -> dict[str, dict[str, Any]]:
    """Read the metadata of `asins` from a gzipped metadata file.

    Each line is a Python literal, not JSON (it uses single quotes), so it is
    read with `ast.literal_eval`, which accepts only literals and never runs
    code, unlike `eval`.

    Args:
        path: The metadata file, one item per line.
        asins: Item ids to keep; other lines are skipped without parsing.

    Returns:
        The metadata of each item found, by item id.

    Raises:
        ValueError: If a line does not start with an item id.
    """
    wanted = set(asins)
    found: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            match = _ASIN.match(line)
            if match is None:
                raise ValueError(f"{path}:{number}: no item id at the start")
            if match.group(1) in wanted:
                found[match.group(1)] = ast.literal_eval(line)
    return found


def item_text(meta: Mapping[str, Any]) -> str:
    """Write an item's title, price, brand and categories as one sentence.

    For example `Title: OPI Nail Lacquer; Price: $12.00; Brand: OPI;
    Categories: Beauty > Makeup > Nails > Nail Polish`. Missing or blank
    fields are left out, HTML entities such as `&amp;` are decoded (some
    titles are escaped twice, `&amp;#8482`, so until nothing changes), and an
    item listed under several category paths gets them all, joined by ` | `.
    """
    parts = []
    title = _clean(meta.get("title"))
    if title:
        parts.append(f"Title: {title}")
    if meta.get("price") is not None:
        parts.append(f"Price: ${meta['price']:.2f}")
    brand = _clean(meta.get("brand"))
    if brand:
        parts.append(f"Brand: {brand}")
    paths = [" > ".join(_clean(c) for c in p) for p in meta.get("categories", [])]
    if paths:
        parts.append("Categories: " + " | ".join(paths))
    return "; ".join(parts)


def _clean(text: str | None) -> str:
    """Decode HTML entities until nothing changes, then strip spaces."""
    text = text or ""
    while (decoded := html.unescape(text)) != text:
        text = decoded
    return text.strip()


@timed
@disk_cache("item_texts")
def load_item_texts(data_dir: Path, items: list[str]) -> list[str]:
    """Download (if needed) the metadata file and return one sentence per item.

    Args:
        data_dir: Where the data files are kept.
        items: Item ids (`Split.items`); the result is in the same order.

    Raises:
        KeyError: If an item has no metadata.
    """
    meta = read_item_meta(download(META, data_dir), items)
    missing = [a for a in items if a not in meta]
    if missing:
        raise KeyError(f"{len(missing)} items have no metadata, e.g. {missing[0]}")
    return [item_text(meta[a]) for a in items]
