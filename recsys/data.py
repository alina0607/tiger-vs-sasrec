"""Download, read and split the Amazon 2014 Beauty reviews.

The preprocessing follows S3-Rec, whose output TIGER uses: keep users and items
with at least 5 reviews, sort each user's reviews by time with a stable sort
(reviews on the same day keep their file order), and hold out the last item
for testing and the one before it for validation.
"""

import gzip
import hashlib
import json
import os
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from recsys.decorators import timed

_BASE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"


@dataclass(frozen=True)
class RemoteFile:
    """A file to download and the checks it must pass.

    Attributes:
        name: File name, both on the server and on disk.
        size: Size in bytes.
        sha256: Hex SHA-256 of the whole file.
    """

    name: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return _BASE_URL + self.name


REVIEWS = RemoteFile(
    "reviews_Beauty_5.json.gz",
    44_819_276,
    "e5925ec99023f1dc9c7d3dff7ef34cfeacd40d1b981b5525e1e4ba9c8abc18fe",
)
META = RemoteFile(
    "meta_Beauty.json.gz",
    99_156_148,
    "c7977dc0e0ead14ac7df8c1c6de74da00e4bebfad87a0f0c5763d4c5bc9b53a0",
)


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, reading it in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(remote: RemoteFile, dest_dir: Path, *, url: str | None = None) -> Path:
    """Download `remote` into `dest_dir` unless a verified copy is already there.

    The data is written to a `.part` file and renamed only after the size and
    hash match, so an interrupted download never leaves a file that looks done.

    Args:
        remote: The file to fetch.
        dest_dir: Where to store it. Created if missing.
        url: Fetch from here instead of `remote.url` (used by the tests).

    Returns:
        The path of the verified file.

    Raises:
        ValueError: If the existing or downloaded file has the wrong size or hash.
    """
    dest = dest_dir / remote.name
    if dest.exists():
        _verify(dest, remote)
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with (
            urllib.request.urlopen(url or remote.url, timeout=60) as response,
            part.open("wb") as f,
        ):
            while chunk := response.read(1 << 20):
                f.write(chunk)
        _verify(part, remote)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    return dest


def _verify(path: Path, remote: RemoteFile) -> None:
    size = path.stat().st_size
    if size != remote.size:
        raise ValueError(f"{path}: {size} bytes, expected {remote.size}")
    digest = sha256_of(path)
    if digest != remote.sha256:
        raise ValueError(f"{path}: SHA-256 {digest}, expected {remote.sha256}")


class Review(NamedTuple):
    """One review: who reviewed which item, and when (Unix seconds, day precision)."""

    user: str
    item: str
    time: int


def read_reviews(path: Path) -> list[Review]:
    """Read a gzipped review file with one JSON object per line, in file order."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [
            Review(r["reviewerID"], r["asin"], r["unixReviewTime"])
            for r in map(json.loads, f)
        ]


def k_core(reviews: Iterable[Review], k: int = 5) -> list[Review]:
    """Drop users and items with fewer than `k` reviews until none are left.

    Removing a user can push an item below `k` and the other way round, so
    the filter repeats until nothing changes. File order is kept.
    """
    kept = list(reviews)
    while True:
        users = Counter(r.user for r in kept)
        items = Counter(r.item for r in kept)
        smaller = [r for r in kept if users[r.user] >= k and items[r.item] >= k]
        if len(smaller) == len(kept):
            return kept
        kept = smaller


@dataclass(frozen=True)
class Split:
    """Every user's item sequence, cut by leave-one-out.

    User `u`'s sequence is `sequences[u]`, oldest first. Ids are assigned in
    order of first appearance, as S3-Rec does, but start at 0: item `j` is
    column `j` of a score matrix.

    Attributes:
        sequences: Item ids per user, oldest first.
        users: Original user id (`reviewerID`) of each user index.
        items: Original item id (`asin`) of each item index.
    """

    sequences: tuple[tuple[int, ...], ...]
    users: tuple[str, ...]
    items: tuple[str, ...]

    def __post_init__(self) -> None:
        short = sum(len(s) < 3 for s in self.sequences)
        if short:
            raise ValueError(f"{short} users have fewer than 3 items")

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_interactions(self) -> int:
        return sum(map(len, self.sequences))

    def train(self) -> list[tuple[int, ...]]:
        """Training sequences: everything but the last two items."""
        return [s[:-2] for s in self.sequences]

    def valid(self) -> tuple[list[tuple[int, ...]], list[int]]:
        """Validation cases: (history without the last two items, second to last)."""
        return [s[:-2] for s in self.sequences], [s[-2] for s in self.sequences]

    def test(self) -> tuple[list[tuple[int, ...]], list[int]]:
        """Test cases: (history without the last item, last item)."""
        return [s[:-1] for s in self.sequences], [s[-1] for s in self.sequences]


def leave_one_out(reviews: Sequence[Review]) -> Split:
    """Group reviews by user, sort each user's by time and number the ids.

    `list.sort` is stable, so reviews with the same time keep their file
    order; S3-Rec relies on the same behaviour.
    """
    by_user: dict[str, list[Review]] = {}
    for r in reviews:
        by_user.setdefault(r.user, []).append(r)
    item_ids: dict[str, int] = {}
    sequences = []
    for user_reviews in by_user.values():
        user_reviews.sort(key=lambda r: r.time)
        sequences.append(
            tuple(item_ids.setdefault(r.item, len(item_ids)) for r in user_reviews)
        )
    return Split(tuple(sequences), tuple(by_user), tuple(item_ids))


@timed
def load_beauty(data_dir: Path) -> Split:
    """Download (if needed), filter and split the Beauty 5-core reviews."""
    path = download(REVIEWS, data_dir)
    return leave_one_out(k_core(read_reviews(path)))
