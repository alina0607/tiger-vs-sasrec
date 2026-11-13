import gzip
import hashlib
import http.server
import json
import threading
from pathlib import Path

import pytest

from recsys.data import (
    REVIEWS,
    RemoteFile,
    Review,
    Split,
    download,
    k_core,
    leave_one_out,
    load_beauty,
    read_reviews,
    sha256_of,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def remote_for(content: bytes) -> RemoteFile:
    return RemoteFile("f.bin", len(content), hashlib.sha256(content).hexdigest())


@pytest.fixture
def server(tmp_path):
    """Serve tmp_path/served over HTTP on 127.0.0.1 and count the requests."""
    root = tmp_path / "served"
    root.mkdir()
    requests = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def do_GET(self):
            requests.append(self.path)
            super().do_GET()

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{httpd.server_port}/f.bin", requests
    httpd.shutdown()
    httpd.server_close()


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "x"
    path.write_bytes(b"a" * 3_000_000)  # more than one 1 MiB chunk
    assert sha256_of(path) == hashlib.sha256(b"a" * 3_000_000).hexdigest()


def test_download_verifies_and_keeps_the_file(tmp_path, server):
    root, url, requests = server
    (root / "f.bin").write_bytes(b"hello")
    path = download(remote_for(b"hello"), tmp_path / "out", url=url)
    assert path.read_bytes() == b"hello"
    assert requests == ["/f.bin"]


def test_download_skips_a_verified_copy(tmp_path, server):
    root, url, requests = server
    (root / "f.bin").write_bytes(b"hello")
    download(remote_for(b"hello"), tmp_path, url=url)
    download(remote_for(b"hello"), tmp_path, url=url)
    assert requests == ["/f.bin"]


def test_download_rejects_a_wrong_hash_and_leaves_nothing(tmp_path, server):
    root, url, _ = server
    (root / "f.bin").write_bytes(b"hellO")  # same size, different bytes
    with pytest.raises(ValueError, match="SHA-256"):
        download(remote_for(b"hello"), tmp_path, url=url)
    assert list(tmp_path.iterdir()) == [root]


def test_download_rejects_a_wrong_size(tmp_path, server):
    root, url, _ = server
    (root / "f.bin").write_bytes(b"hello!")
    with pytest.raises(ValueError, match="6 bytes, expected 5"):
        download(remote_for(b"hello"), tmp_path / "out", url=url)
    assert not (tmp_path / "out" / "f.bin").exists()


def test_download_refuses_a_changed_existing_file(tmp_path):
    (tmp_path / "f.bin").write_bytes(b"hellO")
    with pytest.raises(ValueError, match="SHA-256"):
        download(remote_for(b"hello"), tmp_path, url="http://unused.invalid/")


def test_read_reviews_keeps_file_order(tmp_path):
    path = tmp_path / "r.json.gz"
    rows = [
        {"reviewerID": "u2", "asin": "b", "unixReviewTime": 20, "overall": 5.0},
        {"reviewerID": "u1", "asin": "a", "unixReviewTime": 10},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    assert read_reviews(path) == [Review("u2", "b", 20), Review("u1", "a", 10)]


def test_k_core_keeps_a_file_that_is_already_k_core():
    reviews = [Review(u, i, 0) for u in ("u1", "u2") for i in ("a", "b")]
    assert k_core(reviews, k=2) == reviews


def test_k_core_repeats_until_stable():
    # k = 2. Item b has 1 review -> drop it; then u1 has 1 -> drop u1;
    # then a has 1 -> drop it; then u2 has 1 -> drop u2. Four passes.
    reviews = [
        Review("u1", "a", 0),
        Review("u1", "b", 0),
        Review("u2", "a", 0),
        Review("u2", "c", 0),
        Review("u3", "c", 0),
        Review("u3", "d", 0),
        Review("u4", "c", 0),
        Review("u4", "d", 0),
    ]
    assert k_core(reviews, k=2) == reviews[4:]


def test_leave_one_out_sorts_stably_and_numbers_by_first_appearance():
    reviews = [
        Review("u1", "c", 30),
        Review("u1", "b", 10),
        Review("u1", "a", 10),  # same day as b: stays after it, not sorted by name
        Review("u2", "d", 5),
        Review("u2", "b", 7),
        Review("u2", "c", 9),
    ]
    split = leave_one_out(reviews)
    assert split.users == ("u1", "u2")
    assert split.items == ("b", "a", "c", "d")
    assert split.sequences == ((0, 1, 2), (3, 0, 2))
    assert split.train() == [(0,), (3,)]
    assert split.valid() == ([(0,), (3,)], [1, 0])
    assert split.test() == ([(0, 1), (3, 0)], [2, 2])


def test_split_rejects_sequences_shorter_than_three():
    with pytest.raises(ValueError, match="1 users have fewer than 3 items"):
        Split(((0, 1),), ("u",), ("a", "b"))


@pytest.mark.skipif(
    not (DATA_DIR / REVIEWS.name).exists(), reason="Beauty data not downloaded"
)
def test_beauty_matches_tiger_table_6():
    split = load_beauty(DATA_DIR)
    assert (split.n_users, split.n_items) == (22_363, 12_101)
    assert split.n_interactions == 198_502
    # 198,502 / 22,363 = 8.876...; the table truncates it to 8.87.
    assert split.n_interactions / split.n_users == pytest.approx(8.8764, abs=1e-4)
