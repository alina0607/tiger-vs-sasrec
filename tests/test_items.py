import gzip
from pathlib import Path

import pytest

import recsys.decorators
import recsys.items
from recsys.data import META, REVIEWS, load_beauty
from recsys.items import item_text, load_item_texts, read_item_meta

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

LINES = [
    "{'asin': 'A1', 'title': 'Nail Polish', 'price': 12.0, 'brand': 'OPI', "
    "'categories': [['Beauty', 'Makeup', 'Nails']]}",
    "{'asin': 'B2', 'title': 'Skipped', 'categories': [['Beauty']]}",
    "{'asin': 'C3', 'categories': [['Beauty', 'Hair Care']]}",
]


def write_meta(path, lines):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


@pytest.fixture
def meta_file(tmp_path):
    return write_meta(tmp_path / "meta.json.gz", LINES)


def test_reads_only_the_wanted_items(meta_file):
    meta = read_item_meta(meta_file, ["A1", "C3"])
    assert list(meta) == ["A1", "C3"]
    assert meta["A1"]["price"] == 12.0


def test_a_line_with_code_is_not_run(tmp_path):
    # eval would import os here; literal_eval refuses anything but literals.
    path = write_meta(
        tmp_path / "bad.json.gz", ["{'asin': 'A1', 'x': __import__('os')}"]
    )
    with pytest.raises(ValueError, match="malformed"):
        read_item_meta(path, ["A1"])


def test_a_line_without_an_item_id_is_an_error(tmp_path):
    path = write_meta(tmp_path / "bad.json.gz", ["{'title': 'no id'}"])
    with pytest.raises(ValueError, match=":1: no item id"):
        read_item_meta(path, ["A1"])


def test_text_has_the_papers_fields_in_order():
    text = item_text(
        {
            "title": "Nail Polish",
            "price": 3.0,
            "brand": "OPI",
            "categories": [["Beauty", "Makeup", "Nails"]],
        }
    )
    assert text == (
        "Title: Nail Polish; Price: $3.00; Brand: OPI; "
        "Categories: Beauty > Makeup > Nails"
    )


def test_missing_and_blank_fields_are_left_out():
    text = item_text({"title": " ", "brand": "", "categories": [["Beauty"]]})
    assert text == "Categories: Beauty"


def test_html_entities_are_decoded_even_when_escaped_twice():
    text = item_text(
        {"title": "AVVA&amp;#8482 Liner", "brand": "L&#39;Oreal", "categories": []}
    )
    assert text == "Title: AVVA™ Liner; Brand: L'Oreal"


def test_several_category_paths_are_all_kept():
    text = item_text({"categories": [["Beauty", "Bags"], ["Clothing", "Travel"]]})
    assert text == "Categories: Beauty > Bags | Clothing > Travel"


@pytest.fixture
def fake_download(meta_file, tmp_path, monkeypatch):
    """Skip the size and hash checks of the real metadata file."""
    monkeypatch.setattr(recsys.items, "download", lambda remote, data_dir: meta_file)
    monkeypatch.setattr(recsys.decorators, "CACHE_DIR", tmp_path / "cache")


@pytest.mark.usefixtures("fake_download")
def test_texts_follow_the_order_of_the_items(tmp_path):
    texts = load_item_texts(tmp_path, ["C3", "A1"])
    assert texts[0] == "Categories: Beauty > Hair Care"
    assert texts[1].startswith("Title: Nail Polish")


@pytest.mark.usefixtures("fake_download")
def test_an_item_without_metadata_is_an_error(tmp_path):
    with pytest.raises(KeyError, match="1 items have no metadata, e.g. Z9"):
        load_item_texts(tmp_path, ["A1", "Z9"])


@pytest.mark.skipif(
    not (DATA_DIR / REVIEWS.name).exists() or not (DATA_DIR / META.name).exists(),
    reason="Beauty data not downloaded",
)
def test_beauty_texts(tmp_path, monkeypatch):
    monkeypatch.setattr(recsys.decorators, "CACHE_DIR", tmp_path)
    split = load_beauty(DATA_DIR)
    texts = load_item_texts(DATA_DIR, list(split.items))
    assert len(texts) == 12_101
    assert texts[0] == (
        "Title: OPI Nail Lacquer, Simmer and Shimmer, 0.5-Fluid Ounce; "
        "Price: $12.00; Brand: OPI; Categories: Beauty > Makeup > Nails > Nail Polish"
    )
    assert sum(t.startswith("Title: ") for t in texts) == 12_094
    assert not any("&amp;" in t or "&#" in t for t in texts)
