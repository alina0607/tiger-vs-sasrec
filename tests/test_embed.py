import pytest
import torch

import recsys.decorators
import recsys.embed
from recsys.embed import MODEL, REVISION, embed_texts
from recsys.runlog import RunLog


class FakeEncoder:
    """Stands in for SentenceTransformer: a text's vector is its length, twice."""

    device = "cpu"

    def __init__(self, loads):
        self.loads = loads

    def encode(self, texts, **kwargs):
        assert kwargs["convert_to_tensor"] is True
        return torch.tensor([[len(t), 2 * len(t)] for t in texts], dtype=torch.float64)


@pytest.fixture
def loads(tmp_path, monkeypatch):
    """Model names `_load_model` was asked for; the cache starts empty."""
    names = []

    def load(model_name, revision):
        names.append((model_name, revision))
        return FakeEncoder(names)

    monkeypatch.setattr(recsys.embed, "_load_model", load)
    monkeypatch.setattr(recsys.decorators, "CACHE_DIR", tmp_path)
    return names


def test_one_float32_cpu_row_per_text(loads):
    vectors = embed_texts(["ab", "abcd"])
    assert vectors.dtype == torch.float32
    assert vectors.device.type == "cpu"
    assert vectors.tolist() == [[2.0, 4.0], [4.0, 8.0]]
    assert loads == [(MODEL, REVISION)]


def test_the_same_texts_are_loaded_from_the_cache_without_the_model(loads):
    first = embed_texts(["ab", "abcd"])
    second = embed_texts(["ab", "abcd"])
    assert torch.equal(first, second)
    assert loads == [(MODEL, REVISION)]


def test_other_texts_model_or_revision_are_encoded_again(loads):
    embed_texts(["ab"])
    embed_texts(["abc"])
    embed_texts(["ab"], "other/model")
    embed_texts(["ab"], revision=None)
    assert loads == [
        (MODEL, REVISION),
        (MODEL, REVISION),
        ("other/model", REVISION),
        (MODEL, None),
    ]


def test_default_model_is_sentence_t5_base_pinned_to_a_commit():
    assert MODEL == "sentence-transformers/sentence-t5-base"
    assert len(REVISION) == 40


@pytest.mark.usefixtures("loads")
def test_every_call_is_timed_including_cache_hits():
    with RunLog() as log:
        embed_texts(["ab"])
        embed_texts(["ab"])
    assert len(log.timings["embed_texts"]) == 2
