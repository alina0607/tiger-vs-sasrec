import random

import pytest
import torch
from torch import nn

from recsys.config import SASRecConfig
from recsys.data import Split
from recsys.evaluate import evaluate
from recsys.models.sasrec import SASRec, SASRecNet, _sample_negatives
from recsys.runlog import RunLog

N_ITEMS = 20
SMALL = SASRecConfig(
    hidden=16,
    blocks=1,
    heads=1,
    max_len=8,
    dropout=0.0,
    lr=0.01,
    batch_size=16,
    max_epochs=30,
    eval_every=5,
    patience=3,
)


def counting_split() -> Split:
    """Every user buys consecutive items: the next item is always the last one + 1.

    Sequences wrap around (19 is followed by 0), so every transition, including the
    ones the test answers need, appears in some user's training part.
    """
    sequences = tuple(
        tuple((s + i) % N_ITEMS for i in range(6))
        for s in range(N_ITEMS)
        for _ in range(3)
    )
    return Split(
        sequences, tuple(f"u{i}" for i in range(len(sequences))), tuple("x" * N_ITEMS)
    )


def fitted() -> SASRec:
    model = SASRec(SMALL, device="cpu")
    model.fit(counting_split())
    return model


def test_hidden_states_have_one_vector_per_position():
    net = SASRecNet(N_ITEMS, SMALL)
    assert net(torch.tensor([[N_ITEMS] * 5 + [1, 2, 3]])).shape == (1, 8, 16)


def test_weights_start_small_and_padding_starts_at_zero():
    net = SASRecNet(N_ITEMS, SMALL)
    # PyTorch's default N(0, 1) would give a standard deviation near 1.
    assert net.item_emb.weight[:N_ITEMS].std() < 0.5
    assert torch.all(net.item_emb.weight[N_ITEMS] == 0)


def test_a_position_never_sees_the_items_after_it():
    net = SASRecNet(N_ITEMS, SMALL).eval()
    a = net(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))
    b = net(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 19]]))  # only the last item differs
    assert torch.allclose(a[0, :7], b[0, :7])
    assert not torch.allclose(a[0, 7], b[0, 7])


def test_padding_row_stays_zero_after_training():
    model = fitted()
    assert model.net is not None
    assert torch.all(model.net.item_emb.weight[N_ITEMS] == 0)


def test_a_history_longer_than_max_len_is_read_from_its_end():
    model = fitted()
    long = list(range(0, 12))
    assert torch.equal(model.score_all([long]), model.score_all([long[-8:]]))


def test_learns_that_the_next_item_follows_the_last_one():
    model = fitted()
    scores = evaluate(model, counting_split(), ks=(1, 10))
    # Every item is equally popular here, so guessing gets recall@1 near 1/20.
    assert scores["recall@1"] > 0.7
    assert scores["recall@10"] == 1.0


def test_same_seed_trains_the_same_model():
    a, b = fitted(), fitted()
    histories = [[1, 2, 3], [5, 6]]
    assert torch.equal(a.score_all(histories), b.score_all(histories))


def test_best_validation_checkpoint_is_kept():
    model = fitted()
    best = max(h["ndcg@10"] for h in model.history)
    valid = evaluate(model, counting_split(), part="valid", ks=(10,))["ndcg@10"]
    assert valid == pytest.approx(best)


def test_negatives_are_items_the_user_never_bought():
    targets = torch.tensor([[N_ITEMS, 1, 2], [N_ITEMS, N_ITEMS, 3]])
    seen = [set(range(N_ITEMS - 1)), {3}]  # user 0 bought everything but item 19
    random.seed(0)
    negatives = _sample_negatives(targets, seen, N_ITEMS)
    assert negatives.tolist()[0] == [N_ITEMS, N_ITEMS - 1, N_ITEMS - 1]
    assert negatives[1, 2].item() != 3
    assert negatives[1, :2].tolist() == [N_ITEMS, N_ITEMS]  # padding stays padding


def test_scoring_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="call fit"):
        SASRec(SMALL, device="cpu").score_all([[1]])


def test_has_a_network_instead_of_being_one():
    model = fitted()
    assert isinstance(model.net, SASRecNet)
    assert not isinstance(model, nn.Module)
    assert SASRec.name == "sasrec"


def test_fit_is_timed_and_the_model_is_logged_with_seed_67():
    with RunLog() as log:
        SASRec(SMALL, device="cpu").fit(counting_split())
    assert log.models[0]["name"] == "sasrec"
    assert log.models[0]["config"]["seed"] == 67
    assert "SASRec.fit" in log.timings
