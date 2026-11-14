import random

import pytest
import torch

from recsys.decorators import seeded


def draw():
    return random.random(), torch.rand(3).tolist()


def test_same_seed_gives_the_same_numbers():
    with seeded(67):
        first = draw()
    with seeded(67):
        second = draw()
    assert first == second


def test_the_numbers_do_not_depend_on_what_ran_before():
    random.seed(1)
    torch.manual_seed(1)
    with seeded(67):
        first = draw()
    random.seed(2)
    torch.manual_seed(2)
    with seeded(67):
        second = draw()
    assert first == second


def test_different_seeds_give_different_numbers():
    with seeded(67):
        a = draw()
    with seeded(68):
        b = draw()
    assert a != b


def test_the_outside_state_is_restored():
    random.seed(1)
    torch.manual_seed(1)
    expected = draw()
    random.seed(1)
    torch.manual_seed(1)
    with seeded(67):
        draw()
    assert draw() == expected  # as if the block never ran


def test_state_is_restored_even_when_the_block_raises():
    random.seed(1)
    torch.manual_seed(1)
    expected = draw()
    random.seed(1)
    torch.manual_seed(1)
    with pytest.raises(RuntimeError, match="boom"):
        draw_inside_then_fail()
    assert draw() == expected


def draw_inside_then_fail():
    with seeded(67):
        draw()
        raise RuntimeError("boom")


def test_as_a_decorator_every_call_starts_from_the_seed():
    @seeded(67)
    def sample():
        return draw()

    assert sample() == sample()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_mps_numbers_are_seeded_and_restored():
    with seeded(67):
        a = torch.rand(3, device="mps").cpu()
    with seeded(67):
        b = torch.rand(3, device="mps").cpu()
    assert torch.equal(a, b)
    torch.manual_seed(1)
    expected = torch.rand(3, device="mps").cpu()
    torch.manual_seed(1)
    with seeded(67):
        torch.rand(3, device="mps")
    assert torch.equal(torch.rand(3, device="mps").cpu(), expected)
