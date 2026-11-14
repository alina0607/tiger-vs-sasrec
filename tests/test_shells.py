"""Classes planned for steps 9 and 11: they exist, relate as designed, cannot run."""

import dataclasses

import pytest

from recsys.base import GenerativeRecommender, Recommender, ScoringRecommender
from recsys.config import (
    ModelConfig,
    RQVAEConfig,
    SemanticIdConfig,
    TIGERConfig,
)
from recsys.models.tiger import TIGER, TIGERNet
from recsys.registry import RecommenderMeta
from recsys.semantic_ids import (
    ResidualKMeansTokenizer,
    RQVAETokenizer,
    SemanticIdTokenizer,
)


def test_the_two_branches_of_recommender():
    assert issubclass(GenerativeRecommender, Recommender)
    assert not issubclass(GenerativeRecommender, ScoringRecommender)
    assert issubclass(TIGER, GenerativeRecommender)


def test_tiger_lists_the_methods_still_to_write():
    assert TIGER.__abstractmethods__ == {"fit", "generate", "recommend"}
    with pytest.raises(TypeError, match="abstract class TIGER"):
        TIGER()  # type: ignore[abstract]


def test_an_abstract_model_is_not_registered():
    # conftest gives each test an empty registry, so define a subclass here:
    # like TIGER it is still abstract, and the metaclass skips it.
    class Planned(TIGER):
        name = "planned"

    assert Planned.__abstractmethods__
    assert "planned" not in RecommenderMeta.names()


@pytest.mark.parametrize("cls", [ResidualKMeansTokenizer, RQVAETokenizer])
def test_tokenizers_are_shells(cls):
    assert issubclass(cls, SemanticIdTokenizer)
    assert cls.__abstractmethods__ == {"fit", "encode", "decode"}
    with pytest.raises(TypeError, match="abstract class"):
        cls()


def test_tiger_config_has_the_papers_settings():
    config = TIGERConfig()
    assert isinstance(config, ModelConfig)
    assert (config.encoder_layers, config.decoder_layers) == (4, 4)
    assert (config.heads, config.head_dim, config.d_model, config.d_ff) == (
        6,
        64,
        128,
        1024,
    )
    assert config.user_tokens == 2000
    assert config.seed == 67
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.d_model = 64  # type: ignore[misc]


def test_semantic_id_configs():
    assert (SemanticIdConfig().levels, SemanticIdConfig().codebook_size) == (3, 256)
    rq = RQVAEConfig()
    assert isinstance(rq, SemanticIdConfig)
    assert (rq.hidden, rq.latent) == ((512, 256, 128), 32)


def test_tiger_net_is_a_module_that_cannot_run_yet():
    net = TIGERNet(TIGERConfig())
    with pytest.raises(NotImplementedError, match="step 11"):
        net(None, None)
