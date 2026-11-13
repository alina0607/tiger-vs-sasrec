from dataclasses import dataclass

import pytest

from recsys.base import Recommender
from recsys.config import ModelConfig
from recsys.registry import RecommenderMeta


@dataclass(frozen=True)
class ToyConfig(ModelConfig):
    size: int = 3


# Jobs 1 and 2: checks and registration at the ``class`` statement.


def test_concrete_model_is_registered_by_name():
    class Toy(Recommender):
        name = "toy"
        config_cls = ToyConfig

        def fit(self, split):
            pass

        def recommend(self, histories, k):
            return []

    assert RecommenderMeta.get("toy") is Toy
    assert RecommenderMeta.names() == ["toy"]


def test_abstract_class_needs_no_name_and_is_not_registered():
    class StillAbstract(Recommender):
        pass

    assert RecommenderMeta.names() == []
    assert StillAbstract.__abstractmethods__


def test_missing_name_fails_at_class_definition():
    with pytest.raises(TypeError, match="must set a class attribute 'name'"):

        class NoName(Recommender):
            config_cls = ToyConfig

            def fit(self, split):
                pass

            def recommend(self, histories, k):
                return []


def test_name_inherited_from_a_parent_is_rejected():
    class Parent(Recommender):
        name = "parent"
        config_cls = ToyConfig

        def fit(self, split):
            pass

        def recommend(self, histories, k):
            return []

    with pytest.raises(TypeError, match="must set a class attribute 'name'"):

        class Child(Parent):
            pass


@pytest.mark.parametrize(
    "bad", [None, dict, ToyConfig()], ids=["none", "not-a-config", "an-instance"]
)
def test_config_cls_must_be_a_model_config(bad):
    with pytest.raises(TypeError, match="subclass of ModelConfig"):

        class BadConfig(Recommender):
            name = "bad"
            config_cls = bad

            def fit(self, split):
                pass

            def recommend(self, histories, k):
                return []


def test_duplicate_name_is_rejected():
    class First(Recommender):
        name = "same"
        config_cls = ToyConfig

        def fit(self, split):
            pass

        def recommend(self, histories, k):
            return []

    with pytest.raises(TypeError, match="already used"):

        class Second(Recommender):
            name = "same"
            config_cls = ToyConfig

            def fit(self, split):
                pass

            def recommend(self, histories, k):
                return []

    assert RecommenderMeta.get("same") is First


def test_unknown_name_lists_the_registered_ones():
    class Toy(Recommender):
        name = "toy"
        config_cls = ToyConfig

        def fit(self, split):
            pass

        def recommend(self, histories, k):
            return []

    with pytest.raises(KeyError, match="registered: toy"):
        RecommenderMeta.get("sasrec")


def test_registry_view_is_read_only():
    with pytest.raises(TypeError):
        # The write is the point of the test; mypy rightly reports it.
        RecommenderMeta.registry()["toy"] = object  # type: ignore[index]


# Job 3: the finished object is checked on every instantiation.


@pytest.fixture
def toy_cls():
    class Toy(Recommender):
        name = "toy"
        config_cls = ToyConfig

        def __init__(self, config):
            self.config = config

        def fit(self, split):
            pass

        def recommend(self, histories, k):
            return []

    return Toy


def test_correct_config_is_accepted(toy_cls):
    assert toy_cls(ToyConfig(size=5)).config.size == 5


@pytest.mark.parametrize(
    ("config", "type_name"),
    [({"size": 5}, "dict"), (ModelConfig(), "ModelConfig")],
    ids=["dict", "parent-config"],
)
def test_wrong_config_type_is_rejected(toy_cls, config, type_name):
    with pytest.raises(TypeError, match=f"must be a ToyConfig, got {type_name}$"):
        toy_cls(config)


def test_missing_config_is_rejected():
    class Forgetful(Recommender):
        name = "forgetful"
        config_cls = ToyConfig

        def __init__(self, config):
            pass  # never stores the config

        def fit(self, split):
            pass

        def recommend(self, histories, k):
            return []

    with pytest.raises(TypeError, match="got NoneType"):
        Forgetful(ToyConfig())


def test_check_runs_after_every_init():
    # The parent's __init__ runs before the child stores its config. A
    # check inside the parent's __init__ would see no config; the
    # metaclass check runs after both and sees the finished object.
    seen_by_parent = []

    class Parent(Recommender):
        def __init__(self):
            seen_by_parent.append(getattr(self, "config", None))

    class Child(Parent):
        name = "child"
        config_cls = ToyConfig

        def __init__(self, config):
            super().__init__()
            self.config = config

        def fit(self, split):
            pass

        def recommend(self, histories, k):
            return []

    model = Child(ToyConfig())
    assert seen_by_parent == [None]
    assert isinstance(model.config, ToyConfig)


# Why the metaclass and not only ``__init_subclass__``: the hook runs once per
# class definition, so it can register classes and check their attributes
# (jobs 1 and 2), but it never runs when an instance is created (job 3).


def test_init_subclass_runs_once_per_class_not_per_instance():
    calls = []

    class Base:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            calls.append(cls.__name__)

    class Model(Base):
        pass

    Model()
    Model()
    assert calls == ["Model"]


def test_metaclass_call_runs_on_every_instance():
    calls = []

    class CountingMeta(type):
        def __call__(cls, *args, **kwargs):
            calls.append(cls.__name__)
            return super().__call__(*args, **kwargs)

    class Model(metaclass=CountingMeta):
        pass

    Model()
    Model()
    assert calls == ["Model", "Model"]
