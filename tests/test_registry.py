import unittest
from dataclasses import dataclass
from unittest import mock

from recsys.base import Recommender
from recsys.config import ModelConfig
from recsys.registry import RecommenderMeta


@dataclass(frozen=True)
class ToyConfig(ModelConfig):
    size: int = 3


class RegistryTestCase(unittest.TestCase):
    """Gives every test an empty registry and restores the real one afterwards.

    Model classes defined inside a test register themselves; without this,
    they would leak into other tests and into the real registry.
    """

    def setUp(self):
        self.enterContext(mock.patch.dict(RecommenderMeta._registry, clear=True))


class TestClassDefinition(RegistryTestCase):
    """Job 1 and 2: checks and registration at the ``class`` statement."""

    def test_concrete_model_is_registered_by_name(self):
        class Toy(Recommender):
            name = "toy"
            config_cls = ToyConfig

            def fit(self, split):
                pass

            def recommend(self, histories, k):
                return []

        self.assertIs(RecommenderMeta.get("toy"), Toy)
        self.assertEqual(RecommenderMeta.names(), ["toy"])

    def test_abstract_class_needs_no_name_and_is_not_registered(self):
        class StillAbstract(Recommender):
            pass

        self.assertEqual(RecommenderMeta.names(), [])
        self.assertTrue(StillAbstract.__abstractmethods__)

    def test_missing_name_fails_at_class_definition(self):
        with self.assertRaisesRegex(TypeError, "must set a class attribute 'name'"):

            class NoName(Recommender):
                config_cls = ToyConfig

                def fit(self, split):
                    pass

                def recommend(self, histories, k):
                    return []

    def test_name_inherited_from_a_parent_is_rejected(self):
        class Parent(Recommender):
            name = "parent"
            config_cls = ToyConfig

            def fit(self, split):
                pass

            def recommend(self, histories, k):
                return []

        with self.assertRaisesRegex(TypeError, "must set a class attribute 'name'"):

            class Child(Parent):
                pass

    def test_config_cls_must_be_a_model_config(self):
        for bad in (None, dict, ToyConfig()):
            with (
                self.subTest(config_cls=bad),
                self.assertRaisesRegex(TypeError, "subclass of ModelConfig"),
            ):

                class BadConfig(Recommender):
                    name = "bad"
                    config_cls = bad

                    def fit(self, split):
                        pass

                    def recommend(self, histories, k):
                        return []

    def test_duplicate_name_is_rejected(self):
        class First(Recommender):
            name = "same"
            config_cls = ToyConfig

            def fit(self, split):
                pass

            def recommend(self, histories, k):
                return []

        with self.assertRaisesRegex(TypeError, "already used"):

            class Second(Recommender):
                name = "same"
                config_cls = ToyConfig

                def fit(self, split):
                    pass

                def recommend(self, histories, k):
                    return []

        self.assertIs(RecommenderMeta.get("same"), First)

    def test_unknown_name_lists_the_registered_ones(self):
        class Toy(Recommender):
            name = "toy"
            config_cls = ToyConfig

            def fit(self, split):
                pass

            def recommend(self, histories, k):
                return []

        with self.assertRaisesRegex(KeyError, "registered: toy"):
            RecommenderMeta.get("sasrec")

    def test_registry_view_is_read_only(self):
        with self.assertRaises(TypeError):
            RecommenderMeta.registry()["toy"] = object


class TestInstanceCheck(RegistryTestCase):
    """Job 3: the finished object is checked on every instantiation."""

    def setUp(self):
        super().setUp()

        class Toy(Recommender):
            name = "toy"
            config_cls = ToyConfig

            def __init__(self, config):
                self.config = config

            def fit(self, split):
                pass

            def recommend(self, histories, k):
                return []

        self.Toy = Toy

    def test_correct_config_is_accepted(self):
        model = self.Toy(ToyConfig(size=5))
        self.assertEqual(model.config.size, 5)

    def test_wrong_config_type_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "must be a ToyConfig, got dict"):
            self.Toy({"size": 5})

    def test_parent_config_type_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "got ModelConfig"):
            self.Toy(ModelConfig())

    def test_missing_config_is_rejected(self):
        class Forgetful(Recommender):
            name = "forgetful"
            config_cls = ToyConfig

            def __init__(self, config):
                pass  # never stores the config

            def fit(self, split):
                pass

            def recommend(self, histories, k):
                return []

        with self.assertRaisesRegex(TypeError, "got NoneType"):
            Forgetful(ToyConfig())

    def test_check_runs_after_every_init(self):
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
        self.assertEqual(seen_by_parent, [None])
        self.assertIsInstance(model.config, ToyConfig)


class TestInitSubclassComparison(unittest.TestCase):
    """Why the metaclass and not only ``__init_subclass__``.

    ``__init_subclass__`` runs once per class definition, so it can register
    classes and check their attributes (jobs 1 and 2). It never runs when an
    instance is created, so it cannot check the finished object (job 3).
    """

    def test_init_subclass_runs_once_per_class_not_per_instance(self):
        calls = []

        class Base:
            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__(**kwargs)
                calls.append(cls.__name__)

        class Model(Base):
            pass

        Model()
        Model()
        self.assertEqual(calls, ["Model"])

    def test_metaclass_call_runs_on_every_instance(self):
        calls = []

        class CountingMeta(type):
            def __call__(cls, *args, **kwargs):
                calls.append(cls.__name__)
                return super().__call__(*args, **kwargs)

        class Model(metaclass=CountingMeta):
            pass

        Model()
        Model()
        self.assertEqual(calls, ["Model", "Model"])


if __name__ == "__main__":
    unittest.main()
