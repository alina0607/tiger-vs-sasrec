"""Metaclass that checks, registers and validates every recommender.

``RecommenderMeta`` has three jobs:

1. When a concrete model class is defined, check that it declares ``name``
   and ``config_cls``. A mistake fails at the ``class`` statement, not later
   in a training run.
2. Register the class under its ``name``, so a model can be chosen by a
   string (for example, from the command line).
3. When a model is instantiated, check the finished object after every
   ``__init__`` in its hierarchy has run: it must hold a config of the
   declared type.

Jobs 1 and 2 could also be done with ``__init_subclass__``. Job 3 needs to
run after construction, on every instance, and without relying on each
subclass to call a check; only the metaclass's ``__call__`` wraps the whole
construction.
"""

from abc import ABCMeta
from types import MappingProxyType
from typing import Any, ClassVar

from recsys.config import ModelConfig


class RecommenderMeta(ABCMeta):
    """Metaclass of ``recsys.base.Recommender`` and every model class.

    It derives from ``ABCMeta`` because ``Recommender`` is an abstract base
    class: a class can have only one metaclass, so this one must keep the
    abstract-method checks of ``ABCMeta``.
    """

    # One registry shared by the whole hierarchy: model name -> model class.
    _registry: ClassVar[dict[str, type]] = {}

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwargs: Any,
    ) -> type:
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        # Abstract classes (Recommender, ScoringRecommender, ...) are not
        # models: they need no name and are not registered.
        if cls.__abstractmethods__:
            return cls
        mcls._check_declarations(cls, namespace)
        if cls.name in mcls._registry:
            other = mcls._registry[cls.name]
            raise TypeError(
                f"{cls.__qualname__}: name {cls.name!r} is already used by "
                f"{other.__module__}.{other.__qualname__}"
            )
        mcls._registry[cls.name] = cls
        return cls

    @staticmethod
    def _check_declarations(cls: type, namespace: dict[str, Any]) -> None:
        """Raise ``TypeError`` unless ``cls`` declares a usable name and config."""
        # ``name`` must be set on the class itself: a subclass that inherited
        # its parent's name would silently replace the parent in the registry.
        model_name = namespace.get("name")
        if not isinstance(model_name, str) or not model_name:
            raise TypeError(
                f"{cls.__qualname__} must set a class attribute 'name' "
                f"(a non-empty string) to be registered"
            )
        config_cls = getattr(cls, "config_cls", None)
        if not (isinstance(config_cls, type) and issubclass(config_cls, ModelConfig)):
            raise TypeError(
                f"{cls.__qualname__} must set 'config_cls' to a subclass of ModelConfig"
            )

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # ``super().__call__`` runs ``__new__`` and every ``__init__`` in the
        # hierarchy; the instance returned here is fully constructed.
        instance = super().__call__(*args, **kwargs)
        config = getattr(instance, "config", None)
        if not isinstance(config, cls.config_cls):
            raise TypeError(
                f"{cls.__qualname__}.config must be a {cls.config_cls.__qualname__}, "
                f"got {type(config).__qualname__}"
            )
        return instance

    @classmethod
    def get(cls, name: str) -> type:
        """Return the model class registered under ``name``."""
        try:
            return cls._registry[name]
        except KeyError:
            known = ", ".join(sorted(cls._registry)) or "none"
            raise KeyError(f"no model named {name!r}; registered: {known}") from None

    @classmethod
    def names(cls) -> list[str]:
        """Return the registered model names in alphabetical order."""
        return sorted(cls._registry)

    @classmethod
    def registry(cls) -> MappingProxyType[str, type]:
        """Return a read-only view of the registry."""
        return MappingProxyType(cls._registry)
