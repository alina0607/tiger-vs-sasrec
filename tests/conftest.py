import pytest

from recsys.registry import RecommenderMeta


@pytest.fixture(autouse=True)
def empty_registry(monkeypatch):
    """Give every test an empty model registry and restore the real one afterwards.

    Model classes defined inside a test register themselves; without this,
    they would leak into other tests and into the real registry.
    """
    monkeypatch.setattr(RecommenderMeta, "_registry", {})
