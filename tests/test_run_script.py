import importlib.util
import json
from pathlib import Path

import pytest

from recsys.data import Split
from recsys.models.popularity import PopularityRecommender
from recsys.registry import RecommenderMeta

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run.py"
SPLIT = Split(
    sequences=((0, 1, 2), (0, 2, 1), (1, 0, 2)),
    users=("a", "b", "c"),
    items=tuple(str(i) for i in range(12)),
)


@pytest.fixture
def run_module(monkeypatch):
    """scripts/run.py loaded as a module, reading SPLIT instead of the real data."""
    monkeypatch.setitem(RecommenderMeta._registry, "popularity", PopularityRecommender)
    spec = importlib.util.spec_from_file_location("run_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_beauty", lambda data_dir: SPLIT)
    return module


def test_run_writes_one_log_with_model_timings_and_both_parts(run_module, tmp_path):
    assert run_module.main(["--model", "popularity", "--runs-dir", str(tmp_path)]) == 0
    (path,) = tmp_path.glob("popularity-*.json")
    log = json.loads(path.read_text(encoding="utf-8"))
    assert log["models"] == [{"name": "popularity", "config": {"seed": 42}}]
    assert set(log["results"]) == {"valid", "test"}
    assert set(log["results"]["test"]) == {"ndcg@5", "ndcg@10", "recall@5", "recall@10"}
    assert "PopularityRecommender.fit" in log["timings"]


def test_unknown_model_is_rejected_by_argparse(run_module, tmp_path):
    with pytest.raises(SystemExit):
        run_module.main(["--model", "nope", "--runs-dir", str(tmp_path)])
