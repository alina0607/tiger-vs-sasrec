import tomllib
from pathlib import Path

import recsys

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_version_matches_pyproject():
    with PYPROJECT.open("rb") as f:
        project = tomllib.load(f)["project"]
    assert recsys.__version__ == project["version"]
