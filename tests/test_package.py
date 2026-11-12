import tomllib
import unittest
from pathlib import Path

import recsys

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


class TestPackage(unittest.TestCase):
    def test_version_matches_pyproject(self):
        with PYPROJECT.open("rb") as f:
            project = tomllib.load(f)["project"]
        self.assertEqual(recsys.__version__, project["version"])


if __name__ == "__main__":
    unittest.main()
