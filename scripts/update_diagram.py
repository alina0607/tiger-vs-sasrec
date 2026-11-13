"""Regenerate the class diagram in docs/ with cartograph.

The pre-commit hook runs this and CI checks that docs/ is up to date. By default
the diagram covers the recsys package only; --with-tests also draws the test
classes and writes a separate, untracked pair of files. If docs/classes.layout.json
exists (a board saved with cartograph's save button), the page starts from it.
"""

import argparse
from pathlib import Path

from cartograph.cli import main as extract

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LAYOUT = DOCS / "classes.layout.json"
# Local notes and this script are not part of the package.
ALWAYS_EXCLUDED = ("notes", "scripts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate docs/classes.*")
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="include tests/ and write docs/classes_with_tests.* instead",
    )
    args = parser.parse_args(argv)

    excluded = ALWAYS_EXCLUDED if args.with_tests else (*ALWAYS_EXCLUDED, "tests")
    stem = "classes_with_tests" if args.with_tests else "classes"
    DOCS.mkdir(exist_ok=True)
    cli_args = [
        str(ROOT),
        "--output",
        str(DOCS / f"{stem}.seed.json"),
        "--html",
        str(DOCS / f"{stem}.html"),
    ]
    for pattern in excluded:
        cli_args += ["--exclude", pattern]
    if LAYOUT.exists():
        cli_args += ["--layout", str(LAYOUT)]
    return extract(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())
