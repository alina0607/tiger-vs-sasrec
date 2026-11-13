"""Download the Beauty files and print the statistics of the split.

Run `uv run python scripts/prepare_data.py`; add `--meta` to also fetch the
item metadata used in step 8.
"""

import argparse
from pathlib import Path

from recsys.data import META, download, load_beauty
from recsys.runlog import RunLog

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--meta", action="store_true", help="also fetch item metadata")
    args = parser.parse_args(argv)

    with RunLog() as log:
        split = load_beauty(args.data_dir)
    if args.meta:
        download(META, args.data_dir)

    mean = split.n_interactions / split.n_users
    print(f"users        {split.n_users:,}")
    print(f"items        {split.n_items:,}")
    print(f"interactions {split.n_interactions:,}")
    print(f"mean length  {mean:.3f}")
    for label, seconds in log.timings.items():
        print(f"{label} took {seconds[0]:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
