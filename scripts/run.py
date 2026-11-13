"""Train one registered model, score it on the valid and test cases, save the log.

Run `uv run python scripts/run.py --model popularity`. The run log (model,
configuration, timings, scores) is written to `runs/` as JSON.
"""

import argparse
from datetime import datetime
from pathlib import Path

import recsys.models  # noqa: F401  (importing it registers the models)
from recsys.data import load_beauty
from recsys.evaluate import evaluate
from recsys.registry import RecommenderMeta
from recsys.runlog import RunLog

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Run one model end to end and return the exit code (0).

    Args:
        argv: Command-line arguments; `None` reads `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=RecommenderMeta.names())
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args(argv)

    with RunLog() as log:
        split = load_beauty(args.data_dir)
        model = RecommenderMeta.get(args.model)()
        model.fit(split)
        for part in ("valid", "test"):
            log.add_result(part, evaluate(model, split, part=part))

    print(f"model  {args.model}")
    for part, scores in log.results.items():
        print(f"{part:5}  " + "  ".join(f"{k} {v:.4f}" for k, v in scores.items()))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = args.runs_dir / f"{args.model}-{stamp}.json"
    log.save(path)
    print(f"saved  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
