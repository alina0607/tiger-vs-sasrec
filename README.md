# TIGER vs SASRec

[![CI](https://github.com/alina0607/tiger-vs-sasrec/actions/workflows/ci.yml/badge.svg)](https://github.com/alina0607/tiger-vs-sasrec/actions/workflows/ci.yml)

Two ways to recommend the next item a shopper will buy, built from scratch and compared on the same data:

| | How it picks the top 10 | Paper |
|---|---|---|
| **SASRec** | Scores every item and takes the 10 highest scores | Kang & McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018 ([arXiv:1808.09781](https://arxiv.org/abs/1808.09781)) |
| **TIGER** | Writes the next item's Semantic ID one code at a time (beam search) | Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023 ([arXiv:2305.05065](https://arxiv.org/abs/2305.05065)) |

The project is also a study in Python design: it shows my own function decorators, a metaclass
and an inheritance hierarchy, each written for a concrete need in the pipeline. The models use
PyTorch.

> **Status:** work in progress. This README describes the plan; each section is filled in
> as the code lands, with every number taken from a run.

## Data

Amazon review data (2014 release), **Beauty** category, 5-core, the same data TIGER uses
(22,363 users, 12,101 items, mean sequence length 8.87; TIGER Table 6).

- Split: leave-one-out per user (last item = test, second to last = validation).
- Metrics: Recall@5/10 and NDCG@5/10, ranking against **all** items.
- The data is not included in this repository. A download script will fetch it from the
  [dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html); please cite
  the dataset authors if you use it.

Why this dataset: TIGER Table 1 reports both TIGER and SASRec on it under the same full-ranking
evaluation, so my numbers can sit next to published ones.

| Beauty | Recall@10 | NDCG@10 | Source |
|---|---|---|---|
| SASRec | 0.0605 | 0.0318 | TIGER paper, Table 1 |
| TIGER | 0.0648 | 0.0384 | TIGER paper, Table 1 |
| TIGER (independent reimplementation) | 0.0594 | 0.0321 | [XiaoLongtaoo/TIGER](https://github.com/XiaoLongtaoo/TIGER) |
| Popularity (this repo) | – | – | not run yet |
| SASRec (this repo) | – | – | not run yet |
| TIGER (this repo) | – | – | not run yet |

## Python design (planned)

| Feature | Where | What it is for |
|---|---|---|
| Inheritance | `Recommender` → `ScoringRecommender` → `PopularityRecommender`, `SASRec`; later `GenerativeRecommender` → `TIGER` | The two retrieval styles in the table above share one evaluation loop; each subclass writes only how it produces scores or IDs |
| Metaclass | `RecommenderMeta(abc.ABCMeta)` | Registers each model by name, checks required class attributes when the class is defined, and checks each model after its `__init__` has finished (the part `__init_subclass__` cannot do) |
| Decorators | `@timed`, `@seeded(42)`, `@disk_cache(...)`, `@metric(...)` | Timing, reproducible seeds, caching slow steps (text embeddings, preprocessing) with `hashlib` + `pickle`, and registering metrics |

A table mapping each feature to its exact file and line will replace this one once the code exists.

## Development

Python 3.11 or newer, managed with [uv](https://docs.astral.sh/uv/). `uv sync` creates the
environment from `uv.lock`, including the development tools:

```bash
uv sync
uv run ruff format --check .   # formatting
uv run ruff check .            # lint
uv run mypy recsys tests       # type check
uv run pytest                  # tests
```

`uv run pre-commit install` runs the formatting, lint and type checks before every commit
([`.pre-commit-config.yaml`](.pre-commit-config.yaml)); pytest runs in CI.

GitHub Actions runs the same four checks on every push and pull request
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). On Linux the CPU build of PyTorch is
installed, so CI does not download the CUDA libraries.

## Roadmap

1. Package skeleton and test setup (pytest, ruff, mypy, CI)
2. Model registry (metaclass) and base classes
3. Decorators
4. Data download and preprocessing (check: 22,363 users, 12,101 items)
5. Metrics and full-ranking evaluation
6. Popularity baseline
7. SASRec
8. Item text embeddings (Sentence-T5)
9. Semantic IDs with residual k-means
10. TIGER: sequence-to-sequence model and constrained beam search
11. RQ-VAE Semantic IDs, three seeds, results table
