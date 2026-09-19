# Contributing

This repository accompanies [arXiv:2608.14089](https://arxiv.org/abs/2608.14089). Bug reports and
questions are welcome.

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
uv run ruff check .
```

A pull request needs the suite green and ruff clean. If you change behaviour, the test changes with
it in the same commit.
