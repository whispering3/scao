# Contributing to SCAO

Thank you for your interest in contributing! This document explains how to get set up, what kinds of contributions are welcome, and the process for submitting changes.

---

## Getting started

```bash
git clone https://github.com/whispering3/scao
cd scao
pip install -e ".[dev]"
pytest scao/tests/ -v    # should show 32 passed, 27 passed, 1 skipped
```

---

## What we welcome

| Type | Examples |
|---|---|
| **Bug fixes** | Incorrect gradient scaling, memory leak, wrong dtype cast |
| **Performance** | Faster eigendecomposition, lower memory footprint |
| **New backends** | ROCm support, MPS (Apple Silicon), XLA/TPU |
| **Integrations** | Lightning, Accelerate, DeepSpeed ZeRO-3 |
| **Tests** | Edge cases, new model architectures, fp16 paths |
| **Documentation** | Typos, clearer explanations, new examples |
| **Benchmarks** | Results at larger scales (300M+, 1B+) |

---

## Code style

- **Formatter**: `ruff format scao/`
- **Linter**: `ruff check scao/`
- **Type-checker**: `mypy scao/ --ignore-missing-imports`
- All public functions must have type annotations
- All new features must include tests in `scao/tests/`

---

## Running tests

```bash
# All tests
pytest scao/tests/ -v

# With coverage
pytest scao/tests/ -v --cov=scao --cov-report=term-missing

# Single test file
pytest scao/tests/test_optimizer.py -v -k "test_phase_transition"
```

---

## Submitting a pull request

1. Fork the repo and create a branch: `git checkout -b fix/my-bug`
2. Make your changes and add tests
3. Run `ruff check scao/ && mypy scao/ && pytest scao/tests/ -v`
4. Open a pull request against `main` with a clear description
5. Reference any related issues with `Fixes #123`

---

## Reporting issues

Use [GitHub Issues](https://github.com/whispering3/scao/issues). Include:
- Python version, PyTorch version, OS
- Minimal reproducible example
- Full traceback
- Expected vs actual behavior

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
