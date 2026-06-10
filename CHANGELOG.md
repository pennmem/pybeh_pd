# Changelog

All notable changes to `pybeh_pd` are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## 0.1.0 — 2026-06-10

First packaged release. The package was modernized from a single-file,
`sys.path.append`-style module into a clean, tested, pip-installable package
**without changing any observable analysis behavior** (a golden-master regression
suite locks the numerical output).

### Packaging & build
- Converted to a `src/` layout package: `pybeh_pd/{__init__, _core, _pybeh}.py`.
  The vendored pybeh subset (formerly `pybeh_copy.py`) is now the internal
  `pybeh_pd._pybeh`, imported via a proper intra-package import (no more reliance
  on the repo root being on `sys.path`).
- Added a `pyproject.toml` (hatchling backend); installable via
  `pip install .` / `pip install -e .` / `pip install git+https://…`. Builds clean
  wheel + sdist.
- Single-sourced `__version__` (read by hatchling).
- Added a `pixi.toml` reproducible dev environment (Python 3.11/3.12/3.13) with
  `build` / `test` / `test-cov` tasks.
- Moved example notebooks and `LoftusMasson.m` into `examples/` (excluded from the
  wheel); removed hardcoded `sys.path.append` lines from the notebooks.
- Added an MIT `LICENSE`.

### Compatibility
- Supports Python 3.11–3.13 and both numpy 1.x and 2.x (pandas 2.x/3.x). Verified
  green on numpy 1.24/1.26/2.4 across pandas 2.3/3.0.

### Tests & CI
- Added a golden-master **behavior-lock** regression suite plus comprehensive
  per-function unit tests (hand-coded values, mathematical properties, edge
  cases) covering all public functions.
- Added real-data integration tests on a committed 5-session ltpFR2 sample (no
  rhino/cmlreaders needed at test time), reproducing the temporal contiguity
  effect and temporal clustering factor.
- Added GitHub Actions: test matrix (Python 3.11–3.13 × numpy 1/2), wheel/sdist
  build, and a pyright + ruff lint job.

### Code quality
- Added pyright type annotations across the package (0 errors / 0 warnings,
  basic mode); added `pyrightconfig.json`.
- Removed pyflakes-flagged dead imports and unused locals (behavior-neutral).

### Known bugs (discovered, NOT fixed in this release)
These were found during modernization and intentionally left unchanged so the
behavior baseline stays clean; they are pinned by tests and will be fixed in a
follow-up against the regression suite:
- **B1** `loftus_masson_unequal_variance_kahana` references an undefined `mat`
  (raises on every call).
- **B2** `dist_fact` calls `warnings.warn` without importing `warnings`.
- **B3** `pd_dist_fact_list`'s `if count == np.nan` guard never fires.
- **B4** `pd_min_temp_fact` with the default `max_n_reps=1` raises `IndexError`.
- **B5** the percentile-rank helpers crash on a Python-scalar `actual` under
  numpy 2.
- **B6** operator-precedence bug misclassifies NaN-padded recalls as intrusions
  in `make_recalls_matrix` / `make_poss_recalls_matrix`.
- **B7** `cousineau` with a single condition raises `ZeroDivisionError`.
- **B8** the documented `groupby('subject').apply(pd_crp, …)` idiom breaks on
  pandas 3 (the grouping column is dropped); use an explicit per-subject loop.
