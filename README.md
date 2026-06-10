# pybeh_pd

A lightweight, pure-Python wrapper around [pybeh](https://github.com/pennmem/pybeh)
for analyzing and plotting **free-recall** behavioral data with pandas and seaborn.
It re-expresses pybeh's matrix-based analyses as functions that take **tidy,
long-format pandas DataFrames** of presentation/recall events, and adds
within-subject confidence-interval helpers for plotting.

A minimal subset of `pybeh` is vendored (`pybeh_pd._pybeh`), so **no separate
`pybeh` install is required**.

## What it provides

Given a long-format events DataFrame (one row per presented/recalled item), the
`pd_*` functions compute, per subject:

- **`pd_crp`** — lag conditional response probability (lag-CRP; the temporal
  contiguity effect).
- **`pd_temp_fact`** — temporal clustering factor (Polyn, Norman & Kahana, 2009).
- **`pd_sem_crp`** — semantic CRP, binned by a similarity space.
- **`pd_dist_fact`** — distance/similarity clustering factor.
- **`pd_min_crp` / `pd_min_temp_fact`** — repeated-presentation-aware variants.
- **`pd_sem_crp_list`, `pd_dist_fact_list`, and `*_sub`** — per-list / per-subject
  aggregating variants.
- **CI helpers** for within-subject error bars: `cousineau`
  (Cousineau–Morey–O'Brien), `loftus_masson_analytic`, and the Loftus–Masson
  Kahana ports.

The underlying matrix builders (`make_recalls_matrix`, `get_all_matrices`, ...)
and pybeh primitives (`crp`, `temp_fact`, ...) are also exposed.

## Install

Pure Python — install from source with pip:

```bash
pip install .                                   # from a clone
pip install git+https://github.com/pennmem/pybeh_pd.git   # from GitHub
pip install -e .                                # editable (for development)
```

Requires Python 3.11–3.13 and numpy / pandas / scipy (installed automatically).

## Quickstart

```python
import pandas as pd
import pybeh_pd as pb

# One subject, two 4-item lists. WORD rows are presentations; REC_WORD rows are
# recalls. `itemno` is the item id; recalls reference the presented item numbers.
events = pd.DataFrame({
    "subject": "subj1", "session": 0,
    "list":   [0, 0, 0, 0,  0, 0, 0,    1, 1, 1, 1,  1, 1, 1],
    "type":   ["WORD"] * 4 + ["REC_WORD"] * 3 + ["WORD"] * 4 + ["REC_WORD"] * 3,
    "itemno": [11, 12, 13, 14,  12, 13, 14,   21, 22, 23, 24,  23, 22, 21],
})

# Expects one subject's events; group by subject for many-subject data:
#   events.groupby("subject").apply(pb.pd_crp, itemno_column="itemno")
crp = pb.pd_crp(events, lag_num=3)
print(crp[["lag", "prob"]])
#   lag 0 is NaN; forward (+1) and backward (-1) adjacent transitions dominate.

print("temporal clustering factor:", pb.pd_temp_fact(events))
```

The default column names are `subject` / `session` / `list` (the trial index),
`type` (with `"WORD"` presentations and `"REC_WORD"` recalls), and `itemno`; all
are overridable via keyword arguments (e.g. `itemno_column="item_num"`).

See [`examples/`](examples/) for notebooks demonstrating FR1, catFR1, repFR1, and
the Loftus–Masson confidence intervals on real datasets.

## Development

This repo uses [pixi](https://pixi.sh) for a reproducible dev environment:

```bash
pixi run build      # confirm the package imports
pixi run test       # run the test suite
pixi run test-cov   # with coverage
```

Or with plain pip/pytest:

```bash
pip install -e ".[test]"
pytest
```

The test suite includes a golden-master **behavior-lock** regression suite,
per-function unit tests, and integration tests on a committed 5-session sample of
real ltpFR2 free-recall data (no external data access required).

## Continuous integration

GitHub Actions runs the suite across Python 3.11–3.13 and both numpy 1.x and 2.x
(pandas 2.x/3.x), plus a wheel/sdist build and a pyright + ruff lint job.

## License

MIT.
