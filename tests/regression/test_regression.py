"""Behavior-lock regression test (Phase 4 GATE).

Replays the golden-master values captured in ``golden.npz`` against live output
from :func:`cases.build_cases`. Any divergence beyond tolerance fails — this is
what proves the modernization edits (Phases 6-11) changed no observable output.

Golden was captured on the baseline env (numpy 1.24 / pandas 2.3). The same
assertions are expected to hold on newer stacks (numpy 2 / pandas 3) within
tolerance; a cross-env failure here is itself a finding to report.

Re-capture with ``python tests/regression/capture_golden.py`` only after an
*approved, intentional* behavior change.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from cases import build_cases

GOLDEN = os.path.join(os.path.dirname(__file__), "golden.npz")
RTOL = 1e-9
ATOL = 1e-12


@pytest.fixture(scope="module")
def golden():
    assert os.path.exists(GOLDEN), "run capture_golden.py first"
    with np.load(GOLDEN) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="module")
def live():
    return build_cases()


def test_case_set_matches(golden, live):
    """The set of case names must not drift (no silently dropped coverage)."""
    assert set(golden) == set(live), (
        f"only in golden: {set(golden) - set(live)}; "
        f"only in live: {set(live) - set(golden)}"
    )


@pytest.mark.parametrize("name", sorted(build_cases().keys()))
def test_regression(name, golden, live):
    g = golden[name]
    x = np.asarray(live[name], dtype=float)
    assert x.shape == g.shape, f"{name}: shape {x.shape} != golden {g.shape}"
    assert np.allclose(x, g, rtol=RTOL, atol=ATOL, equal_nan=True), (
        f"{name}: values diverged from golden\n live={x}\n gold={g}"
    )


# --- Locked known-broken behaviors (bugs B1, B4 in bug_report.md) -----------
# These are pinned so a later fix is a *deliberate* change, not silent.

def test_pd_min_temp_fact_default_raises():
    """B4: default max_n_reps=1 path raises IndexError (broken)."""
    import pybeh_pd as pb
    from factories import make_experiment
    df = make_experiment(seed=0)["events"]
    with pytest.raises(IndexError):
        pb.pd_min_temp_fact(df)


def test_loftus_masson_unequal_kahana_raises():
    """B1: function references undefined global `mat` -> NameError."""
    import pybeh_pd as pb
    with pytest.raises(NameError):
        pb.loftus_masson_unequal_variance_kahana(np.ones((4, 3)))
