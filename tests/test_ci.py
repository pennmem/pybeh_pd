"""Skeptical unit tests for the within-subject confidence-interval helpers in
``pybeh_pd``: ``loftus_masson_analytic``, ``cousineau``,
``loftus_masson_equal_variance_kahana`` and ``loftus_masson_unequal_variance_kahana``.

These are ports of ``LoftusMasson.m`` (Loftus & Masson, 1994, PB&R) and the
Cousineau-Morey-O'Brien within-subject normalization.

Tests only PIN current behavior; source is never modified. Known/discovered bugs
are pinned with ``@pytest.mark.pins_bug`` and a comment, and reported back to the
caller.
"""
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy as sp
from scipy import stats

import pybeh_pd as pb


# ---------------------------------------------------------------------------
# Shared hand-computed fixture: 3 subjects x 2 conditions.
#
# dat = [[1, 3],
#        [2, 5],
#        [4, 6]]
# grand mean = 21/6 = 3.5, grand total = 21
# SS_T   = sum((x-3.5)^2) = 6.25+0.25+2.25+2.25+0.25+6.25 = 17.5
# Srow   = [4, 7, 10];  SSrow = (16+49+100)/2 - 21^2/6 = 82.5 - 73.5 = 9.0
# Scol   = [7, 14];     SScol = (49+196)/3 - 73.5 = 81.6667 - 73.5 = 8.16667
# SSint  = 17.5 - 9.0 - 8.16667 = 0.33333
# df_int = (3*2-1)-(3-1)-(2-1) = 5-2-1 = 2
# MSint  = 0.33333/2 = 0.16667
# SE     = sqrt(0.16667/3) = 0.235702
# t.975(df=2) = 4.302653
# CI     = 0.235702 * 4.302653 = 1.0141449740863315  (same for both columns)
# ---------------------------------------------------------------------------
DAT_3x2 = np.array([[1.0, 3.0], [2.0, 5.0], [4.0, 6.0]])
CI_3x2_EQUAL = 1.0141449740863315


def _long_from_matrix(dat, sub_col="sub", cond_col="cond", value_col="val"):
    """Reshape a subjects x conditions matrix to a tidy long DataFrame."""
    n_subs, n_conds = dat.shape
    rows = [
        {sub_col: s, cond_col: c, value_col: float(dat[s, c])}
        for s in range(n_subs)
        for c in range(n_conds)
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# 1. HAND-CODED VALUE TESTS
# ===========================================================================

@pytest.mark.hand_coded
def test_equal_variance_kahana_hand_value():
    """loftus_masson_equal_variance_kahana on the 3x2 fixture matches the
    by-hand Loftus-Masson Eq. (2) CI, is constant across conditions, and has
    length == n_conditions."""
    ci = pb.loftus_masson_equal_variance_kahana(DAT_3x2)

    # Reconstruct the expected value from hand-computed ANOVA pieces.
    SS_int = 0.3333333333333428  # 17.5 - 9.0 - 8.166666...
    df_int = 2
    MS_int = SS_int / df_int
    crit = sp.stats.t.ppf(0.975, df_int)
    expected = np.sqrt(MS_int / 3) * crit

    assert ci.shape == (2,)                       # one entry per condition
    assert np.allclose(ci, expected)
    assert np.allclose(ci, CI_3x2_EQUAL)
    # equal-variance assumption => identical half-width for every condition
    assert ci[0] == ci[1]


@pytest.mark.hand_coded
def test_analytic_CI_equal_matches_kahana_same_data():
    """HIGH-VALUE CHECK: loftus_masson_analytic's 'CI_equal' column is a second
    implementation of the *same* Loftus-Masson Eq. (2). On identical data it
    must agree with loftus_masson_equal_variance_kahana. (They do agree.)"""
    df = _long_from_matrix(DAT_3x2)
    with warnings.catch_warnings():
        # CI_unequal has a NaN (negative estimator -> sqrt warning) for this
        # tiny matrix; irrelevant to the CI_equal comparison.
        warnings.simplefilter("ignore", RuntimeWarning)
        out = pb.loftus_masson_analytic(df, "sub", "cond", "val")

    ci_kahana = pb.loftus_masson_equal_variance_kahana(DAT_3x2)

    assert list(out.columns) == ["cond", "M_C", "CI_unequal", "CI_equal"]
    assert len(out) == 2
    # CI_equal constant across conditions and equal to the hand value.
    assert np.allclose(out["CI_equal"].values, CI_3x2_EQUAL)
    # The two independent implementations of Eq. (2) agree.
    assert np.allclose(out["CI_equal"].values, ci_kahana)
    # condition means
    assert np.allclose(
        sorted(out["M_C"].values), sorted([7 / 3, 14 / 3])
    )


@pytest.mark.hand_coded
def test_analytic_CI_equal_matches_kahana_balanced_random():
    """Same cross-implementation agreement on a larger, well-behaved balanced
    design (so CI_unequal is real, not NaN)."""
    rng = np.random.default_rng(0)
    n, k = 8, 4
    dat = rng.normal(5, 2, (n, k)) + np.arange(k)
    df = _long_from_matrix(dat)

    out = pb.loftus_masson_analytic(df, "sub", "cond", "val")
    ci_kahana = pb.loftus_masson_equal_variance_kahana(dat)

    assert len(out) == k
    assert np.allclose(out["CI_equal"].values, ci_kahana)
    assert (out["CI_unequal"] >= 0).all()


@pytest.mark.hand_coded
def test_cousineau_adj_and_cmo_hand_values():
    """cousineau:
       adj_<v>     = v + grand_mean - subject_mean   (exact)
       cmo_adj_<v> = sqrt(K/(K-1))*(v - subject_mean) + grand_mean  (K=n_conds)
    Hand-checked on the 3x2 fixture (grand mean 3.5, K=2)."""
    df = _long_from_matrix(DAT_3x2)
    out = pb.cousineau(df, "sub", "cond", "val")

    grand = 3.5
    # subject means: sub0 -> (1+3)/2=2.0, sub1 -> (2+5)/2=3.5, sub2 -> (4+6)/2=5.0
    sub_means = {0: 2.0, 1: 3.5, 2: 5.0}
    K = 2
    factor = math.sqrt(K / (K - 1))

    for _, r in out.iterrows():
        sm = sub_means[r["sub"]]
        exp_adj = r["val"] + grand - sm
        exp_cmo = factor * (r["val"] - sm) + grand
        assert r["adj_val"] == pytest.approx(exp_adj)
        assert r["cmo_adj_val"] == pytest.approx(exp_cmo)

    # spot-check the exact emitted values (regression-grade)
    row0 = out[(out["sub"] == 0) & (out["cond"] == 0)].iloc[0]
    assert row0["adj_val"] == pytest.approx(2.5)
    assert row0["cmo_adj_val"] == pytest.approx(math.sqrt(2) * (1.0 - 2.0) + 3.5)

    # required output columns are present
    for col in ["M", "M_S", "adj_val", "cmo_adj_val"]:
        assert col in out.columns


@pytest.mark.hand_coded
def test_cousineau_within_cols_path_hand_values():
    """The within_cols branch groups the grand mean M by within_cols and the
    subject mean M_S by (sub_cols + within_cols). Hand-checked.

    Design: 2 subjects x 2 conds x 2 within-levels, val = sub*10 + cond + wc.
    For wc=0 group values are {0,1,10,11} -> M=5.5; sub0/wc0 subject mean is
    mean(0,1)=0.5; so adj for (sub0,cond0,wc0,val=0) = 0 + 5.5 - 0.5 = 5.0.
    """
    rows = [
        {"sub": s, "cond": c, "wc": w, "val": float(s * 10 + c + w)}
        for s in range(2)
        for c in range(2)
        for w in range(2)
    ]
    df = pd.DataFrame(rows)
    out = pb.cousineau(df, "sub", "cond", "val", within_cols=["wc"])

    r0 = out[(out["sub"] == 0) & (out["cond"] == 0) & (out["wc"] == 0)].iloc[0]
    assert r0["M"] == pytest.approx(5.5)
    assert r0["M_S"] == pytest.approx(0.5)
    assert r0["adj_val"] == pytest.approx(5.0)

    # adj is val + (within-group grand mean) - (within-subject mean) throughout
    expected_adj = df["val"].values + out["M"].values - out["M_S"].values
    assert np.allclose(out["adj_val"].values, expected_adj)


# ===========================================================================
# 2. MATHEMATICAL PROPERTIES
# ===========================================================================

@pytest.mark.property
def test_cousineau_removes_between_subject_variance():
    """Cousineau normalization: the per-subject mean of adj_<v> is constant
    across subjects (between-subject variance removed)."""
    rng = np.random.default_rng(7)
    n, k = 6, 3
    # add large per-subject offsets so between-subject variance dominates
    dat = rng.normal(0, 1, (n, k)) + (np.arange(n) * 100)[:, None]
    df = _long_from_matrix(dat)
    out = pb.cousineau(df, "sub", "cond", "val")

    per_sub = out.groupby("sub")["adj_val"].mean().values
    assert np.allclose(per_sub, per_sub[0])
    # and it equals the grand mean
    assert per_sub[0] == pytest.approx(df["val"].mean())


@pytest.mark.property
def test_cmo_preserves_grand_mean():
    """The Cousineau-Morey-O'Brien adjustment preserves the grand mean."""
    rng = np.random.default_rng(11)
    n, k = 5, 4
    dat = rng.normal(3, 2, (n, k)) + np.arange(k)
    df = _long_from_matrix(dat)
    out = pb.cousineau(df, "sub", "cond", "val")
    assert out["cmo_adj_val"].mean() == pytest.approx(df["val"].mean())


@pytest.mark.property
def test_equal_variance_basic_properties():
    """loftus_masson_equal_variance_kahana: non-negative, all entries equal,
    length == n_conditions."""
    rng = np.random.default_rng(3)
    for n, k in [(4, 2), (6, 3), (10, 5)]:
        dat = rng.normal(0, 1, (n, k))
        ci = pb.loftus_masson_equal_variance_kahana(dat)
        assert ci.shape == (k,)
        assert np.all(ci >= 0)
        assert np.allclose(ci, ci[0])


@pytest.mark.property
def test_equal_variance_shrinks_with_more_subjects():
    """Adding subjects (replicating the design) shrinks the CI.

    NOTE: the rubric's '~1/sqrt(2)' expectation does NOT hold exactly for this
    estimator, and we deliberately do NOT assert it. Replicating the rows keeps
    SS_int proportional but df_int = (n-1)(k-1) more-than-doubles when n -> 2n,
    so both MS_int and the t-criterion change. The robust, true property is that
    the CI strictly shrinks. (Measured ratio ~0.378, not 0.707, for the 3x2
    fixture -- reported as a methodological note, not a code bug.)"""
    dat = DAT_3x2
    ci_n = pb.loftus_masson_equal_variance_kahana(dat)[0]
    ci_2n = pb.loftus_masson_equal_variance_kahana(np.vstack([dat, dat]))[0]
    ci_3n = pb.loftus_masson_equal_variance_kahana(np.vstack([dat, dat, dat]))[0]

    assert ci_2n < ci_n
    assert ci_3n < ci_2n
    assert ci_3n > 0

    # The *SE* part alone (sqrt(MS_int/n)) is what the 1/sqrt(n) intuition refers
    # to; pin that it too fails to scale as 1/sqrt(2) here because MS_int is not
    # replication-invariant (df grows).
    assert not np.isclose(ci_2n / ci_n, 1 / math.sqrt(2), atol=0.05)


@pytest.mark.property
def test_analytic_shape_and_signs():
    """loftus_masson_analytic returns one row per condition; CI_equal is
    constant across conditions; CI_unequal >= 0 on well-behaved data."""
    rng = np.random.default_rng(5)
    n, k = 12, 4
    dat = rng.normal(5, 1.5, (n, k)) + np.arange(k)
    df = _long_from_matrix(dat)
    out = pb.loftus_masson_analytic(df, "sub", "cond", "val")

    assert len(out) == k
    assert out["cond"].nunique() == k
    assert np.allclose(out["CI_equal"].values, out["CI_equal"].iloc[0])
    assert (out["CI_unequal"] >= 0).all()
    assert (out["CI_equal"] >= 0).all()


# ===========================================================================
# 3. EDGE CASES
# ===========================================================================

@pytest.mark.edge_case
@pytest.mark.pins_bug
def test_cousineau_single_condition_raises_zerodivision():
    """EDGE/BUG: with a single condition (K=1) the CMO factor is
    sqrt(K/(K-1)) = sqrt(1/0). Because n_conds is a Python int from
    DataFrame.nunique(), '1/0' is a *Python* division and raises
    ZeroDivisionError (not inf/nan). Pinned as current behavior.

    Source: pybeh_pd.py:818-819 (cousineau).
    """
    df = pd.DataFrame(
        [{"sub": s, "cond": 0, "val": float(s)} for s in range(3)]
    )
    with pytest.raises(ZeroDivisionError):
        pb.cousineau(df, "sub", "cond", "val")


@pytest.mark.edge_case
def test_cousineau_single_subject():
    """Single subject: M_S equals each value within the (only) subject, so the
    Cousineau adj reduces to the condition value (adj = v + grand - subj_mean,
    with subj_mean == grand here only if balanced)."""
    df = pd.DataFrame(
        [{"sub": 0, "cond": c, "val": float(c + 1)} for c in range(3)]
    )
    out = pb.cousineau(df, "sub", "cond", "val")
    grand = df["val"].mean()           # = 2.0
    # one subject => M_S == grand mean for every row
    assert np.allclose(out["M_S"].values, grand)
    # therefore adj == original value
    assert np.allclose(out["adj_val"].values, df["val"].values)


@pytest.mark.edge_case
def test_equal_variance_single_subject_is_nan():
    """Single subject (1 x k): df_int = (1*k-1)-(1-1)-(k-1) = 0, so MS_int and
    the CI are undefined. Pin current behavior: returns NaNs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ci = pb.loftus_masson_equal_variance_kahana(np.array([[1.0, 2.0]]))
    assert ci.shape == (2,)
    assert np.all(np.isnan(ci))


@pytest.mark.edge_case
@pytest.mark.pins_bug
def test_unequal_variance_kahana_raises_nameerror():
    """KNOWN BUG B1: loftus_masson_unequal_variance_kahana's first statement is
    `dat = mat`, referencing an undefined global `mat`, so EVERY call raises
    NameError before any computation. Pinned.

    Source: pybeh_pd.py:852 (`dat = mat`).
    """
    for d in (DAT_3x2, np.ones((4, 3)), np.array([[1.0, 2.0, 3.0]])):
        with pytest.raises(NameError):
            pb.loftus_masson_unequal_variance_kahana(d)
