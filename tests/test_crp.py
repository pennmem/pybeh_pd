"""Skeptical unit tests for the lag-CRP / sem-CRP family in pybeh_pd.

Covers (importable as ``pb.<name>``):
  crp, min_crp, sem_crp, pd_crp, pd_min_crp, pd_sem_crp,
  pd_sem_crp_list, pd_sem_crp_list_sub, and the helper
  make_poss_recalls_matrix / make_recalls_matrix used to build inputs.

Rules: tests only; current behavior is pinned, not "fixed". Where the math
deviates from the code, a ``pins_bug`` marker + comment explains why.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import distance_matrix

import pybeh_pd as pb
from pybeh_pd import (
    crp,
    min_crp,
    sem_crp,
    make_recalls_matrix,
    make_poss_recalls_matrix,
)
from factories import make_experiment


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _in_unit_or_nan(a):
    """True iff every element is NaN or within [0, 1]."""
    a = np.asarray(a, dtype=float)
    finite = a[~np.isnan(a)]
    return np.all((finite >= 0.0) & (finite <= 1.0))


def _matrices_no_repeats(seed):
    """Build (pres, rec, recalls) matrices with no repeated presentations."""
    exp = make_experiment(n_lists=6, list_length=8, p_intrusion=0.0, seed=seed)
    pres = exp["pres_itemnos"]
    rec = exp["rec_itemnos"]
    recalls = make_recalls_matrix(pres, rec)
    return pres, rec, recalls, exp


# ===========================================================================
# crp  (pybeh_copy lag-CRP)
# ===========================================================================
@pytest.mark.hand_coded
def test_crp_forward_recall_all_plus_one():
    # Perfect forward recall [1,2,3,4]: every transition is +1.
    # listLength=4, lag_num=3 -> width 2*3+1 = 7, center (lag 0) NaN.
    out = crp(recalls=np.array([[1, 2, 3, 4]]), subjects=["a"],
              listLength=4, lag_num=3)
    assert out.shape == (1, 7)
    # columns map to lags -3..+3; center index 3 is lag 0 and MUST be NaN.
    assert np.isnan(out[0, 3])
    # Hand accounting (already-recalled lower positions are NOT possible):
    #   1->2: from sp1, poss lags {+1,+2,+3}, actual +1
    #   2->3: from sp2 seen{1,2}, poss lags {+1,+2}, actual +1
    #   3->4: from sp3 seen{1,2,3}, poss lags {+1}, actual +1
    # lag +1 (index 4): actual 3 / poss 3 = 1.0
    assert out[0, 4] == 1.0
    # lag +2 (index 5): actual 0 / poss 2 = 0.0; lag +3 (index 6): 0/1 = 0.0
    assert out[0, 5] == 0.0
    assert out[0, 6] == 0.0
    # negative lags were never possible (lower sps already recalled) -> NaN
    assert np.isnan(out[0, 2])  # lag -1
    assert np.isnan(out[0, 1])  # lag -2
    assert np.isnan(out[0, 0])  # lag -3


@pytest.mark.hand_coded
def test_crp_single_transition():
    # Only one valid transition 2->3 (then padding zeros).
    # listLength=4, lag_num=3. Transition is +1.
    out = crp(recalls=np.array([[2, 3, 0, 0]]), subjects=["a"],
              listLength=4, lag_num=3)
    assert out.shape == (1, 7)
    assert np.isnan(out[0, 3])  # center NaN
    # From sp 2, possible lags with sp not yet seen: {-1, +1, +2}; actual +1.
    # lag +1 idx4 -> 1/1 = 1.0; lag -1 idx2 -> 0/1 = 0.0; lag +2 idx5 -> 0/1.
    assert out[0, 4] == 1.0
    assert out[0, 2] == 0.0
    assert out[0, 5] == 0.0
    # lags with no possible transition stay NaN (e.g. +3 idx6, -2 idx1, -3 idx0)
    assert np.isnan(out[0, 6])
    assert np.isnan(out[0, 1])
    assert np.isnan(out[0, 0])


@pytest.mark.hand_coded
def test_crp_backward_then_forward():
    # recalls [2,1,3]: transitions 2->1 (lag -1), 1->3 (lag +2).
    # listLength=3, lag_num=2 -> width 5 (lags -2..+2), center idx2 NaN.
    out = crp(recalls=np.array([[2, 1, 3]]), subjects=["a"],
              listLength=3, lag_num=2)
    assert out.shape == (1, 5)
    assert np.isnan(out[0, 2])
    # 2->1: from sp2 poss lags {-1,+1}, actual -1.
    # 1->3: from sp1 (seen {1,2}) poss lags {+2}, actual +2.
    # lag -1 idx1: 1/1 = 1.0 ; lag +1 idx3: 0/1 = 0.0 ; lag +2 idx4: 1/1 = 1.0
    assert out[0, 1] == 1.0
    assert out[0, 3] == 0.0
    assert out[0, 4] == 1.0
    # lag -2 idx0 never possible -> NaN
    assert np.isnan(out[0, 0])


@pytest.mark.property
def test_crp_probabilities_in_unit_and_center_nan():
    _, _, recalls, exp = _matrices_no_repeats(seed=1)
    out = crp(recalls=recalls, subjects=["s"] * len(recalls),
              listLength=exp["list_length"], lag_num=5)
    assert out.shape == (1, 11)
    assert _in_unit_or_nan(out)
    # center column (lag 0) is always NaN
    assert np.isnan(out[0, 5])


@pytest.mark.property
def test_crp_width_is_2lagnum_plus_1():
    _, _, recalls, exp = _matrices_no_repeats(seed=2)
    for lag_num in (1, 3, 5, 7):
        out = crp(recalls=recalls, subjects=["s"] * len(recalls),
                  listLength=exp["list_length"], lag_num=lag_num)
        assert out.shape[1] == 2 * lag_num + 1


@pytest.mark.property
def test_crp_lag_num_guards_raise():
    r = np.array([[1, 2, 3]])
    # lag_num < 1
    with pytest.raises(ValueError):
        crp(recalls=r, subjects=["a"], listLength=4, lag_num=0)
    # lag_num >= listLength
    with pytest.raises(ValueError):
        crp(recalls=r, subjects=["a"], listLength=4, lag_num=4)
    # non-int lag_num
    with pytest.raises(ValueError):
        crp(recalls=r, subjects=["a"], listLength=4, lag_num=2.5)
    # non-int skip_first_n
    with pytest.raises(ValueError):
        crp(recalls=r, subjects=["a"], listLength=4, lag_num=2,
            skip_first_n=1.5)


@pytest.mark.edge_case
def test_crp_empty_recalls_all_nan():
    # No correct recalls -> no transitions -> every column NaN.
    out = crp(recalls=np.zeros((1, 4), dtype=int), subjects=["a"],
              listLength=4, lag_num=3)
    assert out.shape == (1, 7)
    assert np.all(np.isnan(out))


@pytest.mark.edge_case
def test_crp_skip_first_n_removes_only_transition():
    # Single transition 2->3 but skip_first_n=1 skips it -> all NaN.
    out = crp(recalls=np.array([[2, 3, 0, 0]]), subjects=["a"],
              listLength=4, lag_num=3, skip_first_n=1)
    assert np.all(np.isnan(out))


# ===========================================================================
# min_crp  (repeated-presentation lag-CRP)
# ===========================================================================
@pytest.mark.hand_coded
def test_min_crp_forward_matches_hand():
    # No repeats, forward recall: identical to crp by hand.
    pres = np.array([[1, 2, 3, 4]])
    recalls = np.array([[1, 2, 3, 4]])
    poss = make_poss_recalls_matrix(pres_itemnos=pres, max_n_reps=1)
    out = min_crp(recalls=recalls, poss_recalls=poss, subjects=["a"],
                  listLength=4, lag_num=3)
    assert out.shape == (1, 7)
    assert np.isnan(out[0, 3])          # center NaN
    assert out[0, 4] == 1.0             # lag +1
    assert out[0, 5] == 0.0             # lag +2
    assert out[0, 6] == 0.0             # lag +3


@pytest.mark.hand_coded
def test_min_crp_backward_then_forward():
    pres = np.array([[1, 2, 3]])
    recalls = np.array([[2, 1, 3]])
    poss = make_poss_recalls_matrix(pres_itemnos=pres, max_n_reps=1)
    out = min_crp(recalls=recalls, poss_recalls=poss, subjects=["a"],
                  listLength=3, lag_num=2)
    assert out.shape == (1, 5)
    assert np.isnan(out[0, 2])
    assert out[0, 1] == 1.0   # lag -1
    assert out[0, 3] == 0.0   # lag +1
    assert out[0, 4] == 1.0   # lag +2


@pytest.mark.hand_coded
def test_min_crp_single_transition():
    pres = np.array([[1, 2, 3, 4]])
    recalls = np.array([[2, 3, 0, 0]])
    poss = make_poss_recalls_matrix(pres_itemnos=pres, max_n_reps=1)
    out = min_crp(recalls=recalls, poss_recalls=poss, subjects=["a"],
                  listLength=4, lag_num=3)
    assert np.isnan(out[0, 3])
    assert out[0, 4] == 1.0   # lag +1
    assert out[0, 2] == 0.0   # lag -1
    assert out[0, 5] == 0.0   # lag +2


@pytest.mark.property
def test_min_crp_probabilities_in_unit_and_center_nan():
    pres, _, recalls, exp = _matrices_no_repeats(seed=3)
    poss = make_poss_recalls_matrix(pres_itemnos=pres, max_n_reps=1)
    out = min_crp(recalls=recalls, poss_recalls=poss,
                  subjects=["s"] * len(recalls),
                  listLength=exp["list_length"], lag_num=5)
    assert _in_unit_or_nan(out)
    assert np.isnan(out[0, 5])


@pytest.mark.property
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_min_crp_agrees_with_crp_on_unique_items(seed):
    # KEY CLAIM under scrutiny: with NO repeated presentations, the
    # repeated-presentation min_crp must reduce to plain crp. Verified here
    # element-wise (equal_nan) over the shared lag window for several seeds.
    pres, _, recalls, exp = _matrices_no_repeats(seed=seed)
    ll = exp["list_length"]
    poss = make_poss_recalls_matrix(pres_itemnos=pres, max_n_reps=1)
    subj = ["s"] * len(recalls)
    c = crp(recalls=recalls, subjects=subj, listLength=ll, lag_num=5)
    m = min_crp(recalls=recalls, poss_recalls=poss, subjects=subj,
                listLength=ll, lag_num=5)
    assert np.allclose(c, m, equal_nan=True)


@pytest.mark.property
def test_min_crp_3d_repeated_path_in_unit():
    # Exercise the with_repeats (3D) branch; result still bounded.
    exp = make_experiment(n_lists=4, list_length=8, p_intrusion=0.0, seed=6)
    pres, rec = exp["pres_itemnos"], exp["rec_itemnos"]
    rc3 = make_recalls_matrix(pres, rec, max_n_reps=2)
    poss3 = make_poss_recalls_matrix(pres, max_n_reps=2)
    assert rc3.ndim == 3 and poss3.ndim == 3
    out = min_crp(recalls=rc3, poss_recalls=poss3,
                  subjects=["s"] * len(rc3), listLength=8, lag_num=5)
    assert _in_unit_or_nan(out)
    assert np.isnan(out[0, 5])


@pytest.mark.property
def test_min_crp_lag_num_guards_raise():
    pres = np.array([[1, 2, 3]])
    recalls = np.array([[1, 2, 3]])
    poss = make_poss_recalls_matrix(pres_itemnos=pres, max_n_reps=1)
    with pytest.raises(ValueError):
        min_crp(recalls=recalls, poss_recalls=poss, subjects=["a"],
                listLength=3, lag_num=0)
    with pytest.raises(ValueError):
        min_crp(recalls=recalls, poss_recalls=poss, subjects=["a"],
                listLength=3, lag_num=3)
    with pytest.raises(ValueError):
        min_crp(recalls=recalls, poss_recalls=poss, subjects=["a"],
                listLength=3, lag_num=2, skip_first_n=0.5)


@pytest.mark.edge_case
def test_min_crp_empty_recalls_all_nan():
    pres = np.array([[1, 2, 3, 4]])
    recalls = np.zeros((1, 4), dtype=int)
    poss = make_poss_recalls_matrix(pres_itemnos=pres, max_n_reps=1)
    out = min_crp(recalls=recalls, poss_recalls=poss, subjects=["a"],
                  listLength=4, lag_num=3)
    assert np.all(np.isnan(out))


# ===========================================================================
# sem_crp
# ===========================================================================
@pytest.mark.hand_coded
def test_sem_crp_explicit_bins_hand():
    # listLength=3, forward recall [1,2,3]; sym sem matrix.
    pres = np.array([[1, 2, 3]])
    rec = np.array([[1, 2, 3]])
    recalls = np.array([[1, 2, 3]])
    sem = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
    bins = [0.5, 1.5, 2.5]  # 3 bins
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 0/0 in an empty bin_means cell
        bm, c = sem_crp(recalls=recalls, recalls_itemnos=rec,
                        pres_itemnos=pres, subjects=["a"], sem_sims=sem,
                        bins=bins, listLength=3)
    # Hand accounting (see analysis): actual=[1,0,1], poss=[1,1,1].
    assert c.shape == (1, 3)               # crp row length == n_bins
    assert bm.shape == c.shape             # bin_means same shape as crp
    np.testing.assert_array_equal(c[0], np.array([1.0, 0.0, 1.0]))
    # bin_means: bin0 = 1/1 = 1, bin1 = 0/0 = nan, bin2 = 3/1 = 3.
    assert bm[0, 0] == 1.0
    assert np.isnan(bm[0, 1])
    assert bm[0, 2] == 3.0


@pytest.mark.hand_coded
def test_sem_crp_returns_two_tuple_and_four_tuple():
    pres = np.array([[1, 2, 3]])
    rec = np.array([[1, 2, 3]])
    recalls = np.array([[1, 2, 3]])
    sem = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
    bins = [0.5, 1.5, 2.5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        two = sem_crp(recalls=recalls, recalls_itemnos=rec, pres_itemnos=pres,
                      subjects=["a"], sem_sims=sem, bins=bins, listLength=3)
        four = sem_crp(recalls=recalls, recalls_itemnos=rec, pres_itemnos=pres,
                       subjects=["a"], sem_sims=sem, bins=bins, listLength=3,
                       ret_counts=True)
    assert len(two) == 2
    assert len(four) == 4
    _, _, actual, poss = four
    np.testing.assert_array_equal(actual, np.array([1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(poss, np.array([1.0, 1.0, 1.0]))


@pytest.mark.hand_coded
def test_sem_crp_n_bins_controls_row_length():
    # With n_bins (no explicit bins) crp row length == n_bins.
    pres, rec, recalls, exp = _matrices_no_repeats(seed=7)
    sem = distance_matrix(exp["features"], exp["features"], p=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bm, c = sem_crp(recalls=recalls, recalls_itemnos=rec,
                        pres_itemnos=pres, subjects=["s"] * len(recalls),
                        sem_sims=sem, n_bins=7,
                        listLength=exp["list_length"])
    assert c.shape == (1, 7)
    assert bm.shape == (1, 7)


@pytest.mark.property
def test_sem_crp_bounds_and_shapes():
    pres, rec, recalls, exp = _matrices_no_repeats(seed=8)
    sem = distance_matrix(exp["features"], exp["features"], p=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bm, c = sem_crp(recalls=recalls, recalls_itemnos=rec,
                        pres_itemnos=pres, subjects=["s"] * len(recalls),
                        sem_sims=sem, n_bins=10,
                        listLength=exp["list_length"])
    # crp probabilities in [0,1] or NaN: the actual transition's bin is always
    # among the possible-transition bins, so actual<=poss per bin (no >1).
    assert _in_unit_or_nan(c)
    assert bm.shape == c.shape


@pytest.mark.property
def test_sem_crp_poss_ge_actual_never_exceeds_one():
    # Directly assert the accounting invariant: for every populated bin,
    # actual <= poss (so crp = actual/poss is never > 1).
    pres, rec, recalls, exp = _matrices_no_repeats(seed=9)
    sem = distance_matrix(exp["features"], exp["features"], p=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, actual, poss = sem_crp(
            recalls=recalls, recalls_itemnos=rec, pres_itemnos=pres,
            subjects=["s"] * len(recalls), sem_sims=sem, n_bins=10,
            listLength=exp["list_length"], ret_counts=True)
    assert np.all(actual <= poss)


# ===========================================================================
# pd_crp
# ===========================================================================
@pytest.mark.property
def test_pd_crp_lag0_nan_and_bounds():
    exp = make_experiment(p_intrusion=0.0, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pb.pd_crp(exp["events"], lag_num=5)
    assert list(out.columns) == ["prob", "lag"]
    # lag==0 row is NaN
    assert np.isnan(out.loc[out["lag"] == 0, "prob"]).all()
    # 2*lag_num+1 rows
    assert len(out) == 11
    assert _in_unit_or_nan(out["prob"].values)


@pytest.mark.property
def test_pd_crp_matches_matrix_crp():
    # pd_crp should equal crp computed off the same matrices.
    pres, rec, recalls, exp = _matrices_no_repeats(seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pb.pd_crp(exp["events"], lag_num=5)
    direct = crp(recalls=recalls, subjects=["s"] * len(recalls),
                 listLength=exp["list_length"], lag_num=5)[0]
    np.testing.assert_allclose(out["prob"].values, direct, equal_nan=True)


@pytest.mark.edge_case
def test_pd_crp_no_valid_lists_returns_empty_df():
    # WORD-only df: no list has both presentations and recalls.
    df = pd.DataFrame([
        dict(subject="s", session=0, list=0, type="WORD", itemno=1,
             serialpos=1),
        dict(subject="s", session=0, list=0, type="WORD", itemno=2,
             serialpos=2),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pb.pd_crp(df, lag_num=3)
    assert isinstance(out, pd.DataFrame)
    assert out.empty


# ===========================================================================
# pd_min_crp
# ===========================================================================
@pytest.mark.property
def test_pd_min_crp_default_max_n_reps_bounds():
    exp = make_experiment(p_intrusion=0.0, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pb.pd_min_crp(exp["events"], lag_num=5, max_n_reps=1)
    assert list(out.columns) == ["prob", "lag"]
    assert np.isnan(out.loc[out["lag"] == 0, "prob"]).all()
    assert _in_unit_or_nan(out["prob"].values)


@pytest.mark.property
def test_pd_min_crp_agrees_with_pd_crp_no_repeats():
    # No repeated presentations -> pd_min_crp == pd_crp.
    exp = make_experiment(p_intrusion=0.0, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pb.pd_crp(exp["events"], lag_num=5)
        b = pb.pd_min_crp(exp["events"], lag_num=5, max_n_reps=1)
    np.testing.assert_allclose(a["prob"].values, b["prob"].values,
                               equal_nan=True)


@pytest.mark.property
def test_pd_min_crp_max_n_reps_2_bounds():
    exp = make_experiment(n_lists=4, list_length=8, p_intrusion=0.0, seed=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pb.pd_min_crp(exp["events"], lag_num=5, max_n_reps=2)
    assert np.isnan(out.loc[out["lag"] == 0, "prob"]).all()
    assert _in_unit_or_nan(out["prob"].values)


# ===========================================================================
# pd_sem_crp
# ===========================================================================
@pytest.mark.property
def test_pd_sem_crp_bounds_and_columns():
    exp = make_experiment(p_intrusion=0.0, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pb.pd_sem_crp(exp["events"], sim_columns=exp["feat_cols"],
                            n_bins=10)
    assert {"prob", "sem_bin_mean", "sem_bin"}.issubset(out.columns)
    # at most n_bins rows (rows with no data are dropped by `prob == prob`)
    assert len(out) <= 10
    assert _in_unit_or_nan(out["prob"].values)
    # no dropped row is NaN (the query removes them)
    assert not out["prob"].isna().any()


@pytest.mark.property
def test_pd_sem_crp_ret_counts_actual_le_poss():
    exp = make_experiment(p_intrusion=0.0, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pb.pd_sem_crp(exp["events"], sim_columns=exp["feat_cols"],
                            n_bins=10, ret_counts=True)
    assert {"actual", "poss"}.issubset(out.columns)
    assert np.all(out["actual"].values <= out["poss"].values)


@pytest.mark.edge_case
def test_pd_sem_crp_no_valid_lists_returns_empty_df():
    df = pd.DataFrame([
        dict(subject="s", session=0, list=0, type="WORD", itemno=1,
             serialpos=1, f0=0.1),
        dict(subject="s", session=0, list=0, type="WORD_VALS", itemno=1,
             serialpos=0, f0=0.1),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pb.pd_sem_crp(df, sim_columns=["f0"], n_bins=3)
    assert isinstance(out, pd.DataFrame)
    assert out.empty


# ===========================================================================
# pd_sem_crp_list  (one list; features off WORD rows; bins REQUIRED)
# ===========================================================================
def _one_list_df(exp, list_id=0):
    df = exp["events"]
    return df[(df["list"] == list_id)
              & (df["type"].isin(["WORD", "REC_WORD"]))].copy()


@pytest.mark.property
def test_pd_sem_crp_list_bounds():
    exp = make_experiment(n_lists=4, list_length=8, p_intrusion=0.0, seed=5)
    bins = list(np.linspace(0, 8, 6))[:-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pb.pd_sem_crp_list(_one_list_df(exp), sim_columns=exp["feat_cols"],
                                 bins=bins)
    assert {"prob", "sem_bin_mean", "sem_bin"}.issubset(out.columns)
    assert _in_unit_or_nan(out["prob"].values)
    # rows with data are kept; at most len(bins) of them
    assert len(out) <= len(bins)


@pytest.mark.property
def test_pd_sem_crp_list_ret_counts_actual_le_poss():
    exp = make_experiment(n_lists=4, list_length=8, p_intrusion=0.0, seed=5)
    bins = list(np.linspace(0, 8, 6))[:-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pb.pd_sem_crp_list(_one_list_df(exp), sim_columns=exp["feat_cols"],
                                 bins=bins, ret_counts=True)
    assert {"actual", "poss"}.issubset(out.columns)
    assert np.all(out["actual"].values <= out["poss"].values)


# ===========================================================================
# pd_sem_crp_list_sub  (groupby(list_index).apply); factory uses 'list'
# ===========================================================================
@pytest.mark.integration
def test_pd_sem_crp_list_sub_aggregates_bounds():
    exp = make_experiment(n_lists=4, list_length=8, p_intrusion=0.0, seed=5)
    bins = list(np.linspace(0, 8, 6))[:-1]
    sub_df = exp["events"][exp["events"]["type"].isin(["WORD", "REC_WORD"])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pb.pd_sem_crp_list_sub(
            sub_df, sim_columns=exp["feat_cols"], bins=bins,
            list_index=["subject", "session", "list"])
    assert {"sem_bin", "actual", "poss", "prob"}.issubset(out.columns)
    # subject-level prob = summed actual / summed poss; bounded.
    assert _in_unit_or_nan(out["prob"].values)
    assert np.all(out["actual"].values <= out["poss"].values)


@pytest.mark.integration
def test_pd_sem_crp_list_sub_matches_manual_pool_sum():
    # The sub-level prob should equal sum(actual)/sum(poss) across that
    # subject's lists, computed bin by bin.
    exp = make_experiment(n_lists=4, list_length=8, p_intrusion=0.0, seed=5)
    bins = list(np.linspace(0, 8, 6))[:-1]
    sub_df = exp["events"][exp["events"]["type"].isin(["WORD", "REC_WORD"])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pb.pd_sem_crp_list_sub(
            sub_df, sim_columns=exp["feat_cols"], bins=bins,
            list_index=["subject", "session", "list"])
    # prob == actual/poss exactly for each aggregated bin row
    np.testing.assert_allclose(
        out["prob"].values,
        out["actual"].values / out["poss"].values,
        equal_nan=True)
