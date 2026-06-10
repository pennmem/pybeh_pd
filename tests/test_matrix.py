"""Skeptical unit tests for the matrix-level helpers in ``pybeh_pd``.

Functions under test (all ``pb.<name>``):
    get_itemno_matrices, get_all_matrices, make_recalls_matrix,
    make_poss_recalls_matrix, get_min_trans, get_sim_mat,
    make_clean_recalls_mask2d (from pybeh_copy, re-exported on pb).

Conventions tested:
    pres/rec itemnos are positive ints; rec rows 0-padded; extra-list
    intrusions become -1 in the recalls matrix; correct recalls become the
    1-based serial position of the item in its presentation row.

Markers: hand_coded, property, edge_case, pins_bug (see pytest.ini).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pybeh_pd as pb
from factories import make_experiment


# ---------------------------------------------------------------------------
# make_recalls_matrix
# ---------------------------------------------------------------------------

@pytest.mark.hand_coded
def test_make_recalls_matrix_basic_serialpos():
    # pres item k is at serial position k+1; recalled items map to those sps,
    # padding 0 stays 0.
    pres = np.array([[1, 2, 3, 4]])
    rec = np.array([[3, 1, 4, 0]])
    out = pb.make_recalls_matrix(pres, rec)
    assert out.shape == (1, 4)
    np.testing.assert_array_equal(out, np.array([[3, 1, 4, 0]]))


@pytest.mark.hand_coded
def test_make_recalls_matrix_extra_list_intrusion_maps_to_minus_one():
    # itemno 5 is not in the presentation row -> -1.
    pres = np.array([[1, 2, 3, 4]])
    rec = np.array([[5, 2, 0, 0]])
    out = pb.make_recalls_matrix(pres, rec)
    np.testing.assert_array_equal(out, np.array([[-1, 2, 0, 0]]))


@pytest.mark.hand_coded
def test_make_recalls_matrix_zero_in_middle_stays_zero():
    # A 0 anywhere in the recall row is "no recall" -> stays 0, surrounding
    # correct recalls still map to serial positions.
    pres = np.array([[10, 20, 30]])
    rec = np.array([[20, 0, 30]])
    out = pb.make_recalls_matrix(pres, rec)
    np.testing.assert_array_equal(out, np.array([[2, 0, 3]]))


@pytest.mark.property
def test_make_recalls_matrix_correct_entries_are_valid_serialpos():
    exp = make_experiment(seed=11)
    pres = exp["pres_itemnos"]
    rec = exp["rec_itemnos"]
    L = exp["list_length"]
    out = pb.make_recalls_matrix(pres, rec)
    assert out.shape == rec.shape
    for t in range(out.shape[0]):
        for r in range(out.shape[1]):
            v = out[t, r]
            if v > 0:
                # valid serial position in 1..L
                assert 1 <= v <= L
                # and the item at that position equals the recalled itemno
                assert pres[t, v - 1] == rec[t, r]
            elif v == 0:
                # padding only where the recall itemno was 0 padding
                assert rec[t, r] == 0
            else:
                # intrusion: -1, and the recalled itemno is genuinely absent
                assert v == -1
                assert rec[t, r] not in set(pres[t].tolist())


@pytest.mark.property
def test_make_recalls_matrix_max_n_reps_2_returns_3d():
    # Item 5 presented at positions 1 and 3 (doubled). With max_n_reps=2 the
    # output is 3D (trials, recalls, 2) and a correct recall of 5 records both
    # serial positions [1, 3].
    pres = np.array([[5, 6, 5, 7]])
    rec = np.array([[5, 7, 0]])
    out = pb.make_recalls_matrix(pres, rec, max_n_reps=2)
    assert out.shape == (1, 3, 2)
    np.testing.assert_array_equal(out[0, 0], np.array([1, 3]))  # both sps of "5"
    np.testing.assert_array_equal(out[0, 1], np.array([4, 0]))  # "7" single sp
    np.testing.assert_array_equal(out[0, 2], np.array([0, 0]))  # padding


@pytest.mark.edge_case
def test_make_recalls_matrix_doubled_item_raises_with_default_max_n_reps():
    # Item 1 presented twice; with max_n_reps=1 recalling it finds 2 serial
    # positions -> Exception.
    pres = np.array([[1, 1, 2]])
    rec = np.array([[1, 0, 0]])
    with pytest.raises(Exception, match="more than max_n_reps"):
        pb.make_recalls_matrix(pres, rec, max_n_reps=1)


@pytest.mark.edge_case
def test_make_recalls_matrix_empty_recall_row_all_zero():
    pres = np.array([[1, 2, 3]])
    rec = np.array([[0, 0, 0]])
    out = pb.make_recalls_matrix(pres, rec)
    np.testing.assert_array_equal(out, np.zeros((1, 3), dtype=int))


@pytest.mark.edge_case
def test_make_recalls_matrix_all_intrusions():
    pres = np.array([[1, 2, 3]])
    rec = np.array([[7, 8, 9]])  # none presented
    out = pb.make_recalls_matrix(pres, rec)
    np.testing.assert_array_equal(out, np.array([[-1, -1, -1]]))


@pytest.mark.edge_case
def test_make_recalls_matrix_single_item_list():
    pres = np.array([[42]])
    rec = np.array([[42, 0]])
    out = pb.make_recalls_matrix(pres, rec)
    np.testing.assert_array_equal(out, np.array([[1, 0]]))


@pytest.mark.edge_case
def test_make_recalls_matrix_rec_longer_than_pres():
    # rec row may have more columns than the list length; trailing zeros pad.
    pres = np.array([[1, 2]])
    rec = np.array([[2, 1, 0, 0, 0]])
    out = pb.make_recalls_matrix(pres, rec)
    np.testing.assert_array_equal(out, np.array([[2, 1, 0, 0, 0]]))


@pytest.mark.pins_bug
def test_make_recalls_matrix_nan_padding_misclassified_as_intrusion():
    # BUG (B-NEW, pybeh_pd.py:74): the docstring says recall rows "may be padded
    # with zeros or NaNs", but the zero/NaN guard is written
    #   (rec_itemnos[t,r]) == 0 | (np.isnan(rec_itemnos[t,r]))
    # which, due to Python operator precedence (| binds tighter than ==),
    # evaluates as  rec == (0 | isnan)  instead of  (rec == 0) | isnan.
    # For a NaN entry: 0 | True == 1, and nan == 1 is False, so the NaN falls
    # through to the else branch and is recorded as an INTRUSION (-1) rather
    # than padding (0).
    #
    # CORRECT behavior would be 0 (padding). We pin the current buggy -1.
    pres = np.array([[1.0, 2.0, 3.0]])
    rec = np.array([[2.0, np.nan, 3.0]])
    out = pb.make_recalls_matrix(pres, rec)
    np.testing.assert_array_equal(out, np.array([[2, -1, 3]]))  # bug: nan -> -1


# ---------------------------------------------------------------------------
# make_poss_recalls_matrix
# ---------------------------------------------------------------------------

@pytest.mark.hand_coded
def test_make_poss_recalls_unique_items_identity():
    # On a unique-item list every item maps to its own serial position.
    pres = np.array([[10, 20, 30, 40]])
    out = pb.make_poss_recalls_matrix(pres)
    np.testing.assert_array_equal(out, np.array([[1, 2, 3, 4]]))


@pytest.mark.hand_coded
def test_make_poss_recalls_zero_padding_stays_zero():
    pres = np.array([[5, 6, 0]])
    out = pb.make_poss_recalls_matrix(pres)
    np.testing.assert_array_equal(out, np.array([[1, 2, 0]]))


@pytest.mark.hand_coded
def test_make_poss_recalls_repeated_item_max_n_reps_2():
    # Item 5 at positions 1 and 3 -> both rows for that item record [1, 3].
    pres = np.array([[5, 6, 5, 7]])
    out = pb.make_poss_recalls_matrix(pres, max_n_reps=2)
    assert out.shape == (1, 4, 2)
    np.testing.assert_array_equal(out[0, 0], np.array([1, 3]))
    np.testing.assert_array_equal(out[0, 1], np.array([2, 0]))
    np.testing.assert_array_equal(out[0, 2], np.array([1, 3]))
    np.testing.assert_array_equal(out[0, 3], np.array([4, 0]))


@pytest.mark.property
def test_make_poss_recalls_unique_lists_is_arange():
    exp = make_experiment(seed=7)
    pres = exp["pres_itemnos"]
    L = exp["list_length"]
    out = pb.make_poss_recalls_matrix(pres)
    assert out.shape == pres.shape
    expected = np.tile(np.arange(1, L + 1), (pres.shape[0], 1))
    np.testing.assert_array_equal(out, expected)


@pytest.mark.edge_case
def test_make_poss_recalls_doubled_item_raises_max_n_reps_1():
    pres = np.array([[3, 3, 4]])
    with pytest.raises(Exception, match="more than max_n_reps"):
        pb.make_poss_recalls_matrix(pres, max_n_reps=1)


# ---------------------------------------------------------------------------
# get_min_trans
# ---------------------------------------------------------------------------

@pytest.mark.hand_coded
def test_get_min_trans_positive_wins_tie():
    # candidates serialpos [1, 3], current recall rec=[2]:
    # diffs sp - r = {1-2=-1, 3-2=+1}. abs equal -> Howard&Kahana rule:
    # sort descending [+1, -1] then argmin(|.|) picks the first, +1.
    assert pb.get_min_trans([1, 3], [2]) == 1


@pytest.mark.hand_coded
def test_get_min_trans_zero_when_same_position():
    # rec position equals a candidate -> lag 0 (smallest abs).
    assert pb.get_min_trans([1, 2, 3], [2]) == 0


@pytest.mark.hand_coded
def test_get_min_trans_picks_smallest_abs_lag():
    # candidates [1, 10], rec=[3]: diffs {-2, +7} -> smallest abs is -2.
    assert pb.get_min_trans([1, 10], [3]) == -2


@pytest.mark.property
def test_get_min_trans_result_is_a_realizable_lag():
    rng = np.random.default_rng(0)
    for _ in range(50):
        sp = sorted(rng.choice(np.arange(1, 12), size=rng.integers(1, 5), replace=False).tolist())
        rec = [int(rng.integers(1, 12))]
        res = pb.get_min_trans(sp, rec)
        all_lags = [s - r for s in sp for r in rec]
        # result is one of the candidate lags and has minimal abs value
        assert res in all_lags
        assert abs(res) == min(abs(x) for x in all_lags)
        # tie-break: among minimal-abs lags, the most positive one is chosen
        min_abs = min(abs(x) for x in all_lags)
        tied = [x for x in all_lags if abs(x) == min_abs]
        assert res == max(tied)


# ---------------------------------------------------------------------------
# make_clean_recalls_mask2d  (from pybeh_copy)
# ---------------------------------------------------------------------------

@pytest.mark.hand_coded
def test_clean_mask_first_correct_only():
    # 1 only for the first occurrence of each positive value; repeats/0/neg -> 0.
    data = np.array([[3, 1, 1, 0, -1]])
    out = np.array(pb.make_clean_recalls_mask2d(data))
    np.testing.assert_array_equal(out, np.array([[1, 1, 0, 0, 0]]))


@pytest.mark.hand_coded
def test_clean_mask_repeat_then_new():
    data = np.array([[2, 2, -1, 3, 0]])
    out = np.array(pb.make_clean_recalls_mask2d(data))
    np.testing.assert_array_equal(out, np.array([[1, 0, 0, 1, 0]]))


@pytest.mark.hand_coded
def test_clean_mask_all_zero_and_all_negative():
    data = np.array([[0, 0, 0], [-1, -2, -1]])
    out = np.array(pb.make_clean_recalls_mask2d(data))
    np.testing.assert_array_equal(out, np.zeros((2, 3), dtype=int))


@pytest.mark.property
def test_clean_mask_is_binary_and_marks_first_occurrence():
    exp = make_experiment(seed=5)
    recalls = pb.make_recalls_matrix(exp["pres_itemnos"], exp["rec_itemnos"])
    out = np.array(pb.make_clean_recalls_mask2d(recalls))
    assert out.shape == recalls.shape
    # strictly 0/1
    assert set(np.unique(out).tolist()).issubset({0, 1})
    # mask == 1 exactly at first occurrence of each positive serial position
    for row_in, row_out in zip(recalls, out):
        seen: set[int] = set()
        for v_in, v_out in zip(row_in, row_out):
            expected = 1 if (v_in > 0 and v_in not in seen) else 0
            assert v_out == expected
            if v_in > 0:
                seen.add(int(v_in))


# ---------------------------------------------------------------------------
# get_sim_mat
# ---------------------------------------------------------------------------

@pytest.mark.hand_coded
def test_get_sim_mat_euclidean_hand_values():
    # Two WORD_VALS items at (0,0) and (3,4) -> distance 5, symmetric, 0 diag.
    df = pd.DataFrame({
        "type": ["WORD_VALS", "WORD_VALS"],
        "itemno": [1, 2],
        "f0": [0.0, 3.0],
        "f1": [0.0, 4.0],
    })
    sm = pb.get_sim_mat(df, ["f0", "f1"])
    assert sm.shape == (2, 2)
    np.testing.assert_allclose(sm, np.array([[0.0, 5.0], [5.0, 0.0]]))


@pytest.mark.hand_coded
def test_get_sim_mat_p1_manhattan():
    df = pd.DataFrame({
        "type": ["WORD_VALS", "WORD_VALS"],
        "itemno": [1, 2],
        "f0": [0.0, 3.0],
        "f1": [0.0, 4.0],
    })
    sm = pb.get_sim_mat(df, ["f0", "f1"], p=1)
    np.testing.assert_allclose(sm, np.array([[0.0, 7.0], [7.0, 0.0]]))


@pytest.mark.hand_coded
def test_get_sim_mat_sorts_by_itemno():
    # Rows given out of itemno order; output should be sorted so item 1 is row 0.
    df = pd.DataFrame({
        "type": ["WORD_VALS", "WORD_VALS"],
        "itemno": [2, 1],
        "f0": [10.0, 0.0],
        "f1": [0.0, 0.0],
    })
    sm = pb.get_sim_mat(df, ["f0", "f1"])
    # both off-diagonals equal the distance 10, diagonal 0
    np.testing.assert_allclose(sm, np.array([[0.0, 10.0], [10.0, 0.0]]))


@pytest.mark.property
def test_get_sim_mat_symmetric_zero_diag_shape():
    exp = make_experiment(seed=2)
    sm = pb.get_sim_mat(exp["events"], exp["feat_cols"])
    pool = exp["n_lists"] * exp["list_length"]
    assert sm.shape == (pool, pool)
    # Minkowski self-distance is 0 on the diagonal
    np.testing.assert_allclose(np.diag(sm), np.zeros(pool), atol=1e-12)
    # symmetric
    np.testing.assert_allclose(sm, sm.T, atol=1e-12)
    # non-negative
    assert (sm >= 0).all()


# ---------------------------------------------------------------------------
# get_itemno_matrices
# ---------------------------------------------------------------------------

@pytest.mark.hand_coded
def test_get_itemno_matrices_pivots_to_serialpos_order():
    df = pd.DataFrame({
        "subject": ["s0"] * 4,
        "session": [0] * 4,
        "list": [0, 0, 1, 1],
        "itemno": [11, 12, 21, 22],
    })
    out = pb.get_itemno_matrices(df)
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, np.array([[11, 12], [21, 22]]))


@pytest.mark.property
def test_get_itemno_matrices_padding_for_ragged_lists():
    # List 0 has 3 items, list 1 has 2 -> ragged; shorter row padded with 0.
    df = pd.DataFrame({
        "subject": ["s0"] * 5,
        "session": [0] * 5,
        "list": [0, 0, 0, 1, 1],
        "itemno": [1, 2, 3, 4, 5],
    })
    out = pb.get_itemno_matrices(df)
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, np.array([[1, 2, 3], [4, 5, 0]]))


@pytest.mark.property
def test_get_itemno_matrices_matches_factory_pres():
    exp = make_experiment(seed=9)
    word_df = exp["events"].query("type == 'WORD'").copy()
    out = pb.get_itemno_matrices(word_df)
    np.testing.assert_array_equal(out, exp["pres_itemnos"])


# ---------------------------------------------------------------------------
# get_all_matrices
# ---------------------------------------------------------------------------

@pytest.mark.property
def test_get_all_matrices_shapes_and_consistency():
    exp = make_experiment(seed=4)
    pres, rec, recalls = pb.get_all_matrices(exp["events"].copy())
    L = exp["list_length"]
    assert pres.shape[1] == L
    assert recalls.shape == rec.shape
    assert pres.shape[0] == exp["n_lists"]
    # recalls derived consistently with pres/rec
    for t in range(recalls.shape[0]):
        for r in range(recalls.shape[1]):
            v = recalls[t, r]
            if v > 0:
                assert pres[t, v - 1] == rec[t, r]
            elif v == 0:
                assert rec[t, r] == 0


@pytest.mark.property
def test_get_all_matrices_matches_factory_pres_matrix():
    exp = make_experiment(seed=8)
    pres, rec, recalls = pb.get_all_matrices(exp["events"].copy())
    np.testing.assert_array_equal(pres, exp["pres_itemnos"])


@pytest.mark.edge_case
def test_get_all_matrices_drops_lists_missing_a_type():
    # A list with WORD rows but no REC_WORD rows (and vice versa) is excluded,
    # because get_all_matrices keeps only lists where both types are present.
    rows = []
    # list 0: complete (pres + rec)
    for pos, ino in enumerate([1, 2, 3]):
        rows.append(dict(subject="s0", session=0, list=0, type="WORD", itemno=ino))
    for ino in [2, 1]:
        rows.append(dict(subject="s0", session=0, list=0, type="REC_WORD", itemno=ino))
    # list 1: presentations only (no recalls) -> should be dropped
    for ino in [4, 5, 6]:
        rows.append(dict(subject="s0", session=0, list=1, type="WORD", itemno=ino))
    df = pd.DataFrame(rows)
    pres, rec, recalls = pb.get_all_matrices(df)
    # only one list survives
    assert pres.shape[0] == 1
    assert rec.shape[0] == 1
    np.testing.assert_array_equal(pres, np.array([[1, 2, 3]]))
    np.testing.assert_array_equal(recalls[0], np.array([2, 1]))


@pytest.mark.edge_case
def test_get_all_matrices_intrusion_in_events_maps_to_minus_one():
    rows = []
    for ino in [1, 2, 3]:
        rows.append(dict(subject="s0", session=0, list=0, type="WORD", itemno=ino))
    # recall includes itemno 99 which was never presented on this list
    for ino in [2, 99, 1]:
        rows.append(dict(subject="s0", session=0, list=0, type="REC_WORD", itemno=ino))
    df = pd.DataFrame(rows)
    pres, rec, recalls = pb.get_all_matrices(df)
    np.testing.assert_array_equal(recalls[0], np.array([2, -1, 1]))
