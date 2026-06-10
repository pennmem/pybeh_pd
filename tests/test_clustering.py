"""Skeptical unit tests for the clustering-factor family in ``pybeh_pd``.

Covered functions
-----------------
- ``temp_percentile_rank`` / ``dist_percentile_rank``   (from ``pybeh_copy``)
- ``temp_fact``                                          (from ``pybeh_copy``)
- ``min_temp_fact``                                      (3D repeated-pres path)
- ``dist_fact``
- ``pd_temp_fact`` / ``pd_min_temp_fact`` / ``pd_dist_fact``
- ``pd_dist_fact_list`` / ``pd_dist_fact_list_sub``

Behavior-preservation rules: where the math disagrees with the code, or a code
path raises, the test PINS the current behavior (``@pytest.mark.pins_bug``) and
the divergence is reported separately -- nothing in the source is changed.

Numpy note: this repo runs on numpy 1.24 (numpy1). The percentile helpers are
invoked internally with numpy scalars + numpy arrays; the tests mirror that
(passing ``np.int64`` / ``np.float64`` scalars) both to match real call sites
and to avoid the numpy-2 0-d ``nonzero`` path (known bug B5), which would only
fire under numpy>=2.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import pybeh_pd as pb
from factories import make_experiment

# The package resolves to the src/ layout (conftest puts src/ on the path).
assert pb.__file__ is not None and pb.__file__.endswith(
    os.path.join("pybeh_pd", "__init__.py")), pb.__file__


# Several pandas wrappers emit SettingWithCopyWarning / FutureWarning that are
# orthogonal to the math under test.
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


# ---------------------------------------------------------------------------
# 1. HAND-CODED VALUE TESTS
# ---------------------------------------------------------------------------

@pytest.mark.hand_coded
def test_temp_percentile_rank_hand():
    # possible sorted DESC = [3, 2, 1]; actual=1 is the smallest distance ->
    # it occupies rank index 2 of 3 -> 2 / (3-1) = 1.0  (max clustering).
    assert pb.temp_percentile_rank(np.int64(1), np.array([3, 2, 1])) == 1.0
    # actual=3 is the largest distance -> index 0 -> 0 / 2 = 0.0 (anti-cluster).
    assert pb.temp_percentile_rank(np.int64(3), np.array([3, 2, 1])) == 0.0
    # actual=2 is the median -> index 1 -> 1 / 2 = 0.5 (chance).
    assert pb.temp_percentile_rank(np.int64(2), np.array([3, 2, 1])) == 0.5


@pytest.mark.hand_coded
def test_temp_percentile_rank_ties_and_none():
    # Ties: sorted DESC [3, 3, 1]; actual=3 matches indices 0 and 1 ->
    # mean(0,1)=0.5 -> 0.5 / (3-1) = 0.25.
    assert pb.temp_percentile_rank(np.int64(3), np.array([3, 3, 1])) == 0.25
    # No match -> None.
    assert pb.temp_percentile_rank(np.int64(5), np.array([3, 2, 1])) is None
    # Fewer than 2 possibles -> None.
    assert pb.temp_percentile_rank(np.int64(1), np.array([1])) is None
    # Exactly 0 possibles -> None.
    assert pb.temp_percentile_rank(np.int64(1), np.array([])) is None


@pytest.mark.hand_coded
def test_dist_percentile_rank_is_similarity_flips_sort():
    poss = np.array([0.1, 0.5, 0.9])
    # Distance mode: sorted DESC [0.9, 0.5, 0.1]. actual=0.1 (smallest distance,
    # i.e. closest) -> index 2 -> 2/2 = 1.0.
    assert pb.dist_percentile_rank(np.float64(0.1), poss) == 1.0
    # Largest distance -> index 0 -> 0.0.
    assert pb.dist_percentile_rank(np.float64(0.9), poss) == 0.0
    # Median -> 0.5.
    assert pb.dist_percentile_rank(np.float64(0.5), poss) == 0.5
    # Similarity mode flips the sort to ASC [0.1, 0.5, 0.9]; actual=0.1 (lowest
    # similarity -> most distant) -> index 0 -> 0.0.
    assert pb.dist_percentile_rank(np.float64(0.1), poss, is_similarity=True) == 0.0
    # Highest similarity -> index 2 -> 1.0.
    assert pb.dist_percentile_rank(np.float64(0.9), poss, is_similarity=True) == 1.0


@pytest.mark.hand_coded
def test_temp_fact_perfect_and_anticlustered():
    # Perfectly clustered recall (adjacent serial positions) -> factor 1.0.
    assert pb.temp_fact(recalls=np.array([[1, 2, 3, 4]]),
                        subjects=["_"], listLength=4)[0] == 1.0
    # Recall [1,4,2,3]: each made transition is the *largest* available ->
    #   1->4: poss |i-1| for i in {2,3,4} = {1,2,3}; actual=3 -> 0.0
    #   4->2: poss |i-4| for i in {2,3} = {2,1}; actual=2 -> 0.0
    #   2->3: only one possible -> None (skipped)
    # mean = 0.0.
    assert pb.temp_fact(recalls=np.array([[1, 4, 2, 3]]),
                        subjects=["_"], listLength=4)[0] == 0.0


@pytest.mark.hand_coded
def test_temp_fact_partial_by_hand():
    # Recall [2,1,3] on listLength 4, fully worked out:
    #   2->1: seen={2}; poss |i-2| for i in {1,3,4} = {1,1,2}; actual=1 ->
    #         sorted DESC [2,1,1], matches idx 1,2 -> mean 1.5 / 2 = 0.75
    #   1->3: seen={2,1}; poss |i-1| for i in {3,4} = {2,3}; actual=2 ->
    #         sorted DESC [3,2], match idx 1 -> 1/1 = 1.0
    # mean = (0.75 + 1.0) / 2 = 0.875.
    got = pb.temp_fact(recalls=np.array([[2, 1, 3, 0]]),
                       subjects=["_"], listLength=4)[0]
    assert got == pytest.approx(0.875)


@pytest.mark.hand_coded
def test_dist_fact_hand_and_similarity_flip():
    dist = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    pres = np.array([[1, 2, 3]])
    # Recall 1->2 (the closest neighbour). poss dist from item1 to {2,3} =
    # [1, 2]; actual = 1 -> sorted DESC [2,1] match idx 1 -> 1.0 (max
    # distance-clustering for the *shortest* hop).
    rec = np.array([[1, 2, 0]])
    assert pb.dist_fact(rec_itemnos=rec, pres_itemnos=pres,
                        subjects=["_"], dist_mat=dist)[0] == 1.0
    # Flipping is_similarity on the same matrix sends a short hop to the
    # opposite end -> 0.0.
    assert pb.dist_fact(rec_itemnos=rec, pres_itemnos=pres, subjects=["_"],
                        dist_mat=dist, is_similarity=True)[0] == 0.0


# ---------------------------------------------------------------------------
# 2. MATHEMATICAL PROPERTIES
# ---------------------------------------------------------------------------

def _in_unit_or_nan(x):
    x = float(x)
    return np.isnan(x) or (0.0 - 1e-12 <= x <= 1.0 + 1e-12)


@pytest.mark.property
def test_temp_fact_in_unit_interval():
    for seed in range(6):
        ex = make_experiment(seed=seed)
        recalls = pb.make_recalls_matrix(ex["pres_itemnos"], ex["rec_itemnos"])
        vals = pb.temp_fact(recalls=recalls,
                            subjects=["_"] * recalls.shape[0],
                            listLength=ex["list_length"])
        assert all(_in_unit_or_nan(v) for v in vals)


@pytest.mark.property
def test_dist_fact_in_unit_interval():
    for seed in range(6):
        ex = make_experiment(seed=seed)
        sims = pb.get_sim_mat(ex["events"], ex["feat_cols"])
        recalls = pb.make_recalls_matrix(ex["pres_itemnos"], ex["rec_itemnos"])
        # get_sim_mat returns Euclidean *distances* -> is_similarity defaults
        # to False, which is correct here. Diagonal is 0, so the B2 warning
        # heuristic is not tripped.
        vals = pb.dist_fact(rec_itemnos=ex["rec_itemnos"],
                            pres_itemnos=ex["pres_itemnos"],
                            subjects=["_"] * recalls.shape[0], dist_mat=sims)
        assert all(_in_unit_or_nan(v) for v in vals)


@pytest.mark.property
def test_dist_fact_random_matrices_stay_in_unit():
    rng = np.random.default_rng(7)
    for _ in range(30):
        d = rng.random((8, 8))
        d = (d + d.T) / 2.0
        np.fill_diagonal(d, 0.0)
        pres = np.array([np.arange(1, 9)])
        rec = np.zeros((1, 8), dtype=int)
        rec[0, :5] = rng.permutation(np.arange(1, 9))[:5]
        v = pb.dist_fact(rec_itemnos=rec, pres_itemnos=pres,
                         subjects=["_"], dist_mat=d)[0]
        assert _in_unit_or_nan(v)


@pytest.mark.property
def test_min_temp_fact_3d_in_unit_interval():
    # Repeated-presentation case (item 2 presented twice). max_n_reps=2 path.
    pres = np.array([[1, 2, 3, 4, 2], [1, 2, 3, 4, 2]])
    rec = np.array([[2, 4, 1, 3, 0], [3, 1, 4, 2, 0]])
    recalls3d = pb.make_recalls_matrix(pres, rec, max_n_reps=2)
    poss3d = pb.make_poss_recalls_matrix(pres, max_n_reps=2)
    vals = pb.min_temp_fact(recalls=recalls3d, poss_recalls=poss3d,
                            subjects=["_", "_"], listLength=5)
    assert all(_in_unit_or_nan(v) for v in vals)


@pytest.mark.property
def test_pd_temp_fact_and_pd_dist_fact_in_unit_interval():
    for seed in range(4):
        ex = make_experiment(seed=seed)
        df, fc = ex["events"], ex["feat_cols"]
        assert _in_unit_or_nan(pb.pd_temp_fact(df))
        assert _in_unit_or_nan(pb.pd_dist_fact(df, sim_columns=fc))


@pytest.mark.property
def test_percentile_rank_always_in_unit_or_none():
    rng = np.random.default_rng(11)
    for _ in range(200):
        poss = rng.integers(0, 6, size=rng.integers(0, 6))
        # actual drawn from the possible set or an out-of-range value.
        if len(poss) and rng.random() < 0.7:
            actual = np.int64(rng.choice(poss))
        else:
            actual = np.int64(99)
        r = pb.temp_percentile_rank(actual, poss)
        assert r is None or _in_unit_or_nan(r)
        if len(poss) < 2:
            assert r is None


@pytest.mark.property
def test_dist_fact_is_similarity_direction_on_hand_example():
    # On the SAME distance matrix, the shortest hop scores high in distance
    # mode (clustering) and low in similarity mode -- opposite ends.
    dist = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    pres = np.array([[1, 2, 3]])
    rec = np.array([[1, 2, 0]])  # closest possible hop
    f = pb.dist_fact(rec_itemnos=rec, pres_itemnos=pres,
                     subjects=["_"], dist_mat=dist, is_similarity=False)[0]
    t = pb.dist_fact(rec_itemnos=rec, pres_itemnos=pres,
                     subjects=["_"], dist_mat=dist, is_similarity=True)[0]
    assert f == 1.0 and t == 0.0
    # With a single binary transition (no ties), the two modes are
    # complementary.
    assert f + t == pytest.approx(1.0)


@pytest.mark.property
def test_dist_fact_negated_distance_matches_similarity_flag():
    # is_similarity=True on a distance matrix should equal is_similarity=False
    # on (max - distance), provided we reset the diagonal so the B2 heuristic
    # (large diagonal => warn) does not fire.
    dist = np.array([[0., 1., 2., 3.],
                     [1., 0., 1., 2.],
                     [2., 1., 0., 1.],
                     [3., 2., 1., 0.]])
    pres = np.array([[1, 2, 3, 4]])
    rec = np.array([[1, 3, 2, 0]])
    negdist = dist.max() - dist
    np.fill_diagonal(negdist, 0.0)
    a = pb.dist_fact(rec_itemnos=rec, pres_itemnos=pres, subjects=["_"],
                     dist_mat=dist, is_similarity=True)[0]
    b = pb.dist_fact(rec_itemnos=rec, pres_itemnos=pres, subjects=["_"],
                     dist_mat=negdist, is_similarity=False)[0]
    assert a == pytest.approx(b)


@pytest.mark.pins_bug
def test_dist_fact_diagonal_heavy_matrix_raises_nameerror_B2():
    # KNOWN BUG B2: dist_fact() calls warnings.warn(...) but ``warnings`` is
    # never imported (pybeh_pd.py:640). A similarity-looking matrix (large
    # diagonal) with is_similarity=False reaches that branch and raises
    # NameError instead of warning. Pin the current behavior.
    diag_heavy = np.array([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])
    pres = np.array([[1, 2, 3]])
    rec = np.array([[1, 2, 0]])
    with pytest.raises(NameError):
        pb.dist_fact(rec_itemnos=rec, pres_itemnos=pres,
                     subjects=["_"], dist_mat=diag_heavy, is_similarity=False)


@pytest.mark.pins_bug
def test_dist_fact_diagonal_heavy_ok_when_is_similarity_true():
    # The same diagonal-heavy matrix does NOT raise when is_similarity=True,
    # because that branch is guarded by ``not is_similarity``. Confirms the B2
    # trigger is specifically the (diagonal-heavy AND is_similarity=False)
    # combination, and that a similarity matrix is handled normally otherwise.
    diag_heavy = np.array([[10., 1., 2.], [1., 10., 1.], [2., 1., 10.]])
    pres = np.array([[1, 2, 3]])
    rec = np.array([[1, 2, 0]])
    v = pb.dist_fact(rec_itemnos=rec, pres_itemnos=pres,
                     subjects=["_"], dist_mat=diag_heavy, is_similarity=True)[0]
    assert _in_unit_or_nan(v)


@pytest.mark.property
def test_temp_fact_random_recalls_average_near_chance():
    # Statistical consistency: uniformly random full recall orders should give
    # a temporal factor that averages to ~0.5 (chance). Use one subject with
    # many trials and assert the aggregate lands in a loose chance band.
    rng = np.random.default_rng(42)
    ll, n_trials = 10, 400
    recalls = np.stack([rng.permutation(np.arange(1, ll + 1))
                        for _ in range(n_trials)])
    val = pb.temp_fact(recalls=recalls,
                       subjects=["_"] * n_trials, listLength=ll)[0]
    assert 0.3 <= val <= 0.7
    # Tighter sanity: this seed is deterministic and very close to 0.5.
    assert val == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# 3. EDGE CASES
# ---------------------------------------------------------------------------

@pytest.mark.edge_case
def test_temp_fact_empty_and_single_recall_nan():
    # No recalls at all -> NaN (count stays 0 -> 0/0).
    assert np.isnan(pb.temp_fact(recalls=np.array([[0, 0, 0, 0]]),
                                 subjects=["_"], listLength=4)[0])
    # A single recall -> no transition possible -> NaN.
    assert np.isnan(pb.temp_fact(recalls=np.array([[1, 0, 0, 0]]),
                                 subjects=["_"], listLength=4)[0])


@pytest.mark.edge_case
def test_temp_fact_all_intrusions_nan():
    # All extra-list intrusions (-1) -> no clean transitions -> NaN.
    assert np.isnan(pb.temp_fact(recalls=np.array([[-1, -1, -1, -1]]),
                                 subjects=["_"], listLength=4)[0])


@pytest.mark.edge_case
def test_dist_fact_empty_and_single_recall_nan():
    dist = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    pres = np.array([[1, 2, 3]])
    assert np.isnan(pb.dist_fact(rec_itemnos=np.array([[0, 0, 0]]),
                                 pres_itemnos=pres, subjects=["_"],
                                 dist_mat=dist)[0])
    assert np.isnan(pb.dist_fact(rec_itemnos=np.array([[1, 0, 0]]),
                                 pres_itemnos=pres, subjects=["_"],
                                 dist_mat=dist)[0])


@pytest.mark.edge_case
def test_dist_fact_ret_counts_shape():
    dist = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    pres = np.array([[1, 2, 3]])
    rec = np.array([[1, 2, 0]])
    final, total, count = pb.dist_fact(rec_itemnos=rec, pres_itemnos=pres,
                                       subjects=["_"], dist_mat=dist,
                                       ret_counts=True)
    assert final[0] == 1.0
    assert total[0] == 1.0
    assert count[0] == 1.0


@pytest.mark.pins_bug
def test_pd_min_temp_fact_default_raises_indexerror_B4():
    # KNOWN BUG B4: pd_min_temp_fact() defaults max_n_reps=1, which makes
    # make_poss_recalls_matrix squeeze to 2D; min_temp_fact then indexes
    # serialpos[0] on a scalar -> IndexError. Pin the raise.
    ex = make_experiment(seed=0)
    with pytest.raises(IndexError):
        pb.pd_min_temp_fact(ex["events"])


@pytest.mark.edge_case
def test_pd_min_temp_fact_max_n_reps_2_returns_unit_float():
    ex = make_experiment(seed=0)
    val = pb.pd_min_temp_fact(ex["events"], max_n_reps=2)
    assert isinstance(val, float)
    assert _in_unit_or_nan(val)


@pytest.mark.edge_case
def test_pd_dist_fact_no_presentations_returns_nan():
    # A subject DataFrame with no WORD/REC_WORD rows (only WORD_VALS) yields an
    # empty presentation matrix -> pres_itemnos.shape[1] == 0 -> np.nan.
    ex = make_experiment(seed=0)
    only_vals = ex["events"].query("type == 'WORD_VALS'").copy()
    assert np.isnan(pb.pd_dist_fact(only_vals, sim_columns=ex["feat_cols"]))


@pytest.mark.edge_case
def test_pd_dist_fact_list_one_list():
    # pd_dist_fact_list reads features off the WORD rows of a single list and
    # returns a one-row frame with dist_fact/total/count; dist_fact in [0,1].
    ex = make_experiment(seed=0)
    list0 = ex["events"].query("list == 0")
    out = pb.pd_dist_fact_list(list0, sim_columns=ex["feat_cols"])
    assert list(out.columns) == ["dist_fact", "total", "count"]
    assert len(out) == 1
    assert _in_unit_or_nan(out["dist_fact"].iloc[0])
    # count is the number of scored transitions; total/count == dist_fact.
    c = out["count"].iloc[0]
    if c > 0:
        assert out["dist_fact"].iloc[0] == pytest.approx(
            out["total"].iloc[0] / c)


@pytest.mark.edge_case
def test_pd_dist_fact_list_sub_aggregates_to_unit():
    # Aggregate across this subject's lists. Default list_index uses 'trial';
    # the factory uses 'list', so pass list_index explicitly (per the harness).
    ex = make_experiment(seed=0)
    df = ex["events"].query("type != 'WORD_VALS'").copy()
    out = pb.pd_dist_fact_list_sub(
        df, sim_columns=ex["feat_cols"],
        list_index=["subject", "session", "list"])
    assert "dist_fact" in out.columns
    assert len(out) == 1
    assert _in_unit_or_nan(out["dist_fact"].iloc[0])
    # Subject-level factor is the transition-weighted pool of all lists.
    assert out["dist_fact"].iloc[0] == pytest.approx(
        out["total"].iloc[0] / out["count"].iloc[0])


# ---------------------------------------------------------------------------
# Scrutiny: does temp_fact handle the seen-set across repeated serial positions?
# ---------------------------------------------------------------------------

@pytest.mark.property
def test_temp_fact_repeat_recall_severs_transition_chain():
    # Scrutiny of the seen-set / clean-mask interaction across a repeated serial
    # position. For [2,2,1] the clean mask is [1,0,1]: the *repeated* '2' is
    # masked at position 1. A transition is scored only when BOTH the current
    # and next positions are clean, so:
    #   j=0 (2->2): mask[1]=0 -> not scored
    #   j=1 (2->1): mask[1]=0 -> not scored
    # The repeat therefore severs the chain on BOTH sides and NO transition is
    # counted -> NaN. (It does NOT silently collapse to the plain [2,1] case.)
    assert np.isnan(pb.temp_fact(recalls=np.array([[2, 2, 1, 0]]),
                                 subjects=["_"], listLength=3)[0])
    # By contrast the un-repeated [2,1] scores a single clean transition:
    #   2->1 on LL=3: seen={2}; poss |i-2| for i in {1,3} = {1,1}; actual=1 ->
    #   sorted DESC [1,1] matches idx 0,1 -> mean 0.5 / 1 = 0.5.
    assert pb.temp_fact(recalls=np.array([[2, 1, 0, 0]]),
                        subjects=["_"], listLength=3)[0] == pytest.approx(0.5)
    # A repeat that does NOT sit between the two scored items leaves a later
    # clean transition intact: [1,2,1,3] -> mask [1,1,0,1]; only the 1->2
    # transition (positions 0,1) is scored. seen={1}; poss |i-1| for i in {2,3}
    # = {1,2}; actual=1 -> sorted DESC [2,1] match idx1 -> 1.0.
    assert pb.temp_fact(recalls=np.array([[1, 2, 1, 3]]),
                        subjects=["_"], listLength=3)[0] == pytest.approx(1.0)
