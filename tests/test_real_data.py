"""Real-data integration tests on 5 ltpFR2 sessions (loaded via cmlreaders).

These run the actual ``pybeh_pd`` pipeline on real scalp-EEG free-recall data and
assert the well-established empirical signatures of free recall — the *temporal
contiguity effect* (Kahana 1996) and above-chance *temporal clustering* (Polyn,
Norman & Kahana 2009). They also confirm the synthetic fixture factory mirrors the
real event schema, which is what licenses the synthetic-data unit tests elsewhere.

Skipped automatically off rhino / without cmlreaders. Data are loaded live and
never committed.
"""
from __future__ import annotations

import numpy as np
import pytest

import pybeh_pd as pb
from real_data import ltpfr2_available, load_ltpfr2_long
from factories import make_experiment

pytestmark = [
    pytest.mark.needs_data,
    pytest.mark.integration,
    pytest.mark.skipif(not ltpfr2_available(),
                       reason="ltpFR2 data / cmlreaders not available"),
]

ITEMNO = "item_num"
LIST_LENGTH = 24  # ltpFR2 lists are 24 items


@pytest.fixture(scope="module")
def ltpfr2():
    return load_ltpfr2_long(subject="LTP093", n_sessions=5)


# --- sanity: we actually loaded 5 sessions of valid free-recall data ---------

def test_loaded_five_sessions(ltpfr2):
    assert ltpfr2["session"].nunique() == 5
    assert set(ltpfr2["type"].unique()) == {"WORD", "REC_WORD"}
    # 24 lists/session x 24 items = 576 study events per session
    study = ltpfr2.query("type == 'WORD'")
    per_sess = study.groupby("session")["list"].nunique()
    assert (per_sess == 24).all()
    assert (study.groupby(["session", "list"]).size() == LIST_LENGTH).all()


# --- the synthetic factory mirrors the real schema (validates fixtures) ------

def test_synthetic_fixture_mirrors_real_schema(ltpfr2):
    syn = make_experiment(seed=0)["events"]
    for col in ["subject", "session", "list", "type"]:
        assert col in ltpfr2.columns and col in syn.columns
    # both encode presentations as WORD and recalls as REC_WORD
    assert {"WORD", "REC_WORD"}.issubset(set(syn["type"].unique()))
    assert {"WORD", "REC_WORD"} == set(ltpfr2["type"].unique())
    # both carry a 1-based integer item id and a serialpos
    assert (ltpfr2.query("type=='WORD'")[ITEMNO] > 0).all()
    assert (syn.query("type=='WORD'")["itemno"] > 0).all()


# --- lag-CRP: temporal contiguity effect -------------------------------------

def test_pd_crp_contiguity_and_asymmetry(ltpfr2):
    crp = pb.pd_crp(ltpfr2, lag_num=5, itemno_column=ITEMNO).set_index("lag")
    prob = crp["prob"]
    # lag 0 is undefined
    assert np.isnan(prob.loc[0])
    # all probabilities are valid
    valid = prob.drop(index=0)
    assert ((valid >= 0) & (valid <= 1)).all()
    # contiguity: adjacent transitions dominate distant ones
    assert prob.loc[1] > prob.loc[5]
    assert prob.loc[-1] > prob.loc[-5]
    # +1 is the single most likely transition
    assert prob.loc[1] == valid.max()
    # forward asymmetry: forward adjacency beats backward adjacency
    assert prob.loc[1] > prob.loc[-1]


# --- temporal clustering factor: above chance --------------------------------

def test_pd_crp_matches_groupby_apply_convention(ltpfr2):
    """The canonical usage (FR1_analyses.ipynb cell 10) is
    ``events.groupby('subject').apply(pb.pd_crp, itemno_column='item_num')``.
    For a single subject that must equal the direct per-subject call, confirming
    we drive pybeh_pd the way the lab's notebooks do."""
    import warnings
    direct = pb.pd_crp(ltpfr2, itemno_column=ITEMNO).sort_values("lag")["prob"].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped = (
            ltpfr2.groupby("subject")
            .apply(pb.pd_crp, itemno_column=ITEMNO)
            .reset_index()
        )
    via = grouped.sort_values("lag")["prob"].values
    assert np.allclose(direct, via, equal_nan=True)


def test_pd_temp_fact_above_chance(ltpfr2):
    tf = pb.pd_temp_fact(ltpfr2, itemno_column=ITEMNO)
    # chance is 0.5; ltpFR2 shows strong clustering
    assert 0.5 < tf <= 1.0
    assert tf > 0.6


def test_temp_fact_per_session_above_chance(ltpfr2):
    for sess, g in ltpfr2.groupby("session"):
        tf = pb.pd_temp_fact(g, itemno_column=ITEMNO)
        assert 0.5 < tf <= 1.0, f"session {sess}: temp_fact={tf}"


# --- matrix builders produce valid serial positions on real data ------------

def test_get_all_matrices_valid_serialpos(ltpfr2):
    pres, rec, recalls = pb.get_all_matrices(ltpfr2, itemno_column=ITEMNO)
    assert pres.shape[1] == LIST_LENGTH
    assert recalls.shape == rec.shape
    # recalls entries are padding (0), intrusion (-1), or a serial position 1..24
    uniq = np.unique(recalls)
    assert uniq.min() >= -1
    assert recalls.max() <= LIST_LENGTH
    correct = recalls[recalls > 0]
    assert ((correct >= 1) & (correct <= LIST_LENGTH)).all()
