"""Real-data integration tests on 5 ltpFR2 sessions (committed fixture).

Runs the actual ``pybeh_pd`` pipeline on real scalp-EEG free-recall data and
asserts both (a) the well-established empirical signatures of free recall — the
*temporal contiguity effect* (Kahana 1996) and above-chance *temporal clustering*
(Polyn, Norman & Kahana 2009) — and (b) that the outputs exactly match the frozen
golden values captured on this fixture, so the real-data behavior is locked too.

Data + golden live in tests/data/ and are loaded from disk; neither cmlreaders nor
rhino is required. The synthetic fixture factory is also checked against the real
event schema, which is what licenses the synthetic-data unit tests elsewhere.
"""
from __future__ import annotations

import numpy as np

import pytest

import pybeh_pd as pb
from real_data import load_ltpfr2_long, load_ltpfr2_golden, capture_outputs
from factories import make_experiment

ITEMNO = "item_num"
LIST_LENGTH = 24  # ltpFR2 lists are 24 items
RTOL, ATOL = 1e-9, 1e-12


# --- sanity: 5 sessions of valid free-recall data ---------------------------

def test_loaded_five_sessions():
    df = load_ltpfr2_long()
    assert df["session"].nunique() == 5
    assert set(df["type"].unique()) == {"WORD", "REC_WORD"}
    study = df.query("type == 'WORD'")
    assert (study.groupby("session")["list"].nunique() == 24).all()
    assert (study.groupby(["session", "list"]).size() == LIST_LENGTH).all()


# --- the synthetic factory mirrors the real schema (validates fixtures) ------

def test_synthetic_fixture_mirrors_real_schema():
    df = load_ltpfr2_long()
    syn = make_experiment(seed=0)["events"]
    for col in ["subject", "session", "list", "type"]:
        assert col in df.columns and col in syn.columns
    assert {"WORD", "REC_WORD"}.issubset(set(syn["type"].unique()))
    assert {"WORD", "REC_WORD"} == set(df["type"].unique())
    assert (df.query("type=='WORD'")[ITEMNO] > 0).all()
    assert (syn.query("type=='WORD'")["itemno"] > 0).all()


# --- lag-CRP: temporal contiguity effect (properties) -----------------------

def test_pd_crp_contiguity_and_asymmetry():
    crp = pb.pd_crp(load_ltpfr2_long(), lag_num=5, itemno_column=ITEMNO).set_index("lag")
    prob = crp["prob"]
    assert np.isnan(prob.loc[0])                       # lag 0 undefined
    valid = prob.drop(index=0)
    assert ((valid >= 0) & (valid <= 1)).all()         # valid probabilities
    assert prob.loc[1] > prob.loc[5]                   # forward contiguity
    assert prob.loc[-1] > prob.loc[-5]                 # backward contiguity
    assert prob.loc[1] == valid.max()                  # +1 most likely
    assert prob.loc[1] > prob.loc[-1]                  # forward asymmetry


def test_pd_crp_matches_per_subject_decomposition():
    """Canonical usage (FR1_analyses.ipynb cell 10) is pd_crp applied per subject:
    ``events.groupby('subject').apply(pb.pd_crp, itemno_column='item_num')``. For
    one subject the direct call must equal that decomposition. (The literal
    groupby-apply idiom itself breaks on pandas 3 — grouping column dropped — see
    bug_report.md; we express the decomposition by explicit slicing here.)"""
    df = load_ltpfr2_long()
    direct = pb.pd_crp(df, itemno_column=ITEMNO).sort_values("lag")["prob"].values
    per_subject = [
        pb.pd_crp(g, itemno_column=ITEMNO).sort_values("lag")["prob"].values
        for _, g in df.groupby("subject")
    ]
    assert len(per_subject) == 1
    assert np.allclose(direct, per_subject[0], equal_nan=True)


# --- temporal clustering factor: above chance (properties) ------------------

def test_pd_temp_fact_above_chance():
    tf = pb.pd_temp_fact(load_ltpfr2_long(), itemno_column=ITEMNO)
    assert 0.5 < tf <= 1.0
    assert tf > 0.6


def test_temp_fact_per_session_above_chance():
    df = load_ltpfr2_long()
    for sess, g in df.groupby("session"):
        tf = pb.pd_temp_fact(g, itemno_column=ITEMNO)
        assert 0.5 < tf <= 1.0, f"session {sess}: temp_fact={tf}"


# --- matrix builders produce valid serial positions on real data ------------

def test_get_all_matrices_valid_serialpos():
    pres, rec, recalls = pb.get_all_matrices(load_ltpfr2_long(), itemno_column=ITEMNO)
    assert pres.shape[1] == LIST_LENGTH
    assert recalls.shape == rec.shape
    assert recalls.min() >= -1
    assert recalls.max() <= LIST_LENGTH
    correct = recalls[recalls > 0]
    assert ((correct >= 1) & (correct <= LIST_LENGTH)).all()


# --- behavior lock: every applicable analysis matches frozen golden ----------
# capture_outputs() is the single source of truth shared with the golden
# regeneration script, so this checks the full real-data output set, not a subset.

GOLDEN_KEYS = sorted(load_ltpfr2_golden().keys())


def test_golden_case_set_is_complete():
    """Live capture and the committed golden must cover the same analyses."""
    live = capture_outputs(load_ltpfr2_long())
    assert set(live) == set(load_ltpfr2_golden())


@pytest.mark.parametrize("name", GOLDEN_KEYS)
def test_real_data_output_matches_golden(name):
    live = capture_outputs(load_ltpfr2_long())
    g = load_ltpfr2_golden()
    x = np.asarray(live[name], dtype=float)
    assert x.shape == g[name].shape, f"{name}: {x.shape} != {g[name].shape}"
    assert np.allclose(x, g[name], rtol=RTOL, atol=ATOL, equal_nan=True), (
        f"{name} diverged from golden")
