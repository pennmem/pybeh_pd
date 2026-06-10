"""Loader for the committed ltpFR2 real-data fixture.

5 sessions of subject LTP093 from the ltpFR2 scalp-EEG free-recall experiment
(an open dataset) are committed as ``tests/data/ltpfr2_5sessions.csv`` so the
real-data integration tests run anywhere — no rhino mount and no cmlreaders
needed. The long-format schema matches what the ``pybeh_pd`` ``pd_*`` functions
expect:

    columns: subject, session, list, type ('WORD'|'REC_WORD'), item_num, serialpos

``item_num`` is the 1-based wordpool id (the ``itemno`` pybeh_pd uses); encoding
events are ``WORD`` and retrieval events are ``REC_WORD``. To refresh the fixture
from rhino, re-run ``tests/data/_regenerate_ltpfr2.py`` (needs cmlreaders).
"""
from __future__ import annotations

import functools
import os

import numpy as np
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIXTURE_CSV = os.path.join(_DATA_DIR, "ltpfr2_5sessions.csv")
GOLDEN_NPZ = os.path.join(_DATA_DIR, "ltpfr2_golden.npz")


@functools.lru_cache(maxsize=1)
def load_ltpfr2_long() -> pd.DataFrame:
    """Load the committed 5-session ltpFR2 fixture as a long DataFrame."""
    return pd.read_csv(FIXTURE_CSV)


@functools.lru_cache(maxsize=1)
def load_ltpfr2_golden() -> dict:
    """Load the frozen pybeh_pd outputs captured on the fixture."""
    with np.load(GOLDEN_NPZ) as data:
        return {k: data[k] for k in data.files}


ITEMNO = "item_num"


def capture_outputs(df) -> dict:
    """Every pybeh_pd analysis applicable to plain free-recall events, as a
    {name: float-ndarray} dict. Single source of truth shared by the golden
    regeneration script and the regression test, so they cannot drift.

    Semantic CRP / distance-factor are intentionally absent: ltpFR2 events carry
    no semantic feature space (those are locked on the synthetic suite instead).
    """
    import pybeh_pd as pb

    def a(x):
        return np.asarray(x, dtype=float)

    out = {}
    pres, rec, recalls = pb.get_all_matrices(df, itemno_column=ITEMNO)
    out["pres_itemnos"] = a(pres)
    out["rec_itemnos"] = a(rec)
    out["recalls"] = a(recalls)
    out["poss_recalls"] = a(pb.make_poss_recalls_matrix(pres))
    out["clean_mask"] = a(pb.make_clean_recalls_mask2d(recalls))
    subs = ["_"] * pres.shape[0]
    out["crp_matrix"] = a(pb.crp(recalls=recalls, subjects=subs,
                                 listLength=pres.shape[1], lag_num=5))
    out["temp_fact_matrix"] = a(pb.temp_fact(recalls=recalls, subjects=subs,
                                             listLength=pres.shape[1]))
    crp = pb.pd_crp(df, lag_num=5, itemno_column=ITEMNO).sort_values("lag")
    out["pd_crp_prob"] = a(crp["prob"].values)
    out["pd_crp_lag"] = a(crp["lag"].values)
    out["pd_min_crp_prob"] = a(pb.pd_min_crp(df, lag_num=5, itemno_column=ITEMNO)
                               .sort_values("lag")["prob"].values)
    out["pd_temp_fact"] = a([pb.pd_temp_fact(df, itemno_column=ITEMNO)])
    out["pd_min_temp_fact_r2"] = a([pb.pd_min_temp_fact(df, itemno_column=ITEMNO,
                                                        max_n_reps=2)])
    sess_ids = sorted(df["session"].unique())
    out["session_ids"] = a(sess_ids)
    out["pd_temp_fact_per_session"] = a(
        [pb.pd_temp_fact(df[df["session"] == s], itemno_column=ITEMNO)
         for s in sess_ids])
    out["pd_crp_prob_per_session"] = a(
        [pb.pd_crp(df[df["session"] == s], lag_num=5, itemno_column=ITEMNO)
         .sort_values("lag")["prob"].values for s in sess_ids])
    return out
