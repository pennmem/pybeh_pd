"""Real-data loader for ltpFR2 integration tests.

Loads behavioral events for a handful of ltpFR2 (scalp-EEG free recall) sessions
from rhino via ``cmlreaders`` and reshapes them into the long-format schema the
``pybeh_pd`` ``pd_*`` functions expect:

    columns: subject, session, list, type ('WORD'|'REC_WORD'), item_num, serialpos

Notes on the ltpFR2 event schema (see `LtpFR2 - Data Set Description.pdf`):
- ``item_num`` is the 1-based wordpool item id (the ``itemno`` pybeh_pd needs).
- the real list number lives in ``trial``; the ``list`` field is a junk -999, so
  we drop ``list`` and rename ``trial`` -> ``list`` (matching the lab's own
  loader in subjective_item_memorability/load_behavioral_data.py).
- encoding events are ``WORD``; retrieval events are ``REC_WORD``.

These are gated behind :func:`ltpfr2_available` so the suite skips cleanly off
rhino or without cmlreaders installed. Data are loaded live (never committed).
"""
from __future__ import annotations

import functools
import os

RHINO_ROOT = "/"
LTPFR2_PROTOCOL_DIR = "/protocols/ltp"


def ltpfr2_available() -> bool:
    """True iff cmlreaders is importable and the ltp protocol tree is mounted."""
    if not os.path.isdir(LTPFR2_PROTOCOL_DIR):
        return False
    try:
        import cmlreaders  # noqa: F401
    except Exception:
        return False
    return True


@functools.lru_cache(maxsize=8)
def load_ltpfr2_long(subject: str = "LTP093", n_sessions: int = 5):
    """Load the first ``n_sessions`` of one ltpFR2 subject as a long DataFrame.

    Cached so the (slow) rhino read happens once per (subject, n_sessions).
    Returns a DataFrame with columns
    [subject, session, list, type, item_num, serialpos], containing only WORD
    and REC_WORD events. Session 23 (a different wordpool) is skipped.
    """
    import cmlreaders as cml
    import pandas as pd

    idx = cml.get_data_index("ltp", rootdir=RHINO_ROOT).query(
        "experiment == 'ltpFR2'")
    rows = (
        idx.query("subject == @subject and session != 23")
        .sort_values("session")
        .head(n_sessions)
    )
    keep = ["subject", "session", "list", "type", "item_num", "serialpos"]
    frames = []
    for r in rows.itertuples():
        ev = cml.CMLReader(
            subject=r.subject, session=int(r.session),
            experiment="ltpFR2", rootdir=RHINO_ROOT,
        ).load("events")
        if "list" in ev.columns:
            ev = ev.drop(columns="list")
        ev = ev.rename(columns={"trial": "list"})
        ev = ev.query("type in ['WORD', 'REC_WORD']")
        frames.append(ev[keep])
    return pd.concat(frames, ignore_index=True)
