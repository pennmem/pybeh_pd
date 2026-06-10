"""Regenerate the committed ltpFR2 real-data fixture + comprehensive golden.

One-time/maintenance script. Requires cmlreaders + the rhino /protocols/ltp mount.
Writes:
  tests/data/ltpfr2_5sessions.csv   - 5 sessions of LTP093 (WORD + REC_WORD events)
  tests/data/ltpfr2_golden.npz      - frozen pybeh_pd outputs on that data

The golden freezes every analysis that applies to plain free-recall events (no
external semantic space): the matrix builders, the lag-CRP family (incl. the
repeated-presentation variants), and the temporal clustering factor, both
pooled-over-sessions and per-session. Semantic CRP / distance-factor are NOT
included here because ltpFR2 events carry no semantic feature columns; those are
locked on the synthetic suite (tests/regression) where features exist.

The integration test reads these committed files and needs neither cmlreaders nor
rhino. Re-run only to refresh the fixture (an intentional, reviewed change).
"""
import os
import sys

import numpy as np
import pandas as pd
import cmlreaders as cml

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # src/ (pybeh_pd pkg)
sys.path.insert(0, os.path.join(HERE, ".."))              # tests/ (real_data)
from real_data import capture_outputs  # noqa: E402

SUBJECT, N_SESSIONS, ITEMNO = "LTP093", 5, "item_num"


def load():
    idx = cml.get_data_index("ltp", rootdir="/").query("experiment == 'ltpFR2'")
    rows = (idx.query("subject == @SUBJECT and session != 23")
            .sort_values("session").head(N_SESSIONS))
    keep = ["subject", "session", "list", "type", "item_num", "serialpos"]
    frames = []
    for r in rows.itertuples():
        ev = cml.CMLReader(subject=r.subject, session=int(r.session),
                           experiment="ltpFR2", rootdir="/").load("events")
        if "list" in ev.columns:
            ev = ev.drop(columns="list")
        ev = ev.rename(columns={"trial": "list"}).query("type in ['WORD','REC_WORD']")
        frames.append(ev[keep])
    return pd.concat(frames, ignore_index=True)


def main():
    df = load()
    df.to_csv(os.path.join(HERE, "ltpfr2_5sessions.csv"), index=False)
    golden = capture_outputs(df)
    np.savez(os.path.join(HERE, "ltpfr2_golden.npz"), **golden)
    print(f"wrote ltpfr2_5sessions.csv rows={len(df)}")
    print(f"golden: {len(golden)} arrays")
    for k in sorted(golden):
        print(f"  {k:28s} shape={golden[k].shape}")


if __name__ == "__main__":
    main()
