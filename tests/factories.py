"""Fixture factories for pybeh_pd regression/behavior tests.

These synthesize the two data structures the package operates on:

1. **pybeh matrices** — ``pres_itemnos`` / ``rec_itemnos`` (and the derived
   recalls matrix), as consumed by the matrix-level functions (``crp``,
   ``temp_fact``, ``sem_crp``, ``dist_fact``, ...).
2. **long-format event DataFrames** — one row per presentation / recall /
   word-value event, as consumed by the ``pd_*`` wrappers.

Everything is seeded so captures are reproducible. A simulated free-recall
"experiment" presents items 1..list_length on each list and recalls a random
ordering of a subset, optionally inserting extra-list intrusions (-1) and
repeats, mirroring the matrix conventions the code documents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_experiment(
    n_lists=6,
    list_length=8,
    n_features=5,
    p_recall=0.7,
    p_intrusion=0.1,
    seed=0,
):
    """Simulate one subject's free-recall experiment.

    Returns a dict with the pybeh-style matrices and a long-format events
    DataFrame (columns: subject, session, list, type, itemno + feature columns
    ``f0..f{n_features-1}`` on the WORD_VALS rows).
    """
    rng = np.random.default_rng(seed)
    pool_size = n_lists * list_length

    # Stable semantic features per pool item (itemnos are 1-based).
    features = rng.normal(size=(pool_size, n_features))

    pres_rows = []
    rec_rows = []
    pres_itemnos = []
    rec_itemnos = []
    max_recs = 0

    for li in range(n_lists):
        itemnos = np.arange(li * list_length, (li + 1) * list_length) + 1
        pres_itemnos.append(itemnos.tolist())
        for pos, ino in enumerate(itemnos):
            # Presentation rows also carry the item's semantic features, so the
            # per-list `pd_*_list` API (which reads features off the WORD rows,
            # not separate WORD_VALS rows) can be exercised.
            prow: dict[str, object] = dict(
                subject="s0", session=0, list=li, type="WORD",
                itemno=int(ino), serialpos=pos + 1)
            for j in range(n_features):
                prow[f"f{j}"] = float(features[ino - 1, j])
            pres_rows.append(prow)

        # Recall a random subset in a random order.
        recalled_mask = rng.random(list_length) < p_recall
        recalled = itemnos[recalled_mask]
        rng.shuffle(recalled)
        recalled = list(recalled)
        # Maybe insert an extra-list intrusion (itemno not in this list -> -1
        # at the matrix level; here use a pool item from another list).
        if rng.random() < p_intrusion and pool_size > list_length:
            outside = [i for i in range(1, pool_size + 1)
                       if i not in set(itemnos.tolist())]
            recalled.insert(rng.integers(0, len(recalled) + 1),
                            int(rng.choice(outside)))

        rec_itemnos.append([int(x) for x in recalled])
        max_recs = max(max_recs, len(recalled))
        for pos, ino in enumerate(recalled):
            rec_rows.append(
                dict(subject="s0", session=0, list=li, type="REC_WORD",
                     itemno=int(ino), serialpos=pos + 1)
            )

    # Word-value rows (semantic features) — one per pool item, itemnos 1-based.
    feat_cols = [f"f{j}" for j in range(n_features)]
    wv_rows = []
    for ino in range(1, pool_size + 1):
        row: dict[str, object] = dict(
            subject="s0", session=0, list=-1, type="WORD_VALS",
            itemno=ino, serialpos=0)
        for j, c in enumerate(feat_cols):
            row[c] = float(features[ino - 1, j])
        wv_rows.append(row)

    events = pd.DataFrame(pres_rows + rec_rows + wv_rows)

    # Rectangular matrices padded with zeros (the matrix convention).
    pres_mat = np.array(pres_itemnos, dtype=int)
    rec_mat = np.zeros((n_lists, max(max_recs, 1)), dtype=int)
    for i, r in enumerate(rec_itemnos):
        rec_mat[i, : len(r)] = r

    return dict(
        events=events,
        pres_itemnos=pres_mat,
        rec_itemnos=rec_mat,
        features=features,
        feat_cols=feat_cols,
        list_length=list_length,
        n_lists=n_lists,
    )
