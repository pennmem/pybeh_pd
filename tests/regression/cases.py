"""Single source of truth for behavior-lock regression cases.

Both the golden-capture script (``capture_golden.py``) and the replay test
(``test_regression.py``) import :func:`build_cases`, so the set of cases and how
each result is reduced to a numpy array can never drift between capture and check.

Each case maps a string name to a float ndarray (NaNs allowed; compared with
``np.isclose(..., equal_nan=True)``). Results that are DataFrames or tuples are
reduced here to arrays. The known-broken
``loftus_masson_unequal_variance_kahana`` is intentionally excluded (it raises;
that behavior is locked separately in the test via ``pytest.raises``).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))              # tests/  (factories)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))  # src/ (pybeh_pd pkg)
import pybeh_pd as pb  # noqa: E402
from factories import make_experiment  # noqa: E402


def _arr(x):
    return np.asarray(x, dtype=float)


def _repeated_pres_case():
    """A hand-built repeated-presentation case for the max_n_reps>1 / 3D paths.

    Two trials, list length 4, item 2 presented twice (serial positions 2 and 4).
    """
    pres = np.array([[1, 2, 3, 2], [1, 2, 3, 2]])
    rec = np.array([[2, 1, 3, 0], [3, 2, 1, 0]])
    return pres, rec


def build_cases():
    cases: dict[str, np.ndarray] = {}

    # --- matrix-level functions over several seeded experiments -------------
    for seed in (0, 1, 2):
        ex = make_experiment(seed=seed)
        pres, rec, ll = ex["pres_itemnos"], ex["rec_itemnos"], ex["list_length"]
        subs = ["_"] * pres.shape[0]
        recalls = pb.make_recalls_matrix(pres, rec)
        poss = pb.make_poss_recalls_matrix(pres)
        sims = pb.get_sim_mat(ex["events"], ex["feat_cols"])

        p = f"seed{seed}"
        cases[f"make_recalls_matrix__{p}"] = _arr(recalls)
        cases[f"make_poss_recalls_matrix__{p}"] = _arr(poss)
        cases[f"clean_mask__{p}"] = _arr(pb.make_clean_recalls_mask2d(recalls))
        cases[f"crp__{p}"] = _arr(pb.crp(recalls=recalls, subjects=subs,
                                         listLength=ll, lag_num=4))
        cases[f"temp_fact__{p}"] = _arr(pb.temp_fact(recalls=recalls,
                                                     subjects=subs, listLength=ll))
        bin_means, crp = pb.sem_crp(recalls=recalls, recalls_itemnos=rec,
                                    pres_itemnos=pres, subjects=subs,
                                    sem_sims=sims, n_bins=5, listLength=ll)
        cases[f"sem_crp_binmeans__{p}"] = _arr(bin_means)
        cases[f"sem_crp_crp__{p}"] = _arr(crp)
        cases[f"dist_fact__{p}"] = _arr(pb.dist_fact(rec_itemnos=rec,
                                                     pres_itemnos=pres,
                                                     subjects=subs, dist_mat=sims))
        cases[f"get_sim_mat__{p}"] = _arr(sims)

    # --- repeated-presentation (3D) paths: min_crp / min_temp_fact ----------
    pres_r, rec_r = _repeated_pres_case()
    subs_r = ["_", "_"]
    recalls3d = pb.make_recalls_matrix(pres_r, rec_r, max_n_reps=2)
    poss3d = pb.make_poss_recalls_matrix(pres_r, max_n_reps=2)
    cases["make_recalls_matrix__rep3d"] = _arr(recalls3d)
    cases["make_poss_recalls_matrix__rep3d"] = _arr(poss3d)
    cases["min_crp__rep"] = _arr(pb.min_crp(recalls=recalls3d, poss_recalls=poss3d,
                                            subjects=subs_r, listLength=4, lag_num=3))
    cases["min_temp_fact__rep"] = _arr(pb.min_temp_fact(
        recalls=recalls3d, poss_recalls=poss3d, subjects=subs_r, listLength=4))

    # --- helpers ------------------------------------------------------------
    cases["get_min_trans"] = _arr([pb.get_min_trans([3, 1], [2]),
                                   pb.get_min_trans([5], [2, 4]),
                                   pb.get_min_trans([1, 4], [4])])
    # Helpers are called internally with numpy scalars + numpy arrays; mirror that
    # (a Python-int `actual` takes a different, numpy-2-incompatible 0-d path).
    cases["temp_percentile_rank"] = _arr([
        pb.temp_percentile_rank(np.int64(2), np.array([1, 2, 3, 4])),
        pb.temp_percentile_rank(np.int64(3), np.array([3, 3, 1])),
        np.nan if pb.temp_percentile_rank(np.int64(1), np.array([1])) is None else 0.0,
    ])
    cases["dist_percentile_rank"] = _arr([
        pb.dist_percentile_rank(np.float64(0.5), np.array([0.1, 0.5, 0.9])),
        pb.dist_percentile_rank(np.float64(0.5), np.array([0.1, 0.5, 0.9]),
                                is_similarity=True),
    ])

    # --- pandas wrappers ----------------------------------------------------
    for seed in (0, 1, 2):
        ex = make_experiment(seed=seed)
        df, fc = ex["events"], ex["feat_cols"]
        p = f"seed{seed}"
        cases[f"pd_crp__{p}"] = _arr(pb.pd_crp(df)["prob"].values)
        cases[f"pd_min_crp__{p}"] = _arr(pb.pd_min_crp(df)["prob"].values)
        cases[f"pd_temp_fact__{p}"] = _arr([pb.pd_temp_fact(df)])
        # pd_min_temp_fact(df) with the default max_n_reps=1 raises (bug B4); the
        # raise is locked in the test. Capture the working max_n_reps=2 path here.
        cases[f"pd_min_temp_fact_r2__{p}"] = _arr([pb.pd_min_temp_fact(df, max_n_reps=2)])
        cases[f"pd_dist_fact__{p}"] = _arr([pb.pd_dist_fact(df, sim_columns=fc)])
        cases[f"pd_sem_crp__{p}"] = _arr(pb.pd_sem_crp(df, sim_columns=fc)["prob"].values)
        cases[f"get_itemno_matrices__{p}"] = _arr(
            pb.get_itemno_matrices(df.query("type == 'WORD'").copy()))
        pres_m, rec_m, recalls_m = pb.get_all_matrices(df)
        cases[f"get_all_matrices_pres__{p}"] = _arr(pres_m)
        cases[f"get_all_matrices_recalls__{p}"] = _arr(recalls_m)

    # --- per-list / per-subject aggregating variants (groupby.apply paths) ---
    # These read semantic features off the WORD rows directly (serialpos-indexed).
    bins = [0.0, 2.0, 3.0, 4.0, 5.0]
    for seed in (0, 1):
        ex = make_experiment(seed=seed)
        df, fc = ex["events"], ex["feat_cols"]
        li = f"seed{seed}"
        one_list = df.query("list == 0").copy()
        dfl = pb.pd_dist_fact_list(one_list, sim_columns=fc)
        cases[f"pd_dist_fact_list_df__{li}"] = _arr(dfl["dist_fact"].values)
        cases[f"pd_dist_fact_list_tc__{li}"] = _arr(
            [dfl["total"].values[0], dfl["count"].values[0]])
        sl = pb.pd_sem_crp_list(one_list, sim_columns=fc, bins=bins)
        cases[f"pd_sem_crp_list__{li}"] = _arr(sl["prob"].values)

        li_idx = ["subject", "session", "list"]
        dsub = pb.pd_dist_fact_list_sub(df, sim_columns=fc, list_index=li_idx)
        cases[f"pd_dist_fact_list_sub__{li}"] = _arr(dsub["dist_fact"].values)
        ssub = pb.pd_sem_crp_list_sub(df, sim_columns=fc, bins=bins, list_index=li_idx)
        cases[f"pd_sem_crp_list_sub__{li}"] = _arr(ssub["prob"].values)

    # --- Loftus-Masson / Cousineau CI helpers -------------------------------
    rng = np.random.default_rng(7)
    mat = rng.normal(size=(10, 4)) + np.arange(4)
    cases["lm_equal_kahana"] = _arr(pb.loftus_masson_equal_variance_kahana(mat))

    long = make_experiment(seed=3)["events"].query("type == 'REC_WORD'").copy()
    long["val"] = rng.normal(size=len(long))
    lm = pb.loftus_masson_analytic(long, "subject", "list", "val")
    cases["lm_analytic_unequal"] = _arr(lm["CI_unequal"].values)
    cases["lm_analytic_equal"] = _arr(lm["CI_equal"].values)
    cous = pb.cousineau(long, "subject", "list", "val")
    cases["cousineau_adj"] = _arr(cous["adj_val"].values)
    cases["cousineau_cmo"] = _arr(cous["cmo_adj_val"].values)

    return cases
