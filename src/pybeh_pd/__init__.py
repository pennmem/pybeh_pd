"""pybeh_pd — pandas-friendly wrappers around pybeh for free-recall analysis.

The public API is unchanged from the original single-file module: ``import
pybeh_pd as pb`` then call ``pb.pd_crp`` / ``pb.pd_temp_fact`` / ``pb.cousineau``
etc. The implementation now lives in two internal modules:

- ``_core``  — the analysis functions and pandas wrappers (formerly pybeh_pd.py).
- ``_pybeh`` — a vendored subset of pybeh (formerly pybeh_copy.py) so no external
  pybeh install is required.

Everything public from ``_core`` (which itself re-exports the vendored ``crp``,
``temp_fact``, ``make_clean_recalls_mask2d``, ``dist_percentile_rank``,
``temp_percentile_rank``) is re-exported here, preserving the original surface.
"""
from ._core import *  # noqa: F401,F403
