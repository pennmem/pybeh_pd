"""Execute the README quickstart so documentation can't silently drift.

Extracts the first ```python fenced block from README.md, runs it, and checks the
documented results (lag-0 CRP is NaN, the +/-1 lag-CRP values, the temporal
clustering factor). If the README example or the API changes incompatibly, this
test fails.
"""
from __future__ import annotations

import os
import re

import numpy as np

README = os.path.join(os.path.dirname(__file__), "..", "README.md")


def _first_python_block(md_text: str) -> str:
    m = re.search(r"```python\n(.*?)```", md_text, re.DOTALL)
    assert m, "no ```python block found in README.md"
    return m.group(1)


def test_readme_quickstart_runs():
    with open(README) as fh:
        code = _first_python_block(fh.read())
    ns: dict = {}
    exec(compile(code, "README.md", "exec"), ns)

    crp = ns["crp"].set_index("lag")["prob"]
    assert np.isnan(crp.loc[0])
    assert np.isclose(crp.loc[1], 2 / 3)
    assert np.isclose(crp.loc[-1], 2 / 3)

    import pybeh_pd as pb
    tf = pb.pd_temp_fact(ns["events"])
    assert np.isclose(tf, 0.875)
