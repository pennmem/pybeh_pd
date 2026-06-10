"""Capture golden-master outputs for the behavior-lock regression suite.

Run ONCE to freeze current behavior:

    python tests/regression/capture_golden.py

Writes ``golden.npz`` next to this file. The replay test compares live output
against it. Re-run only when an *intended* behavior change has been reviewed and
approved (it should not happen during this renovation).
"""
from __future__ import annotations

import os

import numpy as np

from cases import build_cases


def main():
    cases = build_cases()
    out = os.path.join(os.path.dirname(__file__), "golden.npz")
    np.savez(out, **{k: np.asarray(v, dtype=float) for k, v in cases.items()})
    print(f"wrote {len(cases)} golden cases -> {out}")
    for k in sorted(cases):
        v = np.asarray(cases[k], dtype=float)
        print(f"  {k:34s} shape={v.shape}")


if __name__ == "__main__":
    main()
