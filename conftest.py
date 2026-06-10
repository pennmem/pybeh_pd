"""Pytest configuration.

Puts the ``src/`` layout package and the ``tests/`` helper dir on ``sys.path`` so
``import pybeh_pd`` and ``from factories import ...`` / ``from real_data import
...`` resolve during testing without requiring an install. When the package is
also editable-installed (the pixi envs), the import resolves to the same ``src/``
tree, so there is no conflict.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
