"""Pytest configuration.

The presence of this file at the repo root makes pytest add the repo root to
``sys.path``, so the top-level ``pybeh_pd`` / ``pybeh_copy`` modules import during
testing without an install. This is intentional for the pre-packaging phases; once
the package is installed (editable), the import resolves the same way.
"""
