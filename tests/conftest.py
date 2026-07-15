"""Test-process defaults shared by unittest- and pytest-style tests."""

import os


# Rendering tests only save files. A GUI backend is both unnecessary and
# unstable in headless Windows/CI sessions, so select Agg before pyplot loads.
os.environ.setdefault("MPLBACKEND", "Agg")
