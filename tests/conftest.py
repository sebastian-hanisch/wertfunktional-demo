"""Gemeinsame Fixtures: Wurzelordner im Suchpfad, Bevölkerung und kleine Zeilenmengen."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wfk_generator as G  # noqa: E402
import wfk_pop as P  # noqa: E402


@pytest.fixture(scope="session")
def pop():
    return P.bevoelkerung()


@pytest.fixture(scope="session")
def kleine_menge():
    return G.zeilenmenge(4242, 40)
