"""Eckfall-Matrix: jeder Regler der Seitenleiste am Minimum und am Maximum (einzeln und alle zugleich), nur der Lauf zählt (kein Python-Fehler)."""

import os

from streamlit.testing.v1 import AppTest
import pytest

import wfk_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


REGLER = ("slider", "select_slider", "selectbox", "radio", "toggle", "number_input")


def _setze(at, key, wert):
    for art in REGLER:
        try:
            getattr(at.sidebar, art)(key=key).set_value(wert)
            return
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} nicht gefunden")


@pytest.mark.parametrize("key", list(P.SETTING_SPECS))
@pytest.mark.parametrize("rand", ["lo", "hi"])
def test_jeder_regler_am_minimum_und_maximum(key, rand):
    spec = P.SETTING_SPECS[key]
    at = _fresh_app()
    _setze(at, key, spec.lo if rand == "lo" else spec.hi)
    at.run(timeout=TIMEOUT)
    assert not at.exception, (key, rand, [str(e) for e in at.exception])


@pytest.mark.parametrize("rand", ["lo", "hi"])
def test_alle_regler_zugleich_am_minimum_und_am_maximum(rand):
    at = _fresh_app()
    for key, spec in P.SETTING_SPECS.items():
        _setze(at, key, spec.lo if rand == "lo" else spec.hi)
    at.run(timeout=TIMEOUT)
    assert not at.exception, (rand, [str(e) for e in at.exception])
