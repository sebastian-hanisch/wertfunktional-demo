"""Verdikte als reine Funktionen (Zahlen im Text an handgesetzten Eingaben) und je Preset die erwartete Verdikt-Zeile in der App."""

import dataclasses
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import wfk_constants as C
import wfk_evaluation as E

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_fmt():
    assert E.fmt(1.234) == "1,23" and E.fmt(0.4076, 3) == "0,408" and E.fmt(5, 0) == "5"


def test_verdict_zeile_summe_und_name_des_funktionals():
    ez = E.eine_zeile(3, C.HG_BEVOELKERUNG)
    k, t = E.verdict_zeile(ez)
    assert k == "info" and "Zeile 3 der Menge 1" in t and "Ein Beitrag ohne genanntes Wertfunktional ist keine Antwort" in t
    assert E.fmt(float(ez.phi_i[0]), 3) in t and E.fmt(float(ez.phi_k[0]), 3) in t and E.fmt(float(ez.phi_p[0]), 3) in t
    k, _ = E.verdict_zeile(dataclasses.replace(ez, mu=ez.mu + 1.0))                   # Summe passt nicht zu Vorhersage minus Grundwert: Fehler
    assert k == "error"


def test_verdict_wichtigkeit_nennt_zeilenmenge_und_beide_masse():
    k, t = E.verdict_wichtigkeit(E.wichtigkeit_bevoelkerung())
    assert "Zeilenmenge 1 224 gewichtete Stützpunkte" in t and "Rangkorrelation" in t and "Summe der Abweichungen" in t and "wechselt von Art zu Vorlauf" in t


def test_verdict_ziehungen_nennt_die_zahl_der_ziehungen_und_jede_groesse():
    t = E.verdict_ziehungen([E.ziehungen(m) for m in (20, 40)])[1]
    assert "20 000 Ziehungen" in t and "20 Instanzen" in t and "40 Instanzen" in t and "mit Zurücklegen" in t


def test_verdict_zwei_beide_vergleiche():
    k, t = E.verdict_zwei(E.zwei_funktionen(2, 50, 0))
    assert "R² 0,9965" in t and "Menge 3 mit 50 Instanzen" in t and "fünf getrennte Mengen" in t
    k, t = E.verdict_zwei(E.zwei_funktionen(0, 150, 1))
    assert t.startswith("Pfadabhängig gegen interventional auf demselben Baum")


def test_verdict_modelle_nennt_klasse_hintergrund_menge_und_rang_1_je_menge():
    t = E.verdict_modelle(E.modelle(C.KLASSE_NETZ, 2, 1))[1]
    assert "Kleines Netz" in t and "Hintergrund derselben Art" in t and "Menge 2 mit 40 Instanzen" in t and "wechselt mit der Instanzmenge und mit ihm die Trefferzahl" in t
    t = E.verdict_modelle(E.modelle(C.KLASSE_BOOSTING, 0, 0))[1]
    assert "fünf Mengen zu 40 Instanzen" in t and "Hintergrund Bevölkerung" in t


def test_verdict_zwilling_randfaelle_rho_null_und_q_halb():
    assert "kein Zwilling" in E.verdict_zwilling(0, 0.0, 0.9)[1]
    assert "kein Zwilling" not in E.verdict_zwilling(0, 0.5, 0.9)[1]
    assert "kein Zwilling" in E.verdict_zwilling(1, 0.5, 0.5)[1]
    assert "0,0 %" in E.verdict_zwilling(1, 0.5, 0.5)[1]
    assert "41,8 %" in E.verdict_zwilling(0, 0.9, 0.9)[1] and "37,0 %" in E.verdict_zwilling(1, 0.9, 0.9)[1]


def test_verdict_mini_ist_eine_warnung_mit_kippender_rangfolge():
    k, t = E.verdict_mini()
    assert k == "warning" and "kippt" in t and "17/24 gegen 9/8" in t


def _app():
    at = AppTest.from_file(APP, default_timeout=int(os.environ.get("APPTEST_TIMEOUT", "300")))
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at):
    return " ".join(m.value for m in list(at.success) + list(at.info) + list(at.warning) + list(at.error))


@pytest.mark.parametrize("name,erwartet", [
    ("Eine Zeile, vier Antworten", ["Zeile 0 der Menge 1, Hintergrund Bevölkerung", "Random Forest (R²", "Art vor Vorlauf in"]),
    ("Zwei Funktionen, gleiche Daten", ["Pfadabhängig gegen interventional auf demselben Baum"]),
    ("Zwilling (Gauß)", ["Korrelation ρ = 0,900", "41,8 %"]),
    ("Mini-Tabelle (Zoll)", ["Mini-Tabelle (Zoll nur bei Import)", "17/24 gegen 9/8"]),
    ("Boosting", ["Boosting (R²"]),
    ("Netz", ["Kleines Netz (R²"]),
    ("Hintergrund derselben Art", ["nur Zeilen der Art", "Hintergrund derselben Art"]),
])
def test_preset_zeigt_die_erwartete_verdikt_zeile(name, erwartet):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception
    alle = _texte(at)
    for teil in erwartet:
        assert teil in alle, (name, teil)


def test_standard_hat_keine_fehlermeldung_und_zeigt_keine_mini_tabelle():
    at = _app()
    assert not list(at.error) and "Mini-Tabelle (Zoll nur bei Import)" not in _texte(at)
