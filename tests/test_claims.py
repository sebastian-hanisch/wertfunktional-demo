"""Claims: jede Zahl der README steht hier. Je Eintrag: der Wortlaut in der README (Fragment) und die Rechnung, die genau diesen Wortlaut erzeugt (dieselben Funktionen wie die App).

Die Modelle (Experiment 3) sind eingefroren (wfk_models.json, scikit-learn); das Neu-Trainieren prüft tests/test_evaluation.py als Band. Zahlen, die nur eine Messung dieser README sind
(Rechenzeiten, Erwartungen des Plans), stehen in der Tabelle „Zahl → Test“ als „Messung dieser README, nicht im Test“."""

import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import wfk_constants as C
import wfk_evaluation as E
import wfk_texts as TX
import wfk_truth as T
import wfk_twin as TW

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
f = E.fmt


def _bis(t, nd=2, s=1.0):
    return f"{f(t[0] * s, nd)} bis {f(t[1] * s, nd)}"


CLAIMS = []


def claim(fragment, fn):
    CLAIMS.append((fragment, fn))


wb = lambda: E.verdict_wichtigkeit(E.wichtigkeit_bevoelkerung())[1]
# --- Experiment 1
claim("Art 0,651 interventional gegen 0,168 konditional, Vorlauf 0,561 gegen 0,275", lambda: "Art 0,651 interventional gegen 0,168 konditional, Vorlauf 0,561 gegen 0,275" if "Art 0,651 interventional gegen 0,168 konditional, Vorlauf 0,561 gegen 0,275" in wb() else wb())
claim("Rang 1 wechselt von Art zu Vorlauf", lambda: "Rang 1 wechselt von Art zu Vorlauf" if "Rang 1 wechselt von Art zu Vorlauf" in wb() else wb())
claim("Rangkorrelation der beiden Wichtigkeiten 0,70; Betragsmaß: Summe der Abweichungen je Zeile 1,235",
      lambda: f"Rangkorrelation der beiden Wichtigkeiten {f(E.wichtigkeit_bevoelkerung().rangkorrelation)}; Betragsmaß: Summe der Abweichungen je Zeile {f(E.wichtigkeit_bevoelkerung().abstand, 3)}")
claim("Vorlauf 0,140, Kunde 0,066, Art 0,037, Zoll 0,032, Wochenende 0,008",
      lambda: "Vorlauf {}, Kunde {}, Art {}, Zoll {}, Wochenende {}".format(*(f(float(T.loo_exakt()[j]), 3) for j in (4, 1, 0, 2, 3))))
for _m, _a, _b, _c in ((20, "84,5 %", "0,558 bis 0,750", "93,5 %"), (40, "92,4 %", "0,585 bis 0,720", "98,3 %"), (100, "98,8 %", "0,608 bis 0,694", "100,0 %"), (150, "99,7 %", "0,616 bis 0,687", "100,0 %")):
    claim(f"| {_m} | {_a} ({_b}) | {_c} |", lambda m=_m: (lambda z: f"| {m} | {f(z.art_vor_vorlauf_i * 100, 1)} % ({f(z.art_q05, 3)} bis {f(z.art_q95, 3)}) | {f(z.vorlauf_vor_art_k * 100, 1)} % |")(E.ziehungen(m)))
# --- Experiment 2
claim("mit 462 Blättern, R² 0,9965", lambda: f"mit {T.ersatzbaum().n_blaetter} Blättern, R² {f(T.ersatzbaum().r2, 4)}")
claim("um 1,31 bis 1,39 auseinander (Art allein 0,64 bis 0,69)", lambda: (lambda z: f"um {_bis(z.spanne_summe)} auseinander (Art allein {_bis(z.spanne_art)})")(E.zwei_funktionen(0, 150, 0)))
claim("auf demselben Baum**: 0,061 bis 0,066", lambda: f"auf demselben Baum**: {_bis(E.zwei_funktionen(0, 150, 1).spanne_summe, 3)}")
# --- Experiment 3: Tabelle
def _zeile(k):
    m = E.modelle(k, 0, 0)
    pfad = f"{_bis(m.int_gegen_pfad, 3)} / {_bis(m.kond_gegen_pfad, 3)}" if m.int_gegen_pfad else "nicht anwendbar"
    return (f"| {C.KLASSEN[k]} ({f(m.r2, 3)}) | {_bis(m.gleich_i)} / {_bis(m.gleich_k)} | {_bis(m.int_gegen_wahrheit_k)} | {_bis(m.int_gegen_kond)} / {pfad} | "
            f"{_bis(m.gleiches_top, 0, 100)} % | {_bis(m.rangkorr)} |")


for _k, _row in enumerate((
    "| Boosting (0,930) | 1,03 bis 1,18 / 0,15 bis 0,17 | 0,31 bis 0,36 | 0,24 bis 0,28 / 0,061 bis 0,075 / 0,247 bis 0,278 | 76 bis 92 % | 0,92 bis 0,97 |",
    "| Random Forest (0,911) | 1,15 bis 1,34 / 0,18 bis 0,21 | 0,41 bis 0,43 | 0,30 bis 0,33 / 0,083 bis 0,086 / 0,293 bis 0,334 | 66 bis 81 % | 0,80 bis 0,95 |",
    "| Kleines Netz (0,922) | 0,44 bis 0,50 / 0,15 bis 0,18 | 1,53 bis 1,65 | 1,48 bis 1,58 / nicht anwendbar | 32 bis 54 % | 0,70 bis 0,92 |")):
    claim(_row, lambda k=_k: _zeile(k))
claim("(Boosting und Wald: 1,03 bis 1,34 gegen 0,31 bis 0,43), beim Netz ist es umgekehrt (0,44 bis 0,50 gegen 1,53 bis 1,65)",
      lambda: "(Boosting und Wald: {} bis {} gegen {} bis {}), beim Netz ist es umgekehrt ({} gegen {})".format(
          f(min(E.modelle(0, 0, 0).gleich_i[0], E.modelle(1, 0, 0).gleich_i[0])), f(max(E.modelle(0, 0, 0).gleich_i[1], E.modelle(1, 0, 0).gleich_i[1])),
          f(min(E.modelle(0, 0, 0).int_gegen_wahrheit_k[0], E.modelle(1, 0, 0).int_gegen_wahrheit_k[0])), f(max(E.modelle(0, 0, 0).int_gegen_wahrheit_k[1], E.modelle(1, 0, 0).int_gegen_wahrheit_k[1])),
          _bis(E.modelle(2, 0, 0).gleich_i), _bis(E.modelle(2, 0, 0).int_gegen_wahrheit_k)))
claim("bei allen drei Klassen nah an der konditionalen Wahrheit (0,15 bis 0,21)",
      lambda: "bei allen drei Klassen nah an der konditionalen Wahrheit ({} bis {})".format(f(min(E.modelle(k, 0, 0).gleich_k[0] for k in range(3))), f(max(E.modelle(k, 0, 0).gleich_k[1] for k in range(3)))))
claim("Boosting 0, 0, 8, 8, 2; Random Forest 0, 1, 8, 8, 1; Netz 5, 8, 8, 8, 8",
      lambda: "Boosting {}; Random Forest {}; Netz {}".format(*(", ".join(str(t) for t in E.modelle(k, 0, 0).rang1_treffer) for k in range(3))))
claim("In den Mengen 3 und 4 ist der Rang 1 der Wahrheit der Vorlauf, in den Mengen 1, 2 und 5 die Art",
      lambda: "In den Mengen 3 und 4 ist der Rang 1 der Wahrheit der Vorlauf, in den Mengen 1, 2 und 5 die Art" if E.modelle(0, 0, 0).wahr_rang1 == (0, 0, 4, 4, 0) else str(E.modelle(0, 0, 0).wahr_rang1))
claim("(Vorlauf 0,149, Kunde 0,055 und 0,046); beim Netz liegt die Art an zweiter Stelle der Rangfolge (0,061) und das Wochenende bei −0,001",
      lambda: "(Vorlauf {}, Kunde {} und {}); beim Netz liegt die Art an zweiter Stelle der Rangfolge ({}) und das Wochenende bei {}".format(
          f(float(E.modelle(0, 0, 0).loo[4]), 3), f(float(E.modelle(0, 0, 0).loo[1]), 3), f(float(E.modelle(1, 0, 0).loo[1]), 3), f(float(E.modelle(2, 0, 0).loo[0]), 3),
          f(float(E.modelle(2, 0, 0).loo[3]), 3).replace("-", "−")) if E.modelle(2, 0, 0).loo_rang[1] == 0 else "FALSCH")
claim("Boosting 0,136 bis 0,157, Wald 0,158 bis 0,191, Netz 0,138 bis 0,169",
      lambda: "Boosting {}, Wald {}, Netz {}".format(*(_bis(E.modelle(k, 0, 1).gleich_i, 3) for k in range(3))))
# --- Experiment 4 und Mini-Tabelle
claim("bei ρ = 0,9 bekommt der Zwilling konditional 41,8 % der Wichtigkeit, bei ρ = 0,99 49,0 %",
      lambda: "bei ρ = 0,9 bekommt der Zwilling konditional {} % der Wichtigkeit, bei ρ = 0,99 {} %".format(
          re.search(r"Zwilling ([\d,]+) % der Wichtigkeit", E.verdict_zwilling(0, 0.9, 0.9)[1]).group(1), re.search(r"Zwilling ([\d,]+) % der Wichtigkeit", E.verdict_zwilling(0, 0.99, 0.9)[1]).group(1)))
_q = Fraction(9, 10)
claim("konditional (E|φ_1|, E|φ_2|) = (17/50, 1/5), Anteil des Zwillings 10/27, interventional (1/2, 0)",
      lambda: f"konditional (E|φ_1|, E|φ_2|) = ({TW.binaer_wichtigkeit_kond(_q)[0]}, {TW.binaer_wichtigkeit_kond(_q)[1]}), Anteil des Zwillings {TW.binaer_anteil_kond(_q)}, interventional (1/2, 0)")
_mini = lambda: E.verdict_mini()[1]
claim("Grundwert 5/2, Vorhersage 5", lambda: "Grundwert 5/2, Vorhersage 5" if "Grundwert 5/2, Vorhersage 5" in _mini() else _mini())
claim("interventional (3/2, 1), konditional (3/4, 7/4),\npfadabhängig auf Baum 1 (7/4, 3/4), auf Baum 2 (1/2, 2)", lambda: "interventional (3/2, 1), konditional (3/4, 7/4),\npfadabhängig auf Baum 1 (7/4, 3/4), auf Baum 2 (1/2, 2)" if "interventional (3/2, 1), konditional (3/4, 7/4), pfadabhängig auf Baum 1 (7/4, 3/4), auf Baum 2 (1/2, 2)" in _mini() else "FALSCH")
claim("Baum 1 (11/8, 1/2), Baum 2 (17/24, 9/8)", lambda: "Baum 1 (11/8, 1/2), Baum 2 (17/24, 9/8)" if "Baum 1 11/8 gegen 1/2, Baum 2 17/24 gegen 9/8" in _mini() else "FALSCH")
claim("Leave-one-out (Verlustanstieg): Art 2/3, Zoll 1/2", lambda: "Leave-one-out (Verlustanstieg): Art 2/3, Zoll 1/2" if "Leave-one-out (Verlustanstieg): Art 2/3, Zoll 1/2" in _mini() else "FALSCH")


@pytest.mark.parametrize("fragment,fn", CLAIMS, ids=[c[0][:60] for c in CLAIMS])
def test_zahl_der_readme_stimmt_mit_der_rechnung_ueberein(fragment, fn):
    assert fragment in README, f"Wortlaut nicht in der README: {fragment}"
    assert fn() == fragment, f"README sagt {fragment!r}, die Rechnung gibt {fn()!r}"


def test_feste_texte_der_app_stehen_wortgleich_in_der_readme():
    assert TX.ASSUMPTIONS_MARKDOWN.strip() in README
    assert TX.KONVENTION_MARKDOWN.strip() in README


def test_testzahl_der_readme_stimmt_mit_der_suite_ueberein():
    out = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "--collect-only", "-q", "tests", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    gezaehlt = int(re.search(r"(\d+) tests? collected", out).group(1))
    m = re.search(r"(\d+) Tests", README)
    assert m and int(m.group(1)) == gezaehlt, f"README nennt {m and m.group(1)}, die Suite hat {gezaehlt}"


def test_readme_hat_die_pflichtabschnitte_und_den_hinweis_auf_nicht_gemessenes():
    for abschnitt in ("## Warum dieses Problem", "## Befunde", "## Befunde und Korrekturen", "## Ehrliche Grenzen", "## Tests", "## Dateistruktur", "## Bewusst nicht umgesetzt", "## Lokal ausführen", "## Quellen"):
        assert abschnitt in README, abschnitt
    assert "🚧" in README and "Nicht gemessen" in README


def test_readme_ohne_verbotene_worte():
    assert not re.search(r"Saatwert|Schwanz|\b(trägt|tragen|trage)\b", README)


def test_readme_nennt_zaehleinheiten_und_zeilenmenge():
    for wort in ("Zufalls-Seeds", "Zeilenmenge", "exakte Bevölkerung", "Instanzen", "Modellauswertungen"):
        assert wort in README


def test_zitat_fuer_stueck_1_zahlen_stehen_in_der_tabelle():
    """Stück 1 (beitrag-linear-demo) zitiert diese Spannen; hier die Quelle: Boosting 1,03 bis 1,18 / 0,31 bis 0,36, Netz 0,44 bis 0,50 / 1,53 bis 1,65."""
    b, n = E.modelle(C.KLASSE_BOOSTING, 0, 0), E.modelle(C.KLASSE_NETZ, 0, 0)
    assert (_bis(b.gleich_i), _bis(b.int_gegen_wahrheit_k), _bis(n.gleich_i), _bis(n.int_gegen_wahrheit_k)) == ("1,03 bis 1,18", "0,31 bis 0,36", "0,44 bis 0,50", "1,53 bis 1,65")
