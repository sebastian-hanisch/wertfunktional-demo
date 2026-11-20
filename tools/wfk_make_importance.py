"""Rechnet die Wahrheit je Funktional (interventional, konditional, interventional mit Hintergrund derselben Art) der wahren Mitte vor und schreibt wfk_truth.json (eingefroren).

Aufruf (Test-venv, OMP_NUM_THREADS=1), im Ordner der Demo:
  python -X utf8 tools/wfk_make_importance.py            schreibt wfk_truth.json (etwa 2 Minuten)
  python -X utf8 tools/wfk_make_importance.py --check    rechnet neu und vergleicht (Toleranz 1e-9; die Gleitkomma-Letztstelle hängt von der Plattform ab)

Inhalt: die exakte Bevölkerung (1 224 gewichtete Stützpunkte, Wahrheit je Stützpunkt), Menge 1 (300 Zeilen, Experiment 1), fünf Mengen zu 150 Zeilen (Experiment 2) und fünf Mengen
zu 40 Zeilen (Experiment 3, auch mit Hintergrund derselben Art). Die Zeilen selbst werden aus den Zufalls-Seeds neu erzeugt; die Datei hält eine Prüfsumme je Menge."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import wfk_constants as C  # noqa: E402
import wfk_generator as G  # noqa: E402
import wfk_pop as P  # noqa: E402

DATEI = ROOT / "wfk_truth.json"
STAND = "2026-11-19"


def pruefsumme(X) -> float:
    return round(float(np.sum(X * np.arange(1, X.shape[1] + 1))), 6)


def runden(a, nd=10):
    return [[round(float(x), nd) for x in row] for row in np.asarray(a)[:, :C.D_WIRKUNG]]


def satz(seed, n, Xp, wp, mit_art=False):
    Xe = G.zeilenmenge(seed, n)
    pT, pC = P.wahrheit_je_funktional(G.wahrheit, Xe, Xp, wp)
    d = {"seed": seed, "n": n, "pruefsumme": pruefsumme(Xe), "phi_interventional": runden(pT), "phi_konditional": runden(pC)}
    if mit_art:
        d["phi_interventional_art"] = runden(P.shapley_aus_werten(P.werte_interventional_art(G.wahrheit, Xe, Xp, wp)))
    return d


def rechnen():
    Xp, wp = P.bevoelkerung()
    t = time.time()
    pT, pC = P.wahrheit_je_funktional(G.wahrheit, Xp, Xp, wp)
    print(f"Bevölkerung: {time.time() - t:.0f} s")
    e1 = satz(C.SEED_ZEILEN, C.N_ZEILEN, Xp, wp)
    e2 = [satz(s, C.E2_N, Xp, wp) for s in C.E2_SEEDS]
    e3 = [satz(s, C.E3_N, Xp, wp, mit_art=True) for s in C.E3_SEEDS]
    return {
        "meta": {
            "quelle": "tools/wfk_make_importance.py (wfk_pop.wahrheit_je_funktional, wahre Mitte wfk_generator.wahrheit)",
            "stand": STAND,
            "felder": {
                "bevoelkerung": "1 224 Stützpunkte (Gauß-Legendre 24 je Vorlauf-Intervall); Shapley-Wert der wahren Mitte, Spalten Art, Kunde, Zoll, Wochenende, Vorlauf (Gewicht ist Nullspieler)",
                "e1": "Menge 1: 300 natürliche Zeilen (Zufalls-Seed 4242), Wahrheit je Funktional mit der Bevölkerung als Hintergrund",
                "e2": "fünf Mengen zu 150 Zeilen (Zufalls-Seeds 7777 bis 7781)",
                "e3": "fünf Mengen zu 40 Zeilen (Zufalls-Seeds 4242, 5301 bis 5304), zusätzlich interventional mit Hintergrund derselben Art",
            },
        },
        "bevoelkerung": {"phi_interventional": runden(pT), "phi_konditional": runden(pC)},
        "e1": e1, "e2": e2, "e3": e3,
    }


def vergleichen(neu, alt, tol=1e-9):
    fehler = []
    for key in ("phi_interventional", "phi_konditional"):
        d = np.max(np.abs(np.array(neu["bevoelkerung"][key]) - np.array(alt["bevoelkerung"][key])))
        if d > tol:
            fehler.append(f"Bevölkerung {key}: größter Abstand {d:.2e}")
    paare = [(neu["e1"], alt["e1"])] + list(zip(neu["e2"], alt["e2"])) + list(zip(neu["e3"], alt["e3"]))
    for a, b in paare:
        if a["seed"] != b["seed"] or abs(a["pruefsumme"] - b["pruefsumme"]) > 1e-6:
            fehler.append(f"Menge {a['seed']}: Zeilen weichen ab (Prüfsumme)")
        for key in ("phi_interventional", "phi_konditional", "phi_interventional_art"):
            if key in a:
                d = np.max(np.abs(np.array(a[key]) - np.array(b[key])))
                if d > tol:
                    fehler.append(f"Menge {a['seed']} (n = {a['n']}) {key}: größter Abstand {d:.2e}")
    return fehler


if __name__ == "__main__":
    neu = rechnen()
    if "--check" in sys.argv:
        f = vergleichen(neu, json.loads(DATEI.read_text(encoding="utf-8")))
        print("\n".join(f) if f else "wfk_truth.json stimmt mit der Neurechnung überein (Toleranz 1e-9)")
        sys.exit(1 if f else 0)
    DATEI.write_text(json.dumps(neu, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"geschrieben: {DATEI} ({DATEI.stat().st_size / 1e3:.0f} kB)")
