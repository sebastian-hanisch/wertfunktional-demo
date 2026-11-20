"""Gleiches Szenario in zwei Prozessen mit verschiedenem PYTHONHASHSEED: gleiche Zahlen (keine Abhängigkeit von Mengen-Reihenfolgen)."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = (
    "import sys; sys.path.insert(0, sys.argv[1]); import wfk_evaluation as E; "
    "ez = E.eine_zeile(7, 2); zf = E.zwei_funktionen(1, 50, 0); m = E.modelle(1, 0, 0); z = E.ziehungen(40); "
    "print(repr((round(float(ez.phi_i[0]), 12), round(zf.summe, 12), round(m.gleich_i[0], 12), z.art_vor_vorlauf_i, E.verdict_modelle(m)[1])))"
)


def _lauf(hashseed: str) -> str:
    env = {"PYTHONHASHSEED": hashseed, "OMP_NUM_THREADS": "1", "PATH": "", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    return subprocess.run([sys.executable, "-X", "utf8", "-c", CODE, str(ROOT)], capture_output=True, text=True, env=env, timeout=300, check=True).stdout


def test_zwei_prozesse_mit_verschiedenem_hashseed_rechnen_dasselbe():
    assert _lauf("1") == _lauf("2")
