"""Rechnet den pfadabhängigen Beitrag des Ersatzbaums (aus wfk_models.json) für die fünf Mengen zu 150 Zeilen vor und schreibt wfk_pfad.json (eingefroren).

Aufruf (Test-venv, OMP_NUM_THREADS=1), im Ordner der Demo, nachdem tools/wfk_make_models.py den Ersatzbaum geschrieben hat:
  python -X utf8 tools/wfk_make_pfad.py            schreibt wfk_pfad.json (einige Sekunden)
  python -X utf8 tools/wfk_make_pfad.py --check    rechnet neu und vergleicht (Toleranz 1e-9)"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import wfk_constants as C  # noqa: E402
import wfk_generator as G  # noqa: E402
import wfk_pop as P  # noqa: E402
import wfk_tree as TR  # noqa: E402

DATEI = ROOT / "wfk_pfad.json"


def rechnen():
    baum = TR.Tree.aus_dict(json.loads((ROOT / "wfk_models.json").read_text(encoding="utf-8"))["ersatzbaum"]["baum"])
    X = np.vstack([G.zeilenmenge(s, C.E2_N) for s in C.E2_SEEDS])
    phi = P.shapley_aus_werten(TR.pfad_werte(baum, X))
    return {"meta": {"quelle": "tools/wfk_make_pfad.py (wfk_tree.pfad_werte auf dem Ersatzbaum aus wfk_models.json)", "stand": "2026-11-19",
                     "felder": "phi_pfadabhaengig: (750, 5), die fünf Mengen zu 150 Zeilen hintereinander (Zufalls-Seeds 7777 bis 7781)"},
            "phi_pfadabhaengig": [[round(float(x), 10) for x in row] for row in phi]}


if __name__ == "__main__":
    neu = rechnen()
    if "--check" in sys.argv:
        d = np.max(np.abs(np.array(neu["phi_pfadabhaengig"]) - np.array(json.loads(DATEI.read_text(encoding="utf-8"))["phi_pfadabhaengig"])))
        print(f"größter Abstand {d:.2e}" + ("" if d < 1e-9 else " (ZU GROSS)"))
        sys.exit(0 if d < 1e-9 else 1)
    DATEI.write_text(json.dumps(neu, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"geschrieben: {DATEI} ({DATEI.stat().st_size / 1e3:.0f} kB)")
