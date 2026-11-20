"""Trainiert Boosting, Random Forest und ein kleines Netz (scikit-learn, nur Entwicklungs-venv) und rechnet ihre Beiträge je Wertfunktional vor; schreibt wfk_models.json (eingefroren).
Dazu der Ersatzbaum (Baum auf der wahren Mitte) und die exakten Leave-one-out-Werte des besten Vorhersagers.

Aufruf (Test-venv mit scikit-learn, OMP_NUM_THREADS=1), im Ordner der Demo:
  python -X utf8 tools/wfk_make_models.py            schreibt wfk_models.json (einige Minuten)
  python -X utf8 tools/wfk_make_models.py --check    rechnet neu und vergleicht (Toleranz 1e-6 für Beiträge: Trainingsläufe hängen von der Bibliotheksversion und der Plattform ab)

Experiment 3: 8 Zufalls-Seeds (Trainingsdatensätze mit n = 1 000, nur die fünf Merkmale mit Wirkung), fünf getrennte Instanzmengen zu je 40 Zeilen (200 Instanzen je Modell),
Hintergrund: die Bevölkerung des Generators. Je Modell: interventional, konditional (exakt unter der Verteilung des Generators), pfadabhängig (nur Bäume), interventional mit
Hintergrund derselben Art, dazu Leave-one-out als Verlustanstieg (mittlere quadratische Abweichung gegen die wahre Mitte auf 5 000 frischen Zeilen, Modell ohne Merkmal minus Modell mit)."""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import wfk_constants as C  # noqa: E402
import wfk_generator as G  # noqa: E402
import wfk_pop as P  # noqa: E402
import wfk_tree as TR  # noqa: E402

DATEI = ROOT / "wfk_models.json"
STAND = "2026-11-19"
N_PAAR = 5 * C.E3_N


def netz_merkmale(Z, keep):
    """Eingabe des Netzes: Art und Kunde als Dummies, Zoll, Wochenende, Vorlauf / 5 (nur die Merkmale in `keep`)."""
    spalten = []
    if 0 in keep:
        spalten.append(np.eye(4)[Z[:, 0].astype(int)])
    if 1 in keep:
        spalten.append(np.eye(12)[Z[:, 1].astype(int)])
    for j, s in ((2, 1.0), (3, 1.0), (4, 5.0)):
        if j in keep:
            spalten.append((Z[:, j] / s)[:, None])
    return np.column_stack(spalten)


def trainieren(klasse, X5, y, seed, keep=(0, 1, 2, 3, 4)):
    """(vorhersage(Z5) -> (n,), Modellobjekt). Hyperparameter fest je Klasse."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    keep = list(keep)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if klasse == C.KLASSE_BOOSTING:
            m = GradientBoostingRegressor(n_estimators=C.BOOST_TREES, max_depth=C.BOOST_DEPTH, learning_rate=C.BOOST_LR, random_state=seed).fit(X5[:, keep], y)
            return (lambda Z: m.predict(Z[:, keep])), m
        if klasse == C.KLASSE_FOREST:
            m = RandomForestRegressor(n_estimators=C.FOREST_TREES, min_samples_leaf=C.FOREST_MIN_LEAF, random_state=seed, n_jobs=1).fit(X5[:, keep], y)
            return (lambda Z: m.predict(Z[:, keep])), m
        m = MLPRegressor(hidden_layer_sizes=(C.NET_HIDDEN,), activation="tanh", max_iter=2000, tol=1e-6, random_state=seed).fit(netz_merkmale(X5, keep), y)
        return (lambda Z: m.predict(netz_merkmale(Z, keep))), m


def pfad_modell(klasse, m, X):
    if klasse == C.KLASSE_BOOSTING:
        baeume = [TR.Tree.von_sklearn(e[0].tree_) for e in m.estimators_]
        return TR.pfad_werte_ensemble(baeume, C.BOOST_LR, float(m.init_.constant_.ravel()[0]), X)
    baeume = [TR.Tree.von_sklearn(e.tree_) for e in m.estimators_]
    return TR.pfad_werte_ensemble(baeume, 1.0 / len(baeume), 0.0, X)


def runden(a, nd=7):
    return np.round(np.asarray(a, float), nd).tolist()


def rechnen():
    Xp, wp = P.bevoelkerung()
    Xt, mut = G.testsatz()
    mse = lambda p: float(np.mean((p - mut) ** 2))
    Xe = np.vstack([G.zeilenmenge(s, C.E3_N) for s in C.E3_SEEDS])        # 200 Instanzen: fünf Mengen zu 40 Zeilen
    f6 = lambda g: (lambda Z: g(Z[:, :5]))
    out = {"meta": {"quelle": "tools/wfk_make_models.py (scikit-learn, Entwicklungs-venv)", "stand": STAND,
                    "felder": {"klassen": "Boosting 200 Bäume Tiefe 3, Lernrate 0,1; Random Forest 60 Bäume, mindestens 5 Zeilen je Blatt; Netz 16 tanh-Einheiten (Art und Kunde als Dummies); "
                                          "8 Zufalls-Seeds, n = 1 000, Merkmale ohne Gewicht",
                               "phi": "je Klasse, Funktional, Zufalls-Seed: (200, 5) Beiträge, Zeilen = die fünf Mengen zu 40 hintereinander",
                               "ersatzbaum": "Baum auf der wahren Mitte (100 000 natürliche Zeilen, mindestens 100 Zeilen je Blatt) mit seinem interventionalen Beitrag für die Mengen zu 150 Zeilen"}},
           "klassen": {}}
    for k, name in enumerate(C.KLASSEN):
        r2, phi = [], {"interventional": [], "konditional": [], "interventional_art": [], "pfadabhaengig": []}
        loo, t0 = [], time.time()
        for sd in range(C.N_MODELL_SEEDS):
            X, y = G.daten(C.N_TRAIN, sd)
            pred, m = trainieren(k, X[:, :5], y, sd)
            r2.append(1.0 - np.sum((pred(Xt[:, :5]) - mut) ** 2) / np.sum((mut - mut.mean()) ** 2))
            f = f6(pred)
            phi["interventional"].append(runden(P.shapley_aus_werten(P.werte_interventional(f, Xe, Xp, wp))))
            phi["konditional"].append(runden(P.shapley_aus_werten(P.werte_konditional(f, Xe))))
            phi["interventional_art"].append(runden(P.shapley_aus_werten(P.werte_interventional_art(f, Xe, Xp, wp))))
            if k != C.KLASSE_NETZ:
                v = pfad_modell(k, m, Xe)
                assert np.max(np.abs(v[:, -1] - pred(Xe[:, :5]))) < 1e-9, "v(alle) muss die Vorhersage des Modells sein"
                phi["pfadabhaengig"].append(runden(P.shapley_aus_werten(v)))
            base = mse(pred(Xt[:, :5]))
            loo.append([mse(trainieren(k, X[:, :5], y, sd, [i for i in range(5) if i != j])[0](Xt[:, :5])) - base for j in range(5)])
        if not phi["pfadabhaengig"]:
            del phi["pfadabhaengig"]
        out["klassen"][name] = {"r2": runden(r2, 6), "phi": phi, "loo": runden(loo, 6)}
        print(f"{name}: {time.time() - t0:.0f} s, R² {np.mean(r2):.3f}", flush=True)
    # Ersatzbaum
    from sklearn.tree import DecisionTreeRegressor
    Xb, _ = G.merkmale(C.SET_OFFSET + 7000, C.ERSATZ_N)
    dt = DecisionTreeRegressor(min_samples_leaf=C.ERSATZ_MIN_LEAF, random_state=0).fit(Xb[:, :5], G.wahrheit(Xb))
    tree = TR.Tree.von_sklearn(dt.tree_)
    assert np.allclose(tree.vorhersage(Xt), dt.predict(Xt[:, :5]))
    Xe2 = np.vstack([G.zeilenmenge(s, C.E2_N) for s in C.E2_SEEDS])
    fb = lambda Z: tree.vorhersage(Z)
    out["ersatzbaum"] = {"baum": tree.als_dict(), "n_blaetter": tree.n_blaetter, "r2": round(float(1 - np.sum((tree.vorhersage(Xt) - mut) ** 2) / np.sum((mut - mut.mean()) ** 2)), 6),
                         "phi_interventional": runden(P.shapley_aus_werten(P.werte_interventional(fb, Xe2, Xp, wp)))}
    # Leave-one-out des besten Vorhersagers (exakt über die Bevölkerung): Verlustanstieg = Σ w (μ − E[μ | übrige Merkmale])²
    mu_p = G.wahrheit(Xp)
    loo_exakt = []
    for j in range(5):
        mask = [i != j for i in range(5)]
        cond = P.bedingte_erwartung(G.wahrheit, Xp, mask)
        loo_exakt.append(float(wp @ (mu_p - cond) ** 2))
    out["loo_exakt"] = runden(loo_exakt, 8)
    return out


def vergleichen(neu, alt, tol=1e-6):
    fehler = []
    for name, a in neu["klassen"].items():
        b = alt["klassen"][name]
        for fn, werte in a["phi"].items():
            d = np.max(np.abs(np.array(werte) - np.array(b["phi"][fn])))
            if d > tol:
                fehler.append(f"{name} {fn}: größter Abstand {d:.2e}")
        if np.max(np.abs(np.array(a["loo"]) - np.array(b["loo"]))) > 1e-4:
            fehler.append(f"{name} Leave-one-out weicht ab")
    if np.max(np.abs(np.array(neu["ersatzbaum"]["phi_interventional"]) - np.array(alt["ersatzbaum"]["phi_interventional"]))) > tol:
        fehler.append("Ersatzbaum: interventionale Beiträge weichen ab")
    if np.max(np.abs(np.array(neu["loo_exakt"]) - np.array(alt["loo_exakt"]))) > 1e-8:
        fehler.append("Leave-one-out exakt weicht ab")
    return fehler


if __name__ == "__main__":
    neu = rechnen()
    if "--check" in sys.argv:
        f = vergleichen(neu, json.loads(DATEI.read_text(encoding="utf-8")))
        print("\n".join(f) if f else "wfk_models.json stimmt mit der Neurechnung überein")
        sys.exit(1 if f else 0)
    DATEI.write_text(json.dumps(neu, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"geschrieben: {DATEI} ({DATEI.stat().st_size / 1e3:.0f} kB)")
