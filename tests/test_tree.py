"""Pfadabhängiges Wertfunktional (Blattanteile): vektorisierte Blattrechnung gegen Knoten-Rekursion (Orakel, Algorithmus 1), v(alle) = Vorhersage, Blattanteile gegen scikit-learn,
Mini-Bäume gegen die Brüche von Hand, Ensemble (Boosting, Wald) und der eingefrorene Ersatzbaum."""

import itertools

import numpy as np
import pytest

import wfk_constants as C
import wfk_generator as G
import wfk_hand as H
import wfk_pop as P
import wfk_tree as TR
import wfk_truth as T

MASKEN = [[(k >> j) & 1 for j in range(5)] for k in range(32)]


def rekursion(tree, x, S):
    """Knoten-Rekursion (anderer Rechenweg): Merkmal in S folgt x, sonst Mischung nach Trainingsanteilen."""
    def rec(n):
        if tree.left[n] < 0:
            return tree.value[n]
        f = tree.feature[n]
        if f in S:
            return rec(tree.left[n] if x[f] <= tree.threshold[n] else tree.right[n])
        l, r = tree.left[n], tree.right[n]
        return (tree.cover[l] * rec(l) + tree.cover[r] * rec(r)) / tree.cover[n]
    return rec(0)


@pytest.fixture(scope="module")
def sk_baum():
    sk = pytest.importorskip("sklearn.tree")
    X, y = G.daten(600, 3)
    d = sk.DecisionTreeRegressor(min_samples_leaf=15, random_state=0).fit(X[:, :5], y)
    return d, TR.Tree.von_sklearn(d.tree_), X


def test_blattrechnung_gleich_knotenrekursion_fuer_alle_32_koalitionen(sk_baum):
    _, t, X = sk_baum
    Z = G.zeilenmenge(4242, 12)[:, :5]
    v = TR.pfad_werte(t, Z)
    for i in range(len(Z)):
        for k, m in enumerate(MASKEN):
            S = {j for j in range(5) if m[j]}
            assert v[i, k] == pytest.approx(rekursion(t, Z[i], S), abs=1e-12)


def test_v_alle_ist_die_vorhersage_und_v_leer_das_mittel_nach_blattanteilen(sk_baum):
    d, t, X = sk_baum
    Z = G.zeilenmenge(5101, 40)[:, :5]
    v = TR.pfad_werte(t, Z)
    assert np.max(np.abs(v[:, 31] - d.predict(Z))) < 1e-12
    assert np.max(np.abs(t.vorhersage(Z) - d.predict(Z))) < 1e-12
    blatt = t.left < 0
    assert np.allclose(v[:, 0], (t.cover[blatt] * t.value[blatt]).sum() / t.cover[0], atol=1e-12)
    assert np.allclose(v[:, 0], v[0, 0])                                          # v(leer) ist für alle Zeilen gleich


def test_blattanteile_gleich_scikit_learn(sk_baum):
    d, t, X = sk_baum
    assert np.array_equal(t.cover, d.tree_.weighted_n_node_samples)
    assert t.cover[0] == len(X)
    inner = t.left >= 0
    assert np.allclose(t.cover[inner], t.cover[t.left[inner]] + t.cover[t.right[inner]])
    assert t.n_blaetter == d.get_n_leaves()


def test_effizienz_und_nullspieler_im_baum(sk_baum):
    _, t, X = sk_baum
    Z = G.zeilenmenge(5102, 20)[:, :5]
    v = TR.pfad_werte(t, Z)
    phi = P.shapley_aus_werten(v)
    assert np.max(np.abs(phi.sum(axis=1) - (v[:, 31] - v[:, 0]))) < 1e-12
    benutzt = set(int(f) for f in t.feature if f >= 0)
    for j in set(range(5)) - benutzt:
        assert np.all(np.abs(phi[:, j]) < 1e-12)                                  # vom Baum nicht benutztes Merkmal: Beitrag 0 (pfadabhängig)
    assert np.max(np.abs(phi - P.shapley_aus_permutationen(v))) < 1e-12           # Koalitionsformel gegen alle Permutationen


def test_baum_mit_einem_blatt_gibt_ueberall_beitrag_null():
    t = TR.Tree(np.array([-1]), np.array([0.0]), np.array([-1]), np.array([-1]), np.array([2.5]), np.array([100.0]))
    v = TR.pfad_werte(t, G.zeilenmenge(4242, 5)[:, :5])
    assert np.all(v == 2.5) and np.all(P.shapley_aus_werten(v) == 0.0) and t.n_blaetter == 1


def test_dict_rundreise_erhaelt_den_baum(sk_baum):
    _, t, _ = sk_baum
    t2 = TR.Tree.aus_dict(t.als_dict(nd=12))
    Z = G.zeilenmenge(4242, 30)[:, :5]
    assert np.max(np.abs(t.vorhersage(Z) - t2.vorhersage(Z))) < 1e-9
    assert np.max(np.abs(TR.pfad_werte(t, Z) - TR.pfad_werte(t2, Z))) < 1e-9


def _mini_baum(b, spalten=2):
    """Handbaum (Brüche) als Tree-Felder: Merkmal 0 = Art, 1 = Zoll; x <= 0,5 geht nach links (Merkmal = 0)."""
    felder = {"feature": [], "threshold": [], "left": [], "right": [], "value": [], "cover": []}

    def rec(n):
        i = len(felder["feature"])
        for k in felder:
            felder[k].append(None)
        if "wert" in n:
            felder["feature"][i], felder["threshold"][i], felder["left"][i], felder["right"][i] = -1, 0.0, -1, -1
            felder["value"][i], felder["cover"][i] = float(n["wert"]), float(n["cover"])
        else:
            felder["feature"][i], felder["threshold"][i], felder["value"][i], felder["cover"][i] = n["f"], 0.5, 0.0, float(n["cover"])
            felder["left"][i] = rec(n["links"])
            felder["right"][i] = rec(n["rechts"])
        return i

    rec(b)
    return TR.Tree(*(np.array(felder[k], dtype=float if k in ("threshold", "value", "cover") else int) for k in ("feature", "threshold", "left", "right", "value", "cover")))


@pytest.mark.parametrize("baum,erwartet", [(H.BAUM1, (1.75, 0.75)), (H.BAUM2, (0.5, 2.0))])
def test_mini_baeume_numpy_blattrechnung_gleich_den_bruechen_von_hand(baum, erwartet):
    t = _mini_baum(baum)
    x = np.array([[1.0, 1.0, 0, 0, 0]])
    v = TR.pfad_werte(t, x)
    # Koalitionswerte über die Merkmale 0 (Art) und 1 (Zoll); die drei übrigen Merkmale kommen im Baum nicht vor: Beitrag 0
    phi = P.shapley_aus_werten(v)[0]
    assert phi[0] == pytest.approx(erwartet[0], abs=1e-12) and phi[1] == pytest.approx(erwartet[1], abs=1e-12)
    assert np.allclose(phi[2:], 0, atol=1e-12)
    for k in (0, 1, 2, 3):
        S = frozenset(j for j in (0, 1) if (k >> j) & 1)
        assert v[0, k] == pytest.approx(float(H.baum_pfad_wert(baum, (1, 1), S)), abs=1e-12)


def test_ensemble_boosting_und_wald_v_alle_gleich_modellvorhersage():
    ens = pytest.importorskip("sklearn.ensemble")
    X, y = G.daten(400, 2)
    Z = G.zeilenmenge(4242, 10)[:, :5]
    gb = ens.GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=0).fit(X[:, :5], y)
    v = TR.pfad_werte_ensemble([TR.Tree.von_sklearn(e[0].tree_) for e in gb.estimators_], 0.1, float(gb.init_.constant_.ravel()[0]), Z)
    assert np.max(np.abs(v[:, 31] - gb.predict(Z))) < 1e-9
    rf = ens.RandomForestRegressor(n_estimators=12, min_samples_leaf=8, random_state=0, n_jobs=1).fit(X[:, :5], y)
    v = TR.pfad_werte_ensemble([TR.Tree.von_sklearn(e.tree_) for e in rf.estimators_], 1.0 / 12, 0.0, Z)
    assert np.max(np.abs(v[:, 31] - rf.predict(Z))) < 1e-9


def test_eingefrorener_ersatzbaum_ist_ein_baum_auf_der_wahren_mitte():
    er = T.ersatzbaum()
    Xt, mut = G.testsatz()
    assert 440 <= er.n_blaetter <= 520
    r2 = 1 - np.sum((er.baum.vorhersage(Xt) - mut) ** 2) / np.sum((mut - mut.mean()) ** 2)
    assert r2 == pytest.approx(er.r2, abs=1e-6) and r2 > 0.995
    assert er.baum.cover[0] == C.ERSATZ_N
    blatt = er.baum.left < 0
    assert er.baum.cover[blatt].min() >= C.ERSATZ_MIN_LEAF                         # mindestens 100 Zeilen je Blatt
    assert er.baum.value[blatt].min() > -2 and np.all(np.isfinite(er.baum.threshold))
