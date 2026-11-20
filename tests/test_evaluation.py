"""Messfunktionen, eingefrorene Modelle (Effizienz-Konsistenz, Neu-Trainieren als Band), Vergleichswerkzeuge der Tools und die Experimente an handgesetzten Fällen."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import wfk_constants as C
import wfk_evaluation as E
import wfk_generator as G
import wfk_pop as P
import wfk_tree as TR
import wfk_truth as T

ROOT = Path(__file__).resolve().parent.parent


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------- Maße
def test_rangkorrelation_gegen_scipy_und_nan_bei_gleichen_werten():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    for _ in range(40):
        a, b = rng.integers(0, 4, 5).astype(float), rng.integers(0, 4, 5).astype(float)
        erwartet = stats.spearmanr(a, b)[0]
        ist = E.rangkorrelation(a, b)
        assert (np.isnan(erwartet) and np.isnan(ist)) or ist == pytest.approx(erwartet, abs=1e-12)


def test_abstand_wichtigkeit_und_gleiches_top_von_hand():
    a = np.array([[1.0, 0, 0, 0, 0], [0, 0, 0, 0, 2.0]])
    b = np.array([[1.0, 0, 0, 0, 0], [0, 3.0, 0, 0, 0]])
    assert E.abstand(a, b) == pytest.approx((0 + (3 + 2)) / 2)                       # Instanz 2: |0 − 3| + |2 − 0| = 5, Mittel über zwei Instanzen
    assert E.gleiches_top(a, b) == 0.5
    assert np.allclose(E.wichtigkeit(a), [0.5, 0, 0, 0, 1.0])
    assert E.wichtigkeit(a, np.array([0.75, 0.25]))[0] == pytest.approx(0.75)


# ---------------------------------------------------------------------------------------------- Experiment 1
def test_zeile_interventional_live_gleich_dem_eingefrorenen_der_bevoelkerung():
    for z in (0, 17, 299):
        ez = E.eine_zeile(z, C.HG_BEVOELKERUNG)
        assert np.max(np.abs(ez.phi_i - ez.phi_i_bev)) < 1e-9
        assert abs(ez.phi_i.sum() - (ez.mu - ez.grundwert_i)) < 1e-9                  # Effizienz
    ez = E.eine_zeile(5, C.HG_BEVOELKERUNG)
    assert abs(float(ez.phi_k.sum()) - (ez.mu - float(P.bedingte_erwartung(G.wahrheit, ez.x[None, :], [False] * 5)[0]))) < 1e-9


def test_zeile_hintergrund_derselben_art_gibt_der_art_den_beitrag_null():
    for z in (0, 3, 40, 200):
        ez = E.eine_zeile(z, C.HG_ART)
        assert abs(ez.phi_i[0]) < 1e-12                                              # die Art ist im Hintergrund konstant: kein Beitrag
        assert abs(ez.phi_i.sum() - (ez.mu - ez.grundwert_i)) < 1e-9


def test_zeile_stichproben_hintergruende_haben_die_groesse_der_option():
    for opt, n in C.HG_GROESSEN.items():
        X, w, name = E.hintergrund(opt)
        assert len(X) == n and w.sum() == pytest.approx(1.0) and str(n).replace("000", " 000") in name.replace("1 000", "1 000")
        ez = E.eine_zeile(0, opt)
        assert abs(ez.phi_i.sum() - (ez.mu - ez.grundwert_i)) < 1e-9


def test_zeile_pfadabhaengig_live_summiert_zur_vorhersage_minus_grundwert():
    ez = E.eine_zeile(11, C.HG_BEVOELKERUNG)
    assert abs(ez.phi_p.sum() - (ez.vorhersage_baum - ez.grundwert_k)) < 1e-9


def test_wichtigkeit_der_bevoelkerung_und_masse():
    wb = E.wichtigkeit_bevoelkerung()
    assert E.fmt(wb.rangkorrelation) == "0,70" and E.fmt(wb.abstand, 3) == "1,235"
    assert wb.rang1_i == 0 and wb.rang1_k == 4                                       # Art interventional, Vorlauf konditional
    assert np.allclose(wb.interventional[[1, 3]], wb.konditional[[1, 3]], atol=1e-9)  # Kunde und Wochenende unabhängig: beide Funktionale gleich


def test_ziehungen_steigen_mit_der_instanzzahl_und_sind_wahrscheinlichkeiten():
    zs = [E.ziehungen(m) for m in C.ZIEHUNG_GROESSEN]
    a = [z.art_vor_vorlauf_i for z in zs]
    assert a == sorted(a) and all(0 <= x <= 1 for x in a) and zs[0].n_ziehungen == C.ANZ_ZIEHUNGEN
    assert zs[0].art_q05 < zs[0].art_q95 and (zs[3].art_q95 - zs[3].art_q05) < (zs[0].art_q95 - zs[0].art_q05)
    assert zs[3].vorlauf_vor_art_k > 0.999


# ---------------------------------------------------------------------------------------------- Experiment 2
def test_zwei_funktionen_spannen_und_dasselbe_ergebnis_je_menge():
    zf = E.zwei_funktionen(0, 150, 0)
    assert zf.spanne_summe[0] <= zf.summe <= zf.spanne_summe[1] and zf.n_blaetter == T.ersatzbaum().n_blaetter
    assert E.zwei_funktionen(3, 150, 1).spanne_summe == E.zwei_funktionen(0, 150, 1).spanne_summe       # die Spanne gilt für alle fünf Mengen, nicht für die gewählte
    assert E.zwei_funktionen(0, 10, 0).n == 10


def test_pfadabhaengig_gegen_interventional_auf_demselben_baum_ist_viel_naeher_als_mu_gegen_baum():
    a, b = E.zwei_funktionen(0, 150, 0), E.zwei_funktionen(0, 150, 1)
    assert b.spanne_summe[1] * 10 < a.spanne_summe[0]


def test_eingefrorener_pfadbeitrag_gleich_live_rechnung_und_interventional_des_baums_gleich_neurechnung(pop):
    er = T.ersatzbaum()
    X = T.e2_zeilen(1).X[:6]
    live = P.shapley_aus_werten(TR.pfad_werte(er.baum, X))
    assert np.max(np.abs(live - T.pfad_baum_alle()[150:156])) < 1e-9
    Xp, wp = pop
    iv = P.shapley_aus_werten(P.werte_interventional(lambda Z: er.baum.vorhersage(Z), X, Xp, wp))
    assert np.max(np.abs(iv - er.phi_i[150:156])) < 1e-6                          # die Datei hält sieben Stellen


# ---------------------------------------------------------------------------------------------- Experiment 3 und eingefrorene Modelle
@pytest.mark.parametrize("k", [0, 1, 2])
def test_eingefrorene_modellbeitraege_erfuellen_die_effizienz_konsistenz(k):
    """Σφ_int − Σφ_kond = E[f] − E_Hintergrund[f] ist für alle Instanzen derselbe Wert (beide Summen sind f(x) minus eine Konstante); bei pfadabhängig ebenso."""
    kl = T.klasse(k)
    for sd in range(C.N_MODELL_SEEDS):
        di = kl.phi["interventional"][sd].sum(axis=1) - kl.phi["konditional"][sd].sum(axis=1)
        assert np.ptp(di) < 1e-5, (k, sd)
        if "pfadabhaengig" in kl.phi:
            dp = kl.phi["pfadabhaengig"][sd].sum(axis=1) - kl.phi["konditional"][sd].sum(axis=1)
            assert np.ptp(dp) < 1e-5, (k, sd)
        art = np.concatenate([T.e3_zeilen(i).X[:, 0] for i in range(5)]).astype(int)
        da = kl.phi["interventional_art"][sd].sum(axis=1) - kl.phi["konditional"][sd].sum(axis=1)
        for a in range(4):
            if (art == a).any():
                assert np.ptp(da[art == a]) < 1e-5


def test_form_der_eingefrorenen_modelle():
    for k in range(3):
        kl = T.klasse(k)
        assert kl.r2.shape == (8,) and kl.loo.shape == (8, 5) and kl.r2.min() > 0.85
        for fn, a in kl.phi.items():
            assert a.shape == (8, 5 * C.E3_N, 5), fn
        assert ("pfadabhaengig" in kl.phi) == (k != C.KLASSE_NETZ)
    assert T.loo_exakt().shape == (5,) and np.argmax(T.loo_exakt()) == 4              # Vorlauf hat den größten Verlustanstieg


def test_leave_one_out_exakt_gegen_zellenrechnung_der_bevoelkerung(pop):
    """Verlustanstieg ohne Merkmal j = Σ w (μ − E[μ | übrige])² über die Bevölkerung; hier mit einer zweiten, ausgeschriebenen Rechnung für das Wochenende."""
    Xp, wp = pop
    mu = G.wahrheit(Xp)
    schluessel = [tuple(r[[0, 1, 2, 4]]) for r in Xp]                                  # ohne Wochenende: Gruppen gleicher übriger Merkmale
    gruppen = {}
    for i, s in enumerate(schluessel):
        gruppen.setdefault(s, []).append(i)
    verlust = 0.0
    for idx in gruppen.values():
        w = wp[idx]
        m = float(w @ mu[idx] / w.sum())
        verlust += float(w @ (mu[idx] - m) ** 2)
    assert verlust == pytest.approx(T.loo_exakt()[3], abs=1e-6)


@pytest.mark.parametrize("k", [0, 1])
def test_neu_trainieren_gibt_dieselben_beitraege_als_band(k):
    """Boosting und Wald: Zufalls-Seed 0 neu trainiert und für fünf Instanzen neu gerechnet (scikit-learn; Band 1e-6, die Letztstelle hängt von der Plattform ab)."""
    pytest.importorskip("sklearn")
    tool = _tool("wfk_make_models")
    X, y = G.daten(C.N_TRAIN, 0)
    pred, m = tool.trainieren(k, X[:, :5], y, 0)
    Xe = T.e3_zeilen(0).X[:5]
    Xp, wp = P.bevoelkerung()
    phi = P.shapley_aus_werten(P.werte_interventional(lambda Z: pred(Z[:, :5]), Xe, Xp, wp))
    assert np.max(np.abs(phi - T.klasse(k).phi["interventional"][0][:5])) < 1e-6
    pfad = P.shapley_aus_werten(tool.pfad_modell(k, m, Xe))
    assert np.max(np.abs(pfad - T.klasse(k).phi["pfadabhaengig"][0][:5])) < 1e-6


def test_modelle_spannen_und_zaehleinheiten():
    for k in range(3):
        for hint in (0, 1):
            m = E.modelle(k, 0, hint)
            assert m.gleich_i[0] <= m.gleich_i[1] and len(m.rang1_treffer) == 5 and all(0 <= t <= 8 for t in m.rang1_treffer)
            assert (m.int_gegen_pfad is None) == (k == C.KLASSE_NETZ)
            e1 = E.modelle(k, 1, hint)
            assert len(e1.rang1_treffer) == 1 and e1.gleich_i[0] == e1.gleich_i[1]       # eine Menge: Spanne ist ein Punkt


def test_modelle_summe_der_abweichungen_von_hand_fuer_eine_menge():
    kl = T.klasse(C.KLASSE_BOOSTING)
    z = T.e3_zeilen(0)
    werte = [float(np.abs(kl.phi["interventional"][sd][:40] - z.phi_i[:, :5]).mean(axis=0).sum()) for sd in range(8)]
    assert E.modelle(C.KLASSE_BOOSTING, 1, 0).gleich_i[0] == pytest.approx(float(np.mean(werte)), abs=1e-12)


def test_hintergrund_derselben_art_verkleinert_den_abstand_bei_allen_klassen():
    for k in range(3):
        assert E.modelle(k, 0, 1).gleich_i[1] < E.modelle(k, 0, 0).gleich_i[0]


def test_netz_extrapoliert_glatt_interventional_ist_weit_von_der_konditionalen_wahrheit():
    net, boost = E.modelle(C.KLASSE_NETZ, 0, 0), E.modelle(C.KLASSE_BOOSTING, 0, 0)
    assert net.gleich_i[1] < boost.gleich_i[0]                                         # das Netz liegt näher an der interventionalen Wahrheit
    assert net.int_gegen_wahrheit_k[0] > boost.int_gegen_wahrheit_k[1]               # und weiter von der konditionalen


# ---------------------------------------------------------------------------------------------- Tools
def test_vergleichswerkzeuge_erkennen_abweichungen():
    t = _tool("wfk_make_importance")
    alt = json.loads((ROOT / "wfk_truth.json").read_text(encoding="utf-8"))
    import copy
    assert t.vergleichen(alt, copy.deepcopy(alt)) == []
    neu = copy.deepcopy(alt)
    neu["e3"][1]["phi_interventional_art"][3][2] += 1e-6
    neu["e2"][0]["pruefsumme"] += 1.0
    assert len(t.vergleichen(neu, alt)) == 2
    m = _tool("wfk_make_models")
    am = json.loads((ROOT / "wfk_models.json").read_text(encoding="utf-8"))
    assert m.vergleichen(am, copy.deepcopy(am)) == []
    bm = copy.deepcopy(am)
    bm["klassen"]["Boosting"]["phi"]["konditional"][2][4][1] += 1e-3
    assert len(m.vergleichen(bm, am)) == 1


def test_eingefrorene_wahrheit_neu_gerechnet_als_band(pop):
    Xp, wp = pop
    z = T.e3_zeilen(2)
    pT, pC = P.wahrheit_je_funktional(G.wahrheit, z.X[:20], Xp, wp)
    assert np.max(np.abs(pT[:, :5] - z.phi_i[:20])) < 1e-9 and np.max(np.abs(pC[:, :5] - z.phi_k[:20])) < 1e-9
    pa = P.shapley_aus_werten(P.werte_interventional_art(G.wahrheit, z.X[:20], Xp, wp))
    assert np.max(np.abs(pa - z.phi_i_art[:20])) < 1e-9
    for i in range(5):
        for zz in (T.e2_zeilen(i), T.e3_zeilen(i)):
            assert zz.X.shape[1] == 6 and len(zz.phi_i) == len(zz.X)
    assert T.e1_zeilen().X.shape == (300, 6)
