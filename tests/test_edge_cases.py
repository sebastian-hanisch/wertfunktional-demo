"""Randfälle: Hintergründe mit einer Zeile, ρ und q an den Rändern, kleinste und größte Zeile, Worst Case am Regler-Maximum."""

import numpy as np

import wfk_constants as C
import wfk_evaluation as E
import wfk_generator as G
import wfk_pop as P
import wfk_tree as TR
import wfk_truth as T
import wfk_twin as TW


def test_hintergrund_mit_einer_zeile_gibt_effizienz_und_nullspieler():
    Xb = G.zeilenmenge(9100, 1)
    Xe = G.zeilenmenge(4242, 4)
    v = P.werte_interventional(G.wahrheit, Xe, Xb, np.array([1.0]))
    phi = P.shapley_aus_werten(v)
    assert np.max(np.abs(phi.sum(axis=1) - (v[:, -1] - v[:, 0]))) < 1e-12
    assert np.allclose(v[:, 0], v[0, 0])                                              # v(leer) ist der Wert des einen Hintergrunds


def test_instanz_gleich_hintergrundzeile_gibt_beitrag_null():
    x = G.zeilenmenge(4242, 1)
    phi = P.shapley_aus_werten(P.werte_interventional(G.wahrheit, x, x, np.array([1.0])))
    assert np.max(np.abs(phi)) < 1e-12


def test_erste_und_letzte_zeile_in_jeder_hintergrundoption():
    for opt in range(len(C.HG_OPTIONEN)):
        for zeile in (C.ZEILE_MIN, C.ZEILE_MAX):
            ez = E.eine_zeile(zeile, opt)
            assert np.all(np.isfinite(ez.phi_i)) and np.all(np.isfinite(ez.phi_p)) and abs(ez.phi_i.sum() - (ez.mu - ez.grundwert_i)) < 1e-9


def test_zwilling_an_den_raendern_der_regler():
    assert TW.gauss_anteil_kond(C.RHO_MIN) == 0.0 and 0.49 < TW.gauss_anteil_kond(C.RHO_MAX) < 0.5
    assert TW.binaer_anteil_kond(C.Q_MIN) == 0.0 and 0.49 < TW.binaer_anteil_kond(C.Q_MAX) < 0.5
    for fall, rho, q in ((0, C.RHO_MIN, C.Q_MIN), (0, C.RHO_MAX, C.Q_MAX), (1, C.RHO_MIN, C.Q_MIN), (1, C.RHO_MAX, C.Q_MAX)):
        k, t = E.verdict_zwilling(fall, rho, q)
        assert k == "info" and "nan" not in t


def test_kleinste_instanzzahl_und_alle_mengen_im_vergleich_der_funktionen():
    for menge in range(5):
        for fn in (0, 1):
            zf = E.zwei_funktionen(menge, min(C.E2_N_OPTIONEN), fn)
            assert np.isfinite(zf.summe) and 0 <= zf.gleiches_top <= 1


def test_modelle_jede_klasse_menge_und_hintergrund_ist_endlich():
    for k in range(3):
        for menge in range(6):
            for hint in (0, 1):
                m = E.modelle(k, menge, hint)
                assert np.isfinite(m.gleich_i[0]) and np.all(np.isfinite(m.loo)) and len(m.loo_rang) == 5


def test_baum_ohne_inneren_knoten_in_der_pfadrechnung():
    t = TR.Tree(np.array([-1]), np.array([0.0]), np.array([-1]), np.array([-1]), np.array([1.0]), np.array([5.0]))
    assert np.all(TR.pfad_werte(t, G.zeilenmenge(4242, 3)) == 1.0)


def test_worst_case_am_regler_maximum_haengt_nicht():
    """Alle Experimente am Maximum (Zeile 299, Stichprobe von 1 000 Zeilen, 150 Instanzen, jede Klasse und Menge). Nur der Lauf zählt; Zeiten sind plattformabhängig."""
    for opt in (C.HG_S1000, C.HG_ART):
        assert np.isfinite(E.eine_zeile(C.ZEILE_MAX, opt).phi_i).all()
    assert np.isfinite(E.zwei_funktionen(4, 150, 0).summe)
    assert all(np.isfinite(E.ziehungen(m).art_vor_vorlauf_i) for m in C.ZIEHUNG_GROESSEN)
    assert T.e1_zeilen().X.shape[0] == C.N_ZEILEN
