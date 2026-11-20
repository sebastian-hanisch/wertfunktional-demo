"""Orakel für die Wahrheit je Funktional (Shapley-Wert der wahren Mitte unter dem interventionalen und dem konditionalen Wertfunktional).

Anderer Rechenweg: alle 120 Permutationen statt der Koalitionsformel; Effizienz und Nullspieler; die Bevölkerung gegen andere Quadratur (Gauß-Legendre mit 48 Punkten, Mittelpunktsregel
mit 400) und gegen eine Stichprobe natürlicher Zeilen (Band mit Standardfehler); die bedingte Erwartung gegen von Hand hingeschriebene Summen (Art, Vorlauf) und gegen
Funktionen unabhängiger Merkmale, bei denen beide Wertfunktionale gleich sein müssen."""

import itertools

import numpy as np
import pytest

import wfk_constants as C
import wfk_generator as G
import wfk_pop as P

MU = G.wahrheit


def shapley_aus_perm(v):
    """(ne, 32) Koalitionswerte -> (ne, 5) durch Mittel der Grenzbeiträge über alle 120 Reihenfolgen."""
    phi = np.zeros((v.shape[0], 5))
    perms = list(itertools.permutations(range(5)))
    for pm in perms:
        k = 0
        for j in pm:
            phi[:, j] += v[:, k | (1 << j)] - v[:, k]
            k |= 1 << j
    return phi / len(perms)


@pytest.fixture(scope="module")
def klein():
    """Zeilen und ein kleiner gewichteter Hintergrund (60 zufällige Stützpunkte der Bevölkerung, Gewichte neu normiert)."""
    Xp, wp = P.bevoelkerung()
    rng = np.random.default_rng(1)
    idx = rng.choice(len(Xp), 60, replace=False)
    w = wp[idx] / wp[idx].sum()
    return G.zeilenmenge(4242, 12), Xp[idx], w


def test_koalitionsformel_gleich_allen_permutationen_interventional(klein):
    Xe, Xb, wb = klein
    v = P.werte_interventional(MU, Xe, Xb, wb)
    assert np.max(np.abs(P.shapley_aus_werten(v) - shapley_aus_perm(v))) < 1e-13


def test_koalitionsformel_gleich_allen_permutationen_konditional(klein):
    Xe = klein[0]
    v = P.werte_konditional(MU, Xe)
    assert np.max(np.abs(P.shapley_aus_werten(v) - shapley_aus_perm(v))) < 1e-13


def test_effizienz_beider_wertfunktionale(klein, pop):
    Xe, _, _ = klein
    Xp, wp = pop
    emu = float(wp @ MU(Xp))
    pT, pC = P.wahrheit_je_funktional(MU, Xe, Xp, wp)
    mu = MU(Xe)
    assert np.max(np.abs(pT.sum(axis=1) - (mu - emu))) < 1e-9
    assert np.max(np.abs(pC.sum(axis=1) - (mu - emu))) < 1e-9
    assert np.all(pT[:, 5] == 0) and np.all(pC[:, 5] == 0)


def test_nullspieler_gewicht_hat_exakt_null_auch_bei_sechs_spielern(klein):
    Xe, Xb, wb = klein
    for x in Xe[:3]:
        phi, _, _ = P.shapley_aufzaehlung(MU, x, np.where(np.arange(6) == 5, np.random.default_rng(0).normal(size=(len(Xb), 6)), Xb), wb)
        assert phi[5] == 0.0


def test_bevoelkerung_gewichte_und_randverteilungen(pop):
    Xp, wp = pop
    assert len(Xp) == C.N_POP and abs(wp.sum() - 1) < 1e-14
    for a, p in enumerate(C.ART_P):
        assert wp[Xp[:, 0] == a].sum() == pytest.approx(p, abs=1e-14)
    assert wp[Xp[:, 2] == 1].sum() == pytest.approx(0.35 * 0.25, abs=1e-14)
    assert wp[Xp[:, 3] == 1].sum() == pytest.approx(2 / 7, abs=1e-14)
    assert wp[(Xp[:, 0] == 1)] @ Xp[Xp[:, 0] == 1, 4] / wp[Xp[:, 0] == 1].sum() == pytest.approx(5.5, abs=1e-12)
    assert wp[(Xp[:, 0] == 2)] @ Xp[Xp[:, 0] == 2, 4] / wp[Xp[:, 0] == 2].sum() == pytest.approx(3.25, abs=1e-12)
    assert np.all(Xp[:, 5] == 0)


def test_erwartung_der_wahren_mitte_gegen_andere_quadratur_und_stichprobe(pop):
    Xp, wp = pop
    e24 = float(wp @ MU(Xp))
    X48, w48 = P.bevoelkerung(48)
    e48 = float(w48 @ MU(X48))
    Xm, wm = P.bevoelkerung(400, "mid")
    em = float(wm @ MU(Xm))
    assert abs(e24 - e48) < 1e-12                       # Gauß-Legendre 24 gegen 48
    assert abs(e24 - em) < 5e-5                         # gegen Mittelpunktsregel mit 400 Punkten
    Xs, _ = G.merkmale(C.SET_OFFSET + 9001, 400_000)
    mus = MU(Xs)
    se = mus.std() / np.sqrt(len(mus))
    assert abs(mus.mean() - e24) < 4 * se               # Stichprobe natürlicher Zeilen: Band mit Standardfehler


def test_konditionale_erwartung_art_von_hand():
    """E[mu | Art = a] als ausgeschriebene Summe (anderer Rechenweg): über Kunde (gleichverteilt), Zoll, Wochenende und Vorlauf (Mittelpunktsregel mit 4000 Punkten)."""
    uk = np.array(C.UK)
    a_import = np.mean([sum(pz * pw * (np.log(3.5 * np.exp(0.7 * z + 0.3 * w)) + uk[k]) for z, pz in ((0, 0.75), (1, 0.25)) for w, pw in ((0, 5 / 7), (1, 2 / 7))) for k in range(12)])
    q = 4000
    v_ex = 1 + 9 * (np.arange(q) + 0.5) / q
    a_export = np.mean(np.log(0.6 + 0.85 * v_ex)) + 0.3 * uk.mean()
    v_um = 0.5 + 5.5 * (np.arange(q) + 0.5) / q
    a_um = np.mean(np.log(0.3 + 0.8 * v_um)) + 0.3 * uk.mean()
    a_leer = np.mean([sum(pw * (np.log(7.0 * np.exp(0.3 * w)) + uk[k]) for w, pw in ((0, 5 / 7), (1, 2 / 7))) for k in range(12)])
    X = np.array([[a, 0, 0, 0, 0, 0] for a in range(4)], float)
    ist = P.bedingte_erwartung(MU, X, [True, False, False, False, False])
    assert np.allclose(ist, [a_import, a_export, a_um, a_leer], atol=1e-5)


def test_konditionale_erwartung_vorlauf_von_hand_mit_posterior_der_art():
    """E[mu | Vorlauf = v] für v = 3 (in beiden Trägern): Posterior Export gegen Umschlag aus den Dichten 0,30/9 und 0,25/5,5."""
    uk = np.array(C.UK)
    v = 3.0
    pe, pu = 0.30 / 9, 0.25 / 5.5
    w_exp = pe / (pe + pu)
    erwartet = w_exp * (np.log(0.6 + 0.85 * v) + 0.3 * uk.mean()) + (1 - w_exp) * (np.log(0.3 + 0.8 * v) + 0.3 * uk.mean())
    ist = P.bedingte_erwartung(MU, np.array([[0, 0, 0, 0, v, 0.0]]), [False, False, False, False, True])[0]
    assert ist == pytest.approx(erwartet, abs=1e-12)
    # v = 8 liegt nur im Träger von Export; v = 0 nur in Import und Leer (Posterior 0,35 : 0,10)
    assert P.bedingte_erwartung(MU, np.array([[0, 0, 0, 0, 8.0, 0.0]]), [False] * 4 + [True])[0] == pytest.approx(np.log(0.6 + 0.85 * 8) + 0.3 * uk.mean(), abs=1e-12)
    imp = np.mean([sum(pz * pw * (np.log(3.5 * np.exp(0.7 * z + 0.3 * w)) + uk[k]) for z, pz in ((0, 0.75), (1, 0.25)) for w, pw in ((0, 5 / 7), (1, 2 / 7))) for k in range(12)])
    leer = np.mean([sum(pw * (np.log(7.0 * np.exp(0.3 * w)) + uk[k]) for w, pw in ((0, 5 / 7), (1, 2 / 7))) for k in range(12)])
    erwartet0 = (0.35 * imp + 0.10 * leer) / 0.45
    assert P.bedingte_erwartung(MU, np.zeros((1, 6)), [False] * 4 + [True])[0] == pytest.approx(erwartet0, abs=1e-12)


def test_konditionale_erwartung_zoll_gegen_stichprobe():
    """E[mu | Zoll = 1] gegen den Mittelwert der Stichprobenzeilen mit Zollhalt (Band mit Standardfehler)."""
    Xs, _ = G.merkmale(C.SET_OFFSET + 9002, 400_000)
    sel = Xs[:, 2] == 1
    mus = MU(Xs[sel])
    se = mus.std() / np.sqrt(len(mus))
    ist = P.bedingte_erwartung(MU, np.array([[0, 0, 1, 0, 0, 0.0]]), [False, False, True, False, False])[0]
    assert abs(ist - mus.mean()) < 4 * se


def test_bei_unabhaengigen_merkmalen_sind_beide_wertfunktionale_gleich(klein, pop):
    """Eine Funktion von Kunde und Wochenende (unabhängig voneinander und von allem anderen): interventional = konditional = f_j(x_j) − E f_j."""
    Xe, _, _ = klein
    Xp, wp = pop
    f = lambda Z: np.array(C.UK)[Z[:, 1].astype(int)] + 2.0 * Z[:, 3]
    pT, pC = P.wahrheit_je_funktional(f, Xe, Xp, wp)
    assert np.max(np.abs(pT - pC)) < 1e-12
    erwartet_k = np.array(C.UK)[Xe[:, 1].astype(int)] - np.mean(C.UK)
    erwartet_w = 2.0 * (Xe[:, 3] - 2 / 7)
    assert np.allclose(pT[:, 1], erwartet_k, atol=1e-12) and np.allclose(pT[:, 3], erwartet_w, atol=1e-12)
    assert np.allclose(pT[:, [0, 2, 4]], 0, atol=1e-12)


def test_bei_abhaengigen_merkmalen_sind_die_wertfunktionale_verschieden(klein, pop):
    """Gegenprobe zur Absolutaussage: für die wahre Mitte (Art und Vorlauf abhängig) weichen die Wahrheiten im Mittel stark ab."""
    Xe, _, _ = klein
    Xp, wp = pop
    pT, pC = P.wahrheit_je_funktional(MU, Xe, Xp, wp)
    assert np.abs(pT - pC).mean(axis=0)[[0, 4]].min() > 0.1
