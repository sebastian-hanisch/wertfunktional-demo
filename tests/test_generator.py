"""Generator: Kopie der verweilzeit-demo (Kopie-Test gegen eingefrorene Werte der Quelle), SplitMix64-Strom und Verteilung der Merkmale."""

import numpy as np
import pytest

import wfk_constants as C
import wfk_generator as G
import wfk_rng as R

# Werte der Quelle (verweilzeit-demo, vwz_generator.log_median_and_s) an acht Punkten, 2026-11-19 gemessen und hier eingefroren.
PUNKTE = dict(art=[0, 1, 2, 3, 0, 1, 2, 3], kunde=[0, 3, 5, 11, 7, 2, 9, 4], zoll=[0, 0, 0, 0, 1, 0, 0, 0], we=[0, 1, 0, 1, 1, 0, 1, 0],
              vorlauf=[0, 2.5, 1.5, 0, 0, 9.0, 0.5, 0])
MU_QUELLE = [0.8256154575589787, 0.9791428369711116, 0.3387854894204597, 2.5364592207808903, 2.44743080915328, 2.0318536439302717, -0.5324326196098295, 1.923307156952157]
S_QUELLE = [0.55, 0.25, 0.35, 0.9, 0.55, 0.25, 0.35, 0.9]
UK_NUMPY_ERSTE = [-0.42714751093638936, 0.3791185374387331, -0.2611985213877257]


def test_kopie_mitte_und_streuung_gleich_den_werten_der_quelle():
    mu, s = G.mitte_und_streuung(**PUNKTE)
    assert np.allclose(mu, MU_QUELLE, rtol=0, atol=1e-12)
    assert list(s) == S_QUELLE


def test_kundeneffekte_als_literal_gleich_numpy_default_rng():
    uk = np.random.default_rng(12345).normal(0.0, 0.30, 12)
    assert np.allclose(uk, C.UK, rtol=0, atol=1e-12)
    assert list(C.UK[:3]) == UK_NUMPY_ERSTE


def test_mitte_handrechnung_import_mit_zoll_am_wochenende():
    # Import, Kunde 0, Zoll, Wochenende: ln(3,5 e^(0,7 + 0,3)) + Kundeneffekt 0 mal 1,0
    mu, s = G.mitte_und_streuung(0, 0, 1, 1, 0.0)
    assert mu == pytest.approx(np.log(3.5) + 1.0 + C.UK[0], abs=1e-12)
    # Export, Vorlauf 2: ln(0,6 + 0,85 * 2) + 0,3 * Kundeneffekt
    mu, _ = G.mitte_und_streuung(1, 4, 0, 0, 2.0)
    assert mu == pytest.approx(np.log(0.6 + 1.7) + 0.3 * C.UK[4], abs=1e-12)


def test_vektorisierter_strom_gleich_der_klasse_splitmix64_bitgleich():
    for seed in (0, 1, 123456789, 2 ** 63 + 5):
        r = R.SplitMix64(seed)
        erwartet = np.array([r.random() for _ in range(200)])
        assert np.array_equal(G.uniforms(seed, 200), erwartet)


def test_box_muller_gleich_normal_der_klasse_auf_1e_12():
    u = G.uniforms(99, 400)
    r = R.SplitMix64(99)
    klasse = np.array([r.normal() for _ in range(200)])
    vek = G._box_muller(u[0::2], u[1::2])
    assert np.max(np.abs(vek - klasse)) < 1e-12


def test_gewicht_und_rauschen_des_generators_gleich_der_skalaren_ziehung():
    """Zeile i nutzt die Ziehungen 10 i bis 10 i + 9: Nachrechnen mit der Klasse, Zeile für Zeile."""
    X, y = G.daten(6, 3)
    r = R.SplitMix64(C.TRAIN_OFFSET + 3)
    for i in range(6):
        u = [r.random() for _ in range(10)]
        art = min(int(np.searchsorted(np.cumsum(C.ART_P), u[0], side="right")), 3)
        kunde = min(int(u[1] * 12), 11)
        zoll = float(u[2] < 0.25 and art == 0)
        we = float((int(np.floor(u[3] * 7000.0)) % 7) >= 5)
        vor = 1 + 9 * u[4] if art == 1 else (0.5 + 5.5 * u[4] if art == 2 else 0.0)
        g = np.sqrt(-2 * np.log(max(u[5], 1e-12))) * np.cos(2 * np.pi * u[6])
        mu, s = G.mitte_und_streuung(art, kunde, zoll, we, vor)
        z = np.sqrt(-2 * np.log(max(u[7], 1e-12))) * np.cos(2 * np.pi * u[8])
        assert list(X[i, :5]) == [art, kunde, zoll, we, vor]
        assert X[i, 5] == pytest.approx(g, abs=1e-12)
        assert y[i] == pytest.approx(float(mu + s * z), abs=1e-12)


def test_die_ersten_zeilen_haengen_nicht_von_n_ab():
    X1, y1 = G.daten(50, 7)
    X2, y2 = G.daten(500, 7)
    assert np.array_equal(X1, X2[:50]) and np.array_equal(y1, y2[:50])


def test_getrennte_bereiche_trainingsdaten_mengen_testsatz():
    a, _ = G.daten(20, 5)
    b = G.zeilenmenge(5, 20)
    assert not np.array_equal(a, b)
    Xt, _ = G.testsatz()
    assert len(Xt) == C.TEST_N


def test_verteilung_der_merkmale_als_band_mit_standardfehler():
    X, _ = G.merkmale(C.SET_OFFSET + 31, 200_000)
    n = len(X)
    for a, p in enumerate(C.ART_P):
        ant = np.mean(X[:, 0] == a)
        assert abs(ant - p) < 4 * np.sqrt(p * (1 - p) / n)
    imp = X[:, 0] == 0
    assert abs(X[imp, 2].mean() - C.P_ZOLL) < 4 * np.sqrt(0.25 * 0.75 / imp.sum())
    assert X[~imp, 2].sum() == 0                                         # Zoll nur bei Import
    assert abs(X[:, 3].mean() - 2 / 7) < 4 * np.sqrt(2 / 7 * 5 / 7 / n)
    assert np.all(X[X[:, 0] == 0, 4] == 0) and np.all(X[X[:, 0] == 3, 4] == 0)   # Vorlauf nur bei Export und Umschlag
    ex, um = X[X[:, 0] == 1, 4], X[X[:, 0] == 2, 4]
    assert 1 <= ex.min() and ex.max() <= 10 and 0.5 <= um.min() and um.max() <= 6
    assert abs(ex.mean() - 5.5) < 0.05 and abs(um.mean() - 3.25) < 0.05
    assert abs(X[:, 5].mean()) < 0.02 and abs(X[:, 5].std() - 1) < 0.02   # Gewicht ~ N(0, 1)
    assert set(np.unique(X[:, 1])) == set(range(12))


def test_rauschen_gleicht_der_streuung_je_art():
    X, y = G.daten(150_000, 2)
    mu = G.wahrheit(X)
    for a, s in enumerate(C.S_BASE):
        res = (y - mu)[X[:, 0] == a]
        assert abs(res.mean()) < 4 * s / np.sqrt(len(res))
        assert abs(res.std() - s) < 0.02
