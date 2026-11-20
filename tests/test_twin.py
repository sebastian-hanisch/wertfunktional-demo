"""Zwillinge: geschlossene Formeln gegen Quadratur (scipy dblquad), Stichprobe (Band mit Standardfehler), Koalitionsformel und Brüche (Janzing u. a. 2019, Beispiel 1, Stufe W)."""

from fractions import Fraction as Fr
from math import sqrt

import numpy as np
import pytest

import wfk_twin as TW


def test_gauss_anteil_handwerte_bei_rho_09_und_099():
    assert TW.gauss_anteil_kond(0.9) == pytest.approx(0.4180, abs=5e-5)
    assert TW.gauss_anteil_kond(0.99) == pytest.approx(0.4902, abs=5e-5)
    assert TW.gauss_anteil_kond(0.0) == 0.0                                        # ρ = 0: kein Zwilling, alle Funktionale gleich
    assert TW.gauss_anteil_kond(1.0) == pytest.approx(0.5)                          # ρ → 1: Hälfte


def test_gauss_phi_gleich_koalitionsformel_mit_bedingter_erwartung_von_hand():
    """v(leer) = 0, v(A) = x_A, v(B) = E[x_A | x_B] = ρ x_B, v(A, B) = x_A; Shapley von Hand für den Punkt (13/10, −2/5), ρ = 9/10 in Brüchen."""
    xa, xb, rho = Fr(13, 10), Fr(-2, 5), Fr(9, 10)
    phi_a = ((xa - 0) + (xa - rho * xb)) / 2
    phi_b = ((rho * xb - 0) + (xa - xa)) / 2
    a, b = TW.gauss_phi_kond(xa, xb, rho)
    assert (a, b) == (phi_a, phi_b) == (xa - rho * xb / 2, rho * xb / 2)
    assert a + b == xa                                                              # Effizienz


def test_gauss_wichtigkeit_gegen_quadratur():
    dblquad = pytest.importorskip("scipy.integrate").dblquad
    for rho in (0.3, 0.9, 0.99):
        dens = lambda xb, xa: np.exp(-(xa * xa - 2 * rho * xa * xb + xb * xb) / (2 * (1 - rho * rho))) / (2 * np.pi * sqrt(1 - rho * rho))
        ea = dblquad(lambda xb, xa: abs(xa - rho * xb / 2) * dens(xb, xa), -8, 8, -8, 8, epsabs=1e-9, epsrel=1e-9)[0]
        eb = dblquad(lambda xb, xa: abs(rho * xb / 2) * dens(xb, xa), -8, 8, -8, 8, epsabs=1e-9, epsrel=1e-9)[0]
        a, b = TW.gauss_wichtigkeit_kond(rho)
        assert a == pytest.approx(ea, abs=1e-5) and b == pytest.approx(eb, abs=1e-5)
        assert TW.gauss_anteil_kond(rho) == pytest.approx(eb / (ea + eb), abs=1e-5)


def test_gauss_wichtigkeit_gegen_stichprobe_im_band():
    rng = np.random.default_rng(1)
    n = 400_000
    for rho in (0.9, 0.99):
        xa = rng.normal(size=n)
        xb = rho * xa + sqrt(1 - rho * rho) * rng.normal(size=n)
        pa, pb = TW.gauss_phi_kond(xa, xb, rho)
        for emp, theo in ((np.abs(pa), TW.gauss_wichtigkeit_kond(rho)[0]), (np.abs(pb), TW.gauss_wichtigkeit_kond(rho)[1])):
            assert abs(emp.mean() - theo) < 4.5 * emp.std() / sqrt(n)               # Band 4,5 Standardfehler
        ia = np.abs(xa)
        assert abs(ia.mean() - TW.gauss_wichtigkeit_int()[0]) < 4.5 * ia.std() / sqrt(n)


def test_binaerer_zwilling_handwerte_janzing_beispiel_1_mit_q_9_zu_10():
    q = Fr(9, 10)
    assert TW.binaer_phi_kond(1, 1, q) == (Fr(3, 10), Fr(1, 5))
    assert TW.binaer_phi_kond(1, 0, q) == (Fr(7, 10), Fr(-1, 5))
    a, b = TW.binaer_wichtigkeit_kond(q)
    assert (a, b) == (Fr(17, 50), Fr(1, 5))
    assert TW.binaer_anteil_kond(q) == Fr(10, 27)


def test_binaerer_zwilling_wichtigkeit_gegen_aufzaehlung_der_vier_punkte():
    for q in (Fr(1, 2), Fr(3, 5), Fr(9, 10), Fr(99, 100)):
        ea = eb = Fr(0)
        for x1 in (0, 1):
            for x2 in (0, 1):
                p = Fr(1, 2) * (q if x1 == x2 else 1 - q)
                phi1, phi2 = TW.binaer_phi_kond(x1, x2, q)
                ea += p * abs(phi1)
                eb += p * abs(phi2)
        assert (ea, eb) == TW.binaer_wichtigkeit_kond(q)
    assert TW.binaer_phi_kond(1, 1, 0.9) == pytest.approx((0.3, 0.2))                # float und Bruch gleich


def test_binaerer_zwilling_randfaelle_q_halb_und_q_gegen_eins():
    assert TW.binaer_anteil_kond(Fr(1, 2)) == 0                                     # x_2 unabhängig: kein Zwilling
    assert TW.binaer_anteil_kond(Fr(1)) == Fr(1, 2)                                  # x_2 = x_1 immer: Hälfte
    assert TW.binaer_anteil_kond(0.5) == 0.0


def test_effizienz_binaer_in_allen_vier_punkten():
    for q in (Fr(3, 5), Fr(9, 10)):
        for x1 in (0, 1):
            for x2 in (0, 1):
                assert sum(TW.binaer_phi_kond(x1, x2, q)) == x1 - Fr(1, 2)            # Vorhersage minus Grundwert 1/2
