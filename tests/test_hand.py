"""Mini-Vehikel (Brüche): dieselbe Funktion, zwei Bäume, vier Antworten; Orakel: alle Permutationen, Handwerte aus der Tabelle der Verteilung, Leave-one-out durch Neuberechnung."""

import itertools
from fractions import Fraction as Fr

import pytest

import wfk_hand as H


def perm_shapley(v):
    phi = {0: Fr(0), 1: Fr(0)}
    for pm in itertools.permutations([0, 1]):
        cur = frozenset()
        for j in pm:
            phi[j] += v(cur | {j}) - v(cur)
            cur = cur | {j}
    return {j: x / 2 for j, x in phi.items()}


def test_funktion_und_baeume_treffen_die_daten_und_weichen_ausserhalb_ab():
    for p in H.DIST:
        assert H.baum_wert(H.BAUM1, p) == H.MU(p) == H.baum_wert(H.BAUM2, p)
    assert H.MU((0, 1)) == 1 and H.baum_wert(H.BAUM1, (0, 1)) == 1 and H.baum_wert(H.BAUM2, (0, 1)) == 5


def test_vier_antworten_fuer_import_mit_zollhalt():
    m = H.mini()
    assert m.grundwert == Fr(5, 2) and m.vorhersage == 5
    a = m.antworten
    assert a["interventional (μ)"] == {0: Fr(3, 2), 1: Fr(1)}
    assert a["konditional (μ)"] == {0: Fr(3, 4), 1: Fr(7, 4)}
    assert a["pfadabhängig, Baum 1"] == {0: Fr(7, 4), 1: Fr(3, 4)}
    assert a["pfadabhängig, Baum 2"] == {0: Fr(1, 2), 1: Fr(2)}
    assert a["interventional, Baum 1"] == {0: Fr(3, 2), 1: Fr(1)} and a["interventional, Baum 2"] == {0: Fr(1, 2), 1: Fr(2)}
    for name, phi in a.items():
        assert sum(phi.values()) == Fr(5, 2), name                                 # Effizienz: Vorhersage 5 minus Grundwert 5/2


def test_antworten_gleich_allen_permutationen_und_der_tabelle_von_hand():
    x = (1, 1)
    assert perm_shapley(lambda S: H.v_interv(H.MU, x, S)) == {0: Fr(3, 2), 1: Fr(1)}
    assert perm_shapley(lambda S: H.v_kond(H.MU, x, S)) == {0: Fr(3, 4), 1: Fr(7, 4)}
    assert perm_shapley(lambda S: H.baum_pfad_wert(H.BAUM1, x, S)) == {0: Fr(7, 4), 1: Fr(3, 4)}
    assert perm_shapley(lambda S: H.baum_pfad_wert(H.BAUM2, x, S)) == {0: Fr(1, 2), 1: Fr(2)}
    # Handwerte der Koalitionen: v({Zoll}) konditional = 5 (nur Import mit Zollhalt hat Zollhalt), v({Art}) konditional = 4
    assert H.v_kond(H.MU, x, frozenset({1})) == 5 and H.v_kond(H.MU, x, frozenset({0})) == 4
    # Baum 1, Zoll festhalten: Art nicht benutzt für Zoll=1 im Pfad 'Art zuerst' -> Mischung: 1/2 * 1 + 1/2 * 5 = 3
    assert H.baum_pfad_wert(H.BAUM1, x, frozenset({1})) == 3
    # Baum 2, Art festhalten: der Zollknoten mischt (Zollhalt 1/4 -> 5, sonst 3/4 -> Art folgt x: 3): 7/2
    assert H.baum_pfad_wert(H.BAUM2, x, frozenset({0})) == Fr(1, 4) * 5 + Fr(3, 4) * 3 == Fr(7, 2)


def test_wichtigkeit_und_rangfolge_kippt_zwischen_den_baeumen():
    w = H.mini().wichtigkeit
    assert w["interventional (μ)"] == (Fr(5, 4), Fr(1, 2))
    assert w["konditional (μ)"] == (Fr(13, 12), Fr(7, 8))
    assert w["pfadabhängig, Baum 1"] == (Fr(11, 8), Fr(1, 2))
    assert w["pfadabhängig, Baum 2"] == (Fr(17, 24), Fr(9, 8))
    assert w["pfadabhängig, Baum 1"][0] > w["pfadabhängig, Baum 1"][1] and w["pfadabhängig, Baum 2"][0] < w["pfadabhängig, Baum 2"][1]


def test_leave_one_out_durch_neuberechnung_der_gruppenmittel():
    m = H.mini()
    assert m.loo == {"Art": Fr(2, 3), "Zoll": Fr(1, 2)}
    # von Hand: ohne Zoll sagt der Wert nur die Art voraus (Import: Mittel 4): Verlust 1/4 + 1/4 = 1/2; ohne Art nur den Zoll (kein Zoll: Mittel 5/3): 6/9
    assert Fr(1, 4) * 1 + Fr(1, 4) * 1 == m.loo["Zoll"]
    assert Fr(1, 4) * (3 - Fr(5, 3)) ** 2 + Fr(1, 2) * (1 - Fr(5, 3)) ** 2 == m.loo["Art"]


def test_nullspieler_exakt_null_in_jedem_funktional():
    """Dritter, unbenutzter Münzspieler (Verteilung mit einer fairen Münze): Beitrag exakt 0 interventional und konditional."""
    dist = {(a, z, c): p / 2 for (a, z), p in H.DIST.items() for c in (0, 1)}
    mu3 = lambda x: H.MU(x[:2])
    x = (1, 1, 1)
    for v in (lambda S: H.v_interv(mu3, x, S, dist), lambda S: H.v_kond(mu3, x, S, dist)):
        phi = H.shapley([0, 1, 2], v)
        assert phi[2] == 0 and sum(phi.values()) == Fr(5, 2)
        # sechs Permutationen = Koalitionsformel
        perm = {j: Fr(0) for j in range(3)}
        for pm in itertools.permutations(range(3)):
            cur = frozenset()
            for j in pm:
                perm[j] += v(cur | {j}) - v(cur)
                cur |= {j}
        assert {j: s / 6 for j, s in perm.items()} == phi
