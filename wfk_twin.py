"""Zwillinge: das Wertfunktional entscheidet, wer den Beitrag bekommt (geschlossene Formeln, Orakel in den Tests).

Gauß-Fall: f(x) = x_A, (x_A, x_B) standardnormal mit Korrelation rho. Spiel mit zwei Spielern: v(leer) = 0, v(A) = x_A, v(B) = E[x_A | x_B] = rho x_B (konditional) bzw. 0 (interventional),
v(A, B) = x_A. Konditional: phi_A = x_A − rho x_B / 2, phi_B = rho x_B / 2; interventional: phi_A = x_A, phi_B = 0.
Binärer Fall (Janzing u. a., Beispiel 1, Stufe W): x_1 gleichverteilt auf {0, 1}, x_2 = x_1 mit Wahrscheinlichkeit q, f = x_1."""

from fractions import Fraction as Fr
from math import sqrt

import numpy as np


def gauss_phi_kond(xa, xb, rho):
    """Konditionale Beiträge (phi_A, phi_B) im Gauß-Fall."""
    return xa - rho * xb / 2, rho * xb / 2


def gauss_anteil_kond(rho: float) -> float:
    """Anteil des Zwillings an der Wichtigkeit (E|phi_B| / (E|phi_A| + E|phi_B|)) konditional: (rho / 2) / (rho / 2 + sqrt(1 − 3 rho² / 4))."""
    return (rho / 2.0) / (rho / 2.0 + sqrt(1.0 - 0.75 * rho * rho))


def gauss_wichtigkeit_kond(rho: float):
    """(E|phi_A|, E|phi_B|) konditional: phi_A ~ N(0, 1 − 3 rho² / 4), phi_B ~ N(0, rho² / 4); E|N(0, s²)| = s sqrt(2 / pi)."""
    k = sqrt(2.0 / np.pi)
    return sqrt(1.0 - 0.75 * rho * rho) * k, (rho / 2.0) * k


def gauss_wichtigkeit_int() -> tuple:
    """Interventional: phi_A = x_A (E|x_A| = sqrt(2 / pi)), phi_B = 0."""
    return sqrt(2.0 / np.pi), 0.0


def binaer_phi_kond(x1: int, x2: int, q):
    """Konditionale Beiträge (phi_1, phi_2) im binären Fall (q als float oder Fraction)."""
    q = Fr(q) if not isinstance(q, float) else q
    v2 = q if x2 == 1 else 1 - q                     # E[x_1 | x_2]  (P(x_1 = 1 | x_2 = 1) = q, P(x_1 = 1 | x_2 = 0) = 1 − q)
    half = Fr(1, 2) if not isinstance(q, float) else 0.5
    return ((x1 - half) + (x1 - v2)) / 2, ((v2 - half) + 0) / 2


def binaer_wichtigkeit_kond(q):
    """(E|phi_1|, E|phi_2|) konditional im binären Fall; geschlossen: q (3/4 − q/2) + (1 − q) (1/4 + q/2) und |q − 1/2| / 2 (für q >= 1/2)."""
    q = Fr(q) if not isinstance(q, float) else q
    one = Fr(1) if not isinstance(q, float) else 1.0
    a = q * (3 * one / 4 - q / 2) + (1 - q) * (one / 4 + q / 2)
    b = abs(q - one / 2) / 2
    return a, b


def binaer_anteil_kond(q):
    a, b = binaer_wichtigkeit_kond(q)
    return b / (a + b)
