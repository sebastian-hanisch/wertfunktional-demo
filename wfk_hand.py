"""Mini-Vehikel (Handbeispiel-Welt, kein zweites Vehikel der App): Zoll gibt es nur bei Import; dieselbe Funktion, zwei Bäume, vier Antworten, alles in Brüchen.

x = (Art, Zoll): Import (1) oder Export (0), Zollhalt nur bei Import; wahre Mitte μ = 1 + 2 Art + 2 Art Zoll; Verteilung (1, 0) 1/4, (1, 1) 1/4, (0, 0) 1/2.
Baum 1 fragt zuerst die Art, dann den Zoll; Baum 2 zuerst den Zoll, dann die Art. Beide geben auf den drei Stützpunkten dasselbe wie μ, außerhalb der Daten
(Export mit Zollhalt) Baum 1 den Wert 1, Baum 2 den Wert 5 (μ: 1). Die Tests rechnen alles mit anderen Wegen nach (Permutationen, numpy-Blattrechnung aus wfk_tree)."""

import itertools
from dataclasses import dataclass
from fractions import Fraction as Fr
from math import factorial

MU = lambda x: Fr(1 + 2 * x[0] + 2 * x[0] * x[1])
DIST = {(1, 0): Fr(1, 4), (1, 1): Fr(1, 4), (0, 0): Fr(1, 2)}


def shapley(players, v):
    d = len(players)
    phi = {}
    for j in players:
        s = Fr(0)
        rest = [q for q in players if q != j]
        for r in range(d):
            for S in itertools.combinations(rest, r):
                S = frozenset(S)
                s += Fr(factorial(r) * factorial(d - r - 1), factorial(d)) * (v(S | {j}) - v(S))
        phi[j] = s
    return phi


def v_interv(f, x, S, dist=DIST):
    return sum(p * f(tuple(x[j] if j in S else b[j] for j in range(len(x)))) for b, p in dist.items())


def v_kond(f, x, S, dist=DIST):
    num = sum(p * f(b) for b, p in dist.items() if all(b[j] == x[j] for j in S))
    den = sum(p for b, p in dist.items() if all(b[j] == x[j] for j in S))
    return num / den


# Bäume: Blatt = {"wert", "cover"}, innerer Knoten = {"f", "links" (Merkmal = 0), "rechts" (Merkmal = 1), "cover"}
BAUM1 = {"f": 0, "cover": Fr(1), "links": {"wert": Fr(1), "cover": Fr(1, 2)},
         "rechts": {"f": 1, "cover": Fr(1, 2), "links": {"wert": Fr(3), "cover": Fr(1, 4)}, "rechts": {"wert": Fr(5), "cover": Fr(1, 4)}}}
BAUM2 = {"f": 1, "cover": Fr(1), "rechts": {"wert": Fr(5), "cover": Fr(1, 4)},
         "links": {"f": 0, "cover": Fr(3, 4), "links": {"wert": Fr(1), "cover": Fr(1, 2)}, "rechts": {"wert": Fr(3), "cover": Fr(1, 4)}}}


def baum_wert(b, x):
    while "wert" not in b:
        b = b["rechts"] if x[b["f"]] == 1 else b["links"]
    return b["wert"]


def baum_pfad_wert(b, x, S):
    """Pfadabhängiges v(S): Knoten mit Merkmal in S folgen x, sonst Mischung der Äste nach Trainingsanteilen (Knoten-Rekursion, Algorithmus 1)."""
    if "wert" in b:
        return b["wert"]
    if b["f"] in S:
        return baum_pfad_wert(b["rechts"] if x[b["f"]] == 1 else b["links"], x, S)
    return sum(c["cover"] / b["cover"] * baum_pfad_wert(c, x, S) for c in (b["links"], b["rechts"]))


def loo(j: int) -> Fr:
    """Leave-one-out (Verlustanstieg des besten Vorhersagers ohne Merkmal j): gewichtete mittlere quadratische Abweichung der Gruppenmittel der übrigen Merkmale gegen μ."""
    rest = [i for i in (0, 1) if i != j]
    gruppen = {}
    for p, w in DIST.items():
        gruppen.setdefault(tuple(p[i] for i in rest), []).append((p, w))
    verlust = Fr(0)
    for g in gruppen.values():
        wsum = sum(w for _, w in g)
        mean = sum(w * MU(p) for p, w in g) / wsum
        verlust += sum(w * (MU(p) - mean) ** 2 for p, w in g)
    return verlust


@dataclass(frozen=True)
class Mini:
    x: tuple
    grundwert: Fr
    vorhersage: Fr
    antworten: dict              # Name -> {Merkmal: Beitrag} für die Zeile (Import, Zollhalt)
    wichtigkeit: dict            # Name -> (Art, Zoll): Mittel von |phi| über die Verteilung
    auswendig: dict              # Baum -> Vorhersage außerhalb der Daten (Export mit Zollhalt)
    loo: dict                    # Merkmal -> Verlustanstieg


def mini() -> Mini:
    x = (1, 1)
    funcs = {
        "interventional (μ)": lambda p: shapley([0, 1], lambda S: v_interv(MU, p, S)),
        "konditional (μ)": lambda p: shapley([0, 1], lambda S: v_kond(MU, p, S)),
        "pfadabhängig, Baum 1": lambda p: shapley([0, 1], lambda S: baum_pfad_wert(BAUM1, p, S)),
        "pfadabhängig, Baum 2": lambda p: shapley([0, 1], lambda S: baum_pfad_wert(BAUM2, p, S)),
        "interventional, Baum 1": lambda p: shapley([0, 1], lambda S: v_interv(lambda z: baum_wert(BAUM1, z), p, S)),
        "interventional, Baum 2": lambda p: shapley([0, 1], lambda S: v_interv(lambda z: baum_wert(BAUM2, z), p, S)),
    }
    antw = {k: f(x) for k, f in funcs.items()}
    wicht = {k: tuple(sum(w * abs(f(p)[j]) for p, w in DIST.items()) for j in (0, 1)) for k, f in funcs.items()}
    return Mini(x, v_interv(MU, x, frozenset()), MU(x), antw, wicht, {"Baum 1": baum_wert(BAUM1, (0, 1)), "Baum 2": baum_wert(BAUM2, (0, 1))}, {"Art": loo(0), "Zoll": loo(1)})


def bruch(x) -> str:
    x = Fr(x)
    s = str(x) if x.denominator != 1 else str(x.numerator)
    return s.replace("-", "−")
