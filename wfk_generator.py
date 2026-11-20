"""Erfundener Containerstrom: Merkmale und wahre Mitte mu(x) = E[ln Verweilzeit | x] (analytisch bekannt). Nachbau der verweilzeit-demo mit SplitMix64.

Zufallsstrom: Zeile i nutzt die Ziehungen i*10 bis i*10+9 des SplitMix64-Stroms (Zufalls-Seed = Startzustand); die ersten k Zeilen hängen also nicht von n ab.
Ziehung 0 Art, 1 Kunde, 2 Zoll, 3 Wochentag, 4 Vorlauf, 5 und 6 Gewicht (Box-Muller), 7 und 8 Rauschen (Box-Muller), 9 frei.
Die Ziehungen sind vektorisiert und bitgleich zur Klasse `wfk_rng.SplitMix64` (tests/test_generator.py); die Normalzahlen nutzen numpy-log/-cos und stimmen
mit `SplitMix64.normal()` auf 1e-12 überein (die Letztstelle hängt von der C-Bibliothek ab, deshalb prüfen die Tests Bänder)."""

import numpy as np

import wfk_constants as C

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)
_UK = np.array(C.UK)
_U_SCALE = np.array(C.U_SCALE)
_S_BASE = np.array(C.S_BASE)
_CUM = np.cumsum(C.ART_P)


def uniforms(seed: int, count: int) -> np.ndarray:
    """Die ersten `count` Gleichverteilten in [0, 1) des SplitMix64-Stroms mit Startzustand `seed` (wie `SplitMix64.random()` nacheinander)."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed & ((1 << 64) - 1)) + idx * _GAMMA
        z = (z ^ (z >> np.uint64(30))) * _C1
        z = (z ^ (z >> np.uint64(27))) * _C2
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _box_muller(u1, u2):
    return np.sqrt(-2.0 * np.log(np.maximum(u1, 1e-12))) * np.cos(2.0 * np.pi * u2)


def mitte_und_streuung(art, kunde, zoll, we, vorlauf):
    """Wahre Mitte mu und Streuung s des Rauschens (ln Verweilzeit ~ N(mu, s^2)); vektorisiert, akzeptiert Skalare."""
    art = np.asarray(art).astype(int)
    kunde = np.asarray(kunde).astype(int)
    zoll, we, vorlauf = np.asarray(zoll, float), np.asarray(we, float), np.asarray(vorlauf, float)
    u = _UK[kunde] * _U_SCALE[art]
    med = np.select([art == 0, art == 1, art == 2], [3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vorlauf, 0.3 + 0.8 * vorlauf],
                    default=7.0 * np.exp(0.30 * we))
    return np.log(med) + u, _S_BASE[art]


def wahrheit(X) -> np.ndarray:
    """Wahre Mitte mu(x) für Zeilen mit den Spalten Art, Kunde, Zoll, Wochenende, Vorlauf (weitere Spalten werden nicht gelesen)."""
    X = np.atleast_2d(X)
    return mitte_und_streuung(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])[0]


def merkmale(seed: int, n: int):
    """(X (n, 6), u (n, 10)): Merkmale der ersten n Zeilen des Stroms mit Startzustand `seed`; u sind die Ziehungen der Zeile."""
    u = uniforms(seed, n * C.DRAWS_PER_ROW).reshape(n, C.DRAWS_PER_ROW)
    art = np.minimum(np.searchsorted(_CUM, u[:, 0], side="right"), 3)
    kunde = np.minimum((u[:, 1] * C.K_KUNDEN).astype(int), C.K_KUNDEN - 1)
    zoll = ((u[:, 2] < C.P_ZOLL) & (art == 0)).astype(float)
    we = ((np.floor(u[:, 3] * 7000.0).astype(int) % 7) >= 5).astype(float)
    vorlauf = np.where(art == 1, 1.0 + 9.0 * u[:, 4], np.where(art == 2, 0.5 + 5.5 * u[:, 4], 0.0))
    gewicht = _box_muller(u[:, 5], u[:, 6])
    X = np.column_stack([art, kunde, zoll, we, vorlauf, gewicht]).astype(float)
    return X, u


def daten(n: int, seed: int, bereich: int = C.TRAIN_OFFSET):
    """(X, y): n Zeilen und ln Verweilzeit. `seed` ist der Zufalls-Seed der Anzeige; `bereich` trennt Trainingsdaten, Zeilenmengen und Testsatz."""
    X, u = merkmale(bereich + seed, n)
    mu, s = mitte_und_streuung(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])
    return X, mu + s * _box_muller(u[:, 7], u[:, 8])


def zeilenmenge(seed: int, n: int = C.N_ZEILEN) -> np.ndarray:
    """Natürliche Zeilen einer Zeilenmenge (nur Merkmale)."""
    return merkmale(C.SET_OFFSET + seed, n)[0]


def testsatz():
    X, _ = merkmale(C.SET_OFFSET + C.TEST_SEED, C.TEST_N)
    return X, wahrheit(X)
