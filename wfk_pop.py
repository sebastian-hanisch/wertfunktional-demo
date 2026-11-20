"""Bevölkerung des Generators und die Wertfunktionale (interventional, konditional) der Wahrheit je Funktional.

Bevölkerung: die gemeinsame Verteilung der fünf Merkmale mit Wirkung (Art, Kunde, Zoll, Wochenende, Vorlauf) als gewichtete Stützpunkte (Vorlauf über
Gauß-Legendre); das Rauschmerkmal Gewicht ist unabhängig, ein Nullspieler der wahren Mitte, und steht in den Stützpunkten als 0.
Wahrheit je Funktional = Shapley-Wert der wahren Mitte mu unter genau diesem Wertfunktional:
  interventional  v(S) = Σ_b w_b mu(x_S, b_Rest)             (Hintergrund b: die Bevölkerung)
  konditional     v(S) = E[mu(X) | X_S = x_S]                (gemeinsame Verteilung des Generators, exakt bis auf die Quadratur über den Vorlauf)
Shapley-Wert = gewichtetes Mittel der Grenzbeiträge über alle Koalitionen (5 Spieler, 32 Koalitionen)."""

import itertools
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

import wfk_constants as C

D5 = C.D_WIRKUNG
_MASKS = np.array([[(k >> j) & 1 for j in range(D5)] for k in range(1 << D5)], dtype=bool)
_SIZE = _MASKS.sum(axis=1)
_W = np.array([factorial(s) * factorial(D5 - s - 1) / factorial(D5) if s < D5 else 0.0 for s in range(D5 + 1)])
PA = np.array(C.ART_P)


def vorlauf_knoten(lo: float, hi: float, q: int, regel: str = "gl"):
    """Knoten und Gewichte (Summe 1) für den Vorlauf, gleichverteilt auf [lo, hi]: Gauß-Legendre ("gl") oder Mittelpunktsregel ("mid")."""
    if regel == "gl":
        x, w = leggauss(q)
        return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * w
    h = (hi - lo) / q
    return lo + (np.arange(q) + 0.5) * h, np.full(q, 1.0 / q)


def bevoelkerung(q: int = C.Q_POP, regel: str = "gl"):
    """(X (N, 6), w (N,)): gewichtete Stützpunkte der Verteilung der Merkmale (Spalte Gewicht = 0); die Gewichte summieren sich zu 1."""
    X, w = [], []
    for a in range(4):
        for k in range(C.K_KUNDEN):
            zoll_opts = [(0, 1.0)] if a != 0 else [(0, 1 - C.P_ZOLL), (1, C.P_ZOLL)]
            for z, pz in zoll_opts:
                for we, pw in ((0, 1 - C.P_WOCHENENDE), (1, C.P_WOCHENENDE)):
                    if a == 1:
                        vs, wv = vorlauf_knoten(1.0, 10.0, q, regel)
                    elif a == 2:
                        vs, wv = vorlauf_knoten(0.5, 6.0, q, regel)
                    else:
                        vs, wv = np.array([0.0]), np.array([1.0])
                    for v, wvv in zip(vs, wv):
                        X.append([a, k, z, we, v, 0.0])
                        w.append(PA[a] / C.K_KUNDEN * pz * pw * wvv)
    return np.array(X, float), np.array(w, float)


def shapley_aus_werten(v: np.ndarray) -> np.ndarray:
    """(ne, 32) Koalitionswerte (Bitmaske über die Spalten 0 bis 4) -> (ne, 5) Shapley-Werte (Koalitionsformel)."""
    phi = np.zeros((v.shape[0], D5))
    for j in range(D5):
        for k in range(1 << D5):
            if not (k >> j) & 1:
                phi[:, j] += _W[_SIZE[k]] * (v[:, k | (1 << j)] - v[:, k])
    return phi


def werte_interventional(f, Xe, Xb, wb, chunk: int = 400_000) -> np.ndarray:
    """Interventionale Koalitionswerte (ne, 32): v(S) = Σ w_b f(x_S, b_Rest) mit gewichtetem Hintergrund (Xb, wb); f bekommt Zeilen mit 6 Spalten."""
    ne, nb = len(Xe), len(Xb)
    out = np.zeros((ne, 1 << D5))
    step = max(1, chunk // ((1 << D5) * nb))
    for a in range(0, ne, step):
        xs = Xe[a:a + step]
        H = np.where(_MASKS[None, :, None, :], xs[:, None, None, :D5], Xb[None, None, :, :D5])
        Z = np.zeros(H.shape[:-1] + (C.D,))
        Z[..., :D5] = H
        out[a:a + step] = f(Z.reshape(-1, C.D)).reshape(len(xs), 1 << D5, nb) @ wb
    return out


def _art_posterior(X, beobachtet):
    """Posterior der Art je Zeile gegeben die beobachteten Spalten: (n, 4)."""
    n = len(X)
    if beobachtet[0]:
        W = np.zeros((n, 4))
        W[np.arange(n), X[:, 0].astype(int)] = 1.0
        return W
    W = np.tile(PA, (n, 1))
    if beobachtet[2]:
        z = X[:, 2]
        L = np.ones((n, 4))
        L[:, 0] = np.where(z == 1, C.P_ZOLL, 1 - C.P_ZOLL)
        L[:, 1:] = np.where(z[:, None] == 1, 0.0, 1.0)
        W = W * L
    if beobachtet[4]:
        v = X[:, 4]
        L = np.zeros((n, 4))
        L[:, 0] = v == 0
        L[:, 3] = v == 0
        L[:, 1] = np.where((v >= 1) & (v <= 10), 1 / 9, 0)
        L[:, 2] = np.where((v >= 0.5) & (v <= 6), 1 / 5.5, 0)
        W = W * L
    s = W.sum(axis=1, keepdims=True)
    return W / np.where(s > 0, s, 1.0)


def bedingte_erwartung(f, X, beobachtet, q: int = C.Q_POP, regel: str = "gl", chunk: int = 600_000) -> np.ndarray:
    """E[f(X) | X_beobachtet = x_beobachtet] exakt unter der Verteilung des Generators (Spalten 0 bis 4, Quadratur über den Vorlauf); (n,)."""
    n = len(X)
    W = _art_posterior(X, beobachtet)
    total = np.zeros(n)
    for a in range(4):
        if W[:, a].max() <= 0:
            continue
        opts = [[(np.full(n, float(a)), 1.0)]]
        opts.append([(X[:, 1], 1.0)] if beobachtet[1] else [(np.full(n, float(k)), 1 / C.K_KUNDEN) for k in range(C.K_KUNDEN)])
        opts.append([(X[:, 2], 1.0)] if beobachtet[2] else ([(np.zeros(n), 1 - C.P_ZOLL), (np.ones(n), C.P_ZOLL)] if a == 0 else [(np.zeros(n), 1.0)]))
        opts.append([(X[:, 3], 1.0)] if beobachtet[3] else [(np.zeros(n), 1 - C.P_WOCHENENDE), (np.ones(n), C.P_WOCHENENDE)])
        if beobachtet[4]:
            opts.append([(X[:, 4], 1.0)])
        elif a in (1, 2):
            xs, ws = vorlauf_knoten(*((1.0, 10.0) if a == 1 else (0.5, 6.0)), q, regel)
            opts.append([(np.full(n, x), w) for x, w in zip(xs, ws)])
        else:
            opts.append([(np.zeros(n), 1.0)])
        batch, wts, acc = [], [], np.zeros(n)

        def leeren():
            nonlocal batch, wts
            if not batch:
                return 0.0
            res = (np.array(wts)[:, None] * f(np.vstack(batch)).reshape(len(wts), n)).sum(axis=0)
            batch, wts = [], []
            return res

        for combo in itertools.product(*opts):
            Xc = np.zeros((n, C.D))
            w = 1.0
            for j, (val, wt) in enumerate(combo):
                Xc[:, j] = val
                w *= wt
            batch.append(Xc)
            wts.append(w)
            if len(batch) * n >= chunk:
                acc = acc + leeren()
        acc = acc + leeren()
        total += W[:, a] * acc
    return total


def werte_konditional(f, Xe, q: int = C.Q_POP, regel: str = "gl") -> np.ndarray:
    """Konditionale Koalitionswerte (ne, 32)."""
    return np.column_stack([bedingte_erwartung(f, Xe, list(_MASKS[k]), q, regel) for k in range(1 << D5)])


def wahrheit_je_funktional(f, Xe, Xb=None, wb=None, q: int = C.Q_POP):
    """(phi_interventional, phi_konditional), je (ne, 6) mit Nullspalte Gewicht, für die wahre Mitte f."""
    if Xb is None:
        Xb, wb = bevoelkerung(q)
    pad = lambda p: np.column_stack([p, np.zeros(len(p))])
    return pad(shapley_aus_werten(werte_interventional(f, Xe, Xb, wb))), pad(shapley_aus_werten(werte_konditional(f, Xe, q)))


def shapley_aufzaehlung(f, x, B, w=None) -> tuple:
    """Exakter interventionaler Shapley-Wert für beliebig viele Merkmale d (Aufzählung aller 2^d Koalitionen) für EINE Zeile x gegen den Hintergrund B.

    Rückgabe (phi (d,), v(leer), v(alle)). Gewichte w (Summe 1) optional, sonst gleiches Gewicht. f bekommt Zeilen mit d Spalten."""
    x = np.asarray(x, float)
    B = np.asarray(B, float)
    d = len(x)
    K = 1 << d
    masks = np.array([[(k >> j) & 1 for j in range(d)] for k in range(K)], dtype=bool)
    size = masks.sum(axis=1)
    wk = np.array([factorial(s) * factorial(d - s - 1) / factorial(d) if s < d else 0.0 for s in range(d + 1)])
    H = np.where(masks[:, None, :], x[None, None, :], B[None, :, :])
    wb = np.full(len(B), 1.0 / len(B)) if w is None else np.asarray(w)
    vals = f(H.reshape(-1, d)).reshape(K, -1) @ wb
    phi = np.zeros(d)
    for j in range(d):
        for k in range(K):
            if not masks[k, j]:
                phi[j] += wk[size[k]] * (vals[k | (1 << j)] - vals[k])
    return phi, float(vals[0]), float(vals[K - 1])


def werte_interventional_art(f, Xe, Xb, wb) -> np.ndarray:
    """Wie `werte_interventional`, aber der Hintergrund ist auf die Zeilen der Art der Instanz eingeschränkt (Gewichte neu normiert): das Modell wird nur an Kombinationen
    befragt, die es in dieser Art gibt. Der Beitrag des Merkmals Art ist dort per Konstruktion 0."""
    out = np.zeros((len(Xe), 1 << D5))
    art_e, art_b = Xe[:, 0].astype(int), Xb[:, 0].astype(int)
    for a in range(4):
        me, mb = art_e == a, art_b == a
        if me.any() and mb.any():
            out[me] = werte_interventional(f, Xe[me], Xb[mb], wb[mb] / wb[mb].sum())
    return out


def shapley_aus_permutationen(v: np.ndarray) -> np.ndarray:
    """Anderer Rechenweg als `shapley_aus_werten`: Mittel der Grenzbeiträge über alle 120 Reihenfolgen der fünf Spieler ((ne, 32) -> (ne, 5))."""
    phi = np.zeros((v.shape[0], D5))
    perms = list(itertools.permutations(range(D5)))
    for pm in perms:
        k = 0
        for j in pm:
            phi[:, j] += v[:, k | (1 << j)] - v[:, k]
            k |= 1 << j
    return phi / len(perms)
