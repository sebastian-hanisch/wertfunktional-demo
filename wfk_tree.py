"""Bäume und das pfadabhängige Wertfunktional (Blattanteile, Lundberg u. a. 2020, Algorithmus 1; Stufe W).

Pfadabhängig: v(S) = Erwartung des Baums, wenn an jedem Knoten, dessen Merkmal in S liegt, dem Wert der Zeile gefolgt wird und an jedem anderen Knoten beide Äste nach ihren
Trainingsanteilen (cover) gemischt werden. v(alle) ist die Vorhersage des Baums, v(leer) das Mittel der Blattwerte nach Trainingsanteilen. Das Funktional hängt an der
**Baumform**, nicht nur an der Funktion. Gerechnet wird über die Blätter: P(Blatt | x, S) = Produkt über die Knoten des Pfades (Merkmal in S: 1 oder 0 je nachdem, ob x
den Pfad nimmt, sonst der Anteil des Kindes); Orakel in den Tests: Knoten-Rekursion."""

from dataclasses import dataclass

import numpy as np

import wfk_constants as C

D5 = C.D_WIRKUNG
_MASKS = np.array([[(k >> j) & 1 for j in range(D5)] for k in range(1 << D5)], dtype=bool)


@dataclass(frozen=True)
class Tree:
    """Binärbaum in Feldern wie scikit-learn: Blätter haben feature = -1; x[feature] <= threshold geht nach links; cover = Trainingsanteile je Knoten."""

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray
    cover: np.ndarray

    @staticmethod
    def von_sklearn(t) -> "Tree":
        """Aus `sklearn.tree._tree.Tree` (nur in tools/ und tests/ gebraucht)."""
        return Tree(t.feature.astype(int).copy(), t.threshold.astype(float).copy(), t.children_left.astype(int).copy(), t.children_right.astype(int).copy(),
                    t.value[:, 0, 0].astype(float).copy(), t.weighted_n_node_samples.astype(float).copy())

    @staticmethod
    def aus_dict(d: dict) -> "Tree":
        return Tree(*(np.array(d[k], dtype=float if k in ("threshold", "value", "cover") else int) for k in ("feature", "threshold", "left", "right", "value", "cover")))

    def als_dict(self, nd: int = 12) -> dict:
        return {"feature": self.feature.tolist(), "threshold": [round(float(x), nd) for x in self.threshold], "left": self.left.tolist(), "right": self.right.tolist(),
                "value": [round(float(x), nd) for x in self.value], "cover": [round(float(x), nd) for x in self.cover]}

    @property
    def n_blaetter(self) -> int:
        return int(np.sum(self.left < 0))

    def vorhersage(self, X) -> np.ndarray:
        """Vorhersage für Zeilen X (n, >= 5): vektorisierter Abstieg."""
        X = np.asarray(X)
        node = np.zeros(len(X), dtype=int)
        while True:
            f = self.feature[node]
            inner = f >= 0
            if not inner.any():
                return self.value[node]
            go_left = X[np.arange(len(X)), np.where(inner, f, 0)] <= self.threshold[node]
            node = np.where(inner, np.where(go_left, self.left[node], self.right[node]), node)

    def pfade(self):
        """Je Blatt der Pfad als gepolsterte Felder (L, Dmax): Merkmal (-1 = Polster), Schwelle, ging-links, Anteil des Kindes am Elternknoten; dazu Blattwerte (L,)."""
        blatt, pf, pt, pl, pr = [], [], [], [], []

        def rec(n, fe, th, le, ra):
            if self.left[n] < 0:
                blatt.append(n)
                pf.append(fe), pt.append(th), pl.append(le), pr.append(ra)
                return
            for child, ging_links in ((self.left[n], True), (self.right[n], False)):
                rec(child, fe + [int(self.feature[n])], th + [float(self.threshold[n])], le + [ging_links], ra + [float(self.cover[child] / self.cover[n])])

        rec(0, [], [], [], [])
        L = len(blatt)
        dmax = max(len(p) for p in pf) if L else 0
        F = np.full((L, max(dmax, 1)), -1, dtype=int)
        T_ = np.zeros((L, max(dmax, 1)))
        Le = np.zeros((L, max(dmax, 1)), dtype=bool)
        R = np.ones((L, max(dmax, 1)))
        for i in range(L):
            m = len(pf[i])
            F[i, :m], T_[i, :m], Le[i, :m], R[i, :m] = pf[i], pt[i], pl[i], pr[i]
        return F, T_, Le, R, self.value[np.array(blatt)]


def pfad_werte(tree: Tree, X, chunk: int = 3_000_000) -> np.ndarray:
    """Pfadabhängige Koalitionswerte (n, 32) eines Baums für Zeilen X (Spalten 0 bis 4); Bitmaske über die fünf Merkmale."""
    X = np.asarray(X, float)[:, :D5]
    F, T_, Le, R, val = tree.pfade()
    L, Dm = F.shape
    gueltig = F >= 0
    Fc = np.where(gueltig, F, 0)
    n = len(X)
    out = np.zeros((n, 1 << D5))
    step = max(1, chunk // (L * Dm * (1 << D5)))
    in_s = _MASKS[:, Fc]                                                     # (32, L, Dm): Merkmal des Knotens in der Koalition?
    for a in range(0, n, step):
        xs = X[a:a + step]
        nimmt = ((xs[:, Fc] <= T_[None]) == Le[None]) | ~gueltig[None]       # (m, L, Dm): x nimmt den Pfad an diesem Knoten
        p = np.where(in_s[None], nimmt[:, None].astype(float), R[None, None])    # (m, 32, L, Dm)
        p = np.where(gueltig[None, None], p, 1.0)
        out[a:a + step] = p.prod(axis=3) @ val
    return out


def pfad_werte_ensemble(baeume, gewicht: float, start: float, X) -> np.ndarray:
    """Ensemble: v(S) = start + gewicht * Summe der Baumwerte (Boosting: gewicht = Lernrate; Random Forest: gewicht = 1 / Zahl der Bäume, start = 0)."""
    tot = np.zeros((len(X), 1 << D5))
    for t in baeume:
        tot += pfad_werte(t, X)
    return start + gewicht * tot
