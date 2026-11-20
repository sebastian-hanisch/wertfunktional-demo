"""Eingefrorene Wahrheit je Funktional (wfk_truth.json), eingefrorene Modelle (wfk_models.json) und der Ersatzbaum."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import wfk_constants as C
import wfk_generator as G
import wfk_pop as P
import wfk_tree as TR

WURZEL = Path(__file__).resolve().parent
DATEI_TRUTH = WURZEL / "wfk_truth.json"
DATEI_MODELS = WURZEL / "wfk_models.json"
DATEI_PFAD = WURZEL / "wfk_pfad.json"


@dataclass(frozen=True)
class Zeilen:
    """Eine Zeilenmenge mit der Wahrheit je Funktional der wahren Mitte (Spalten Art, Kunde, Zoll, Wochenende, Vorlauf; Gewicht ist Nullspieler)."""

    name: str
    X: np.ndarray
    w: np.ndarray | None
    phi_i: np.ndarray
    phi_k: np.ndarray
    phi_i_art: np.ndarray | None = None

    @property
    def gewichte(self) -> np.ndarray:
        return self.w if self.w is not None else np.full(len(self.X), 1.0 / len(self.X))


@lru_cache(maxsize=1)
def _truth() -> dict:
    return json.loads(DATEI_TRUTH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _models() -> dict:
    return json.loads(DATEI_MODELS.read_text(encoding="utf-8"))


def _arr(x):
    return np.array(x, dtype=float)


@lru_cache(maxsize=1)
def bevoelkerung_zeilen() -> Zeilen:
    Xp, wp = P.bevoelkerung()
    r = _truth()["bevoelkerung"]
    return Zeilen("exakte Bevölkerung", Xp, wp, _arr(r["phi_interventional"]), _arr(r["phi_konditional"]))


def _satz(d, name, mit_art=False) -> Zeilen:
    return Zeilen(name, G.zeilenmenge(d["seed"], d["n"]), None, _arr(d["phi_interventional"]), _arr(d["phi_konditional"]), _arr(d["phi_interventional_art"]) if mit_art else None)


@lru_cache(maxsize=1)
def e1_zeilen() -> Zeilen:
    return _satz(_truth()["e1"], "Menge 1 (300 Zeilen, Zufalls-Seed 4242)")


@lru_cache(maxsize=5)
def e2_zeilen(i: int) -> Zeilen:
    d = _truth()["e2"][i]
    return _satz(d, f"Menge {i + 1} ({d['n']} Zeilen, Zufalls-Seed {d['seed']})")


@lru_cache(maxsize=5)
def e3_zeilen(i: int) -> Zeilen:
    d = _truth()["e3"][i]
    return _satz(d, f"Menge {i + 1} ({d['n']} Zeilen, Zufalls-Seed {d['seed']})", mit_art=True)


@dataclass(frozen=True)
class Ersatz:
    baum: TR.Tree
    n_blaetter: int
    r2: float
    phi_i: np.ndarray            # (5 * 150, 5) interventionaler Beitrag des Baums für die fünf Mengen zu 150 Zeilen hintereinander


@lru_cache(maxsize=1)
def ersatzbaum() -> Ersatz:
    d = _models()["ersatzbaum"]
    return Ersatz(TR.Tree.aus_dict(d["baum"]), d["n_blaetter"], d["r2"], _arr(d["phi_interventional"]))


@dataclass(frozen=True)
class Klasse:
    name: str
    r2: np.ndarray                       # (8,)
    phi: dict                            # Funktional -> (8 Zufalls-Seeds, 200, 5)
    loo: np.ndarray                      # (8, 5) Verlustanstieg


@lru_cache(maxsize=3)
def klasse(k: int) -> Klasse:
    d = _models()["klassen"][C.KLASSEN[k]]
    return Klasse(C.KLASSEN[k], _arr(d["r2"]), {fn: _arr(v) for fn, v in d["phi"].items()}, _arr(d["loo"]))


def loo_exakt() -> np.ndarray:
    return _arr(_models()["loo_exakt"])


def stichprobe(n: int):
    """Feste Ziehung natürlicher Zeilen als Hintergrund-Stichprobe (Zufalls-Seed 9100): (X (n, 6), w gleich)."""
    X, _ = G.merkmale(C.SET_OFFSET + C.HG_STICHPROBE_SEED, n)
    return X, np.full(n, 1.0 / n)


@lru_cache(maxsize=1)
def pfad_baum_alle() -> np.ndarray:
    """Pfadabhängiger Beitrag des Ersatzbaums (eingefroren, tools/wfk_make_pfad.py): (750, 5), die fünf Mengen zu 150 Zeilen hintereinander."""
    return _arr(json.loads(DATEI_PFAD.read_text(encoding="utf-8"))["phi_pfadabhaengig"])
