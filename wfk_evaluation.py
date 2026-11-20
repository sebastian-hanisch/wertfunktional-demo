"""Messungen und Verdikte: dieselbe Funktion, vier Wertfunktionale (interventional, konditional, pfadabhängig, Leave-one-out).

Zähleinheiten stehen bei jeder Zahl: Instanzen (Zeilen, für die ein Beitrag gerechnet wird; die Zeilenmenge nennt Zahl und Art), Zufalls-Seeds (Trainingsdatensätze), Ziehungen,
Koalitionen, Modellauswertungen. Rangmaß: Rangkorrelation der Wichtigkeit über die fünf Merkmale mit Wirkung (Mittelränge bei Gleichstand), gleiches größtes Merkmal je Instanz,
Rang 1; Betragsmaß: Summe über die fünf Merkmale des mittleren |φ_A − φ_B| je Instanz. Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import wfk_constants as C
import wfk_generator as G
import wfk_pop as P
import wfk_tree as TR
import wfk_truth as T
import wfk_twin as TW
from wfk_hand import bruch  # noqa: F401  (für die App)

NAMEN5 = C.NAMEN[:5]


def fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def mittelraenge(a) -> np.ndarray:
    a = np.asarray(a, float)
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(1, len(a) + 1, dtype=float)
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def rangkorrelation(a, b) -> float:
    ra, rb = mittelraenge(a), mittelraenge(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def wichtigkeit(phi, w=None) -> np.ndarray:
    """Mittleres |φ| je Merkmal (gewichtet, falls w)."""
    a = np.abs(np.asarray(phi)[:, :5])
    return a.mean(axis=0) if w is None else w @ a


def abstand(phi_a, phi_b, w=None) -> float:
    """Betragsmaß: Summe über die fünf Merkmale des mittleren |φ_A − φ_B| je Instanz."""
    d = np.abs(np.asarray(phi_a)[:, :5] - np.asarray(phi_b)[:, :5])
    return float((d.mean(axis=0) if w is None else w @ d).sum())


def gleiches_top(phi_a, phi_b, w=None) -> float:
    g = np.argmax(np.abs(np.asarray(phi_a)[:, :5]), axis=1) == np.argmax(np.abs(np.asarray(phi_b)[:, :5]), axis=1)
    return float(g.mean() if w is None else w @ g)


# ------------------------------------------------------------------------------------------------ Experiment 1: eine Zeile, vier Antworten
def hintergrund(option: int, X_zeile=None):
    """(Xb, wb, name) für die gewählte Hintergrund-Option; 'derselben Art' braucht die Zeile und wird in `eine_zeile` gesondert behandelt."""
    if option in C.HG_GROESSEN:
        X, w = T.stichprobe(C.HG_GROESSEN[option])
        return X, w, C.HG_OPTIONEN[option]
    Xp, wp = P.bevoelkerung()
    return Xp, wp, C.HG_OPTIONEN[option]


@dataclass(frozen=True)
class EineZeile:
    zeile: int
    x: np.ndarray
    option: int
    hintergrund: str
    phi_i: np.ndarray            # (5) interventional, Hintergrund wie gewählt (live)
    phi_k: np.ndarray            # (5) konditional (eingefroren)
    phi_p: np.ndarray            # (5) pfadabhängig auf dem Ersatzbaum (live)
    phi_i_bev: np.ndarray        # (5) interventional mit der Bevölkerung (eingefroren, Vergleich)
    vorhersage_baum: float
    mu: float
    grundwert_i: float
    grundwert_k: float
    loo_exakt: np.ndarray        # (5) Verlustanstieg des besten Vorhersagers (global, keine Zeile)


def eine_zeile(zeile: int, option: int) -> EineZeile:
    z = T.e1_zeilen()
    x = z.X[zeile]
    if option == C.HG_ART:
        Xp, wp = P.bevoelkerung()
        v = P.werte_interventional_art(G.wahrheit, x[None, :], Xp, wp)
        name = f"Bevölkerung, nur Zeilen der Art {C.ART_LABELS[int(x[0])]}"
    else:
        Xb, wb, name = hintergrund(option)
        v = P.werte_interventional(G.wahrheit, x[None, :], Xb, wb)
    er = T.ersatzbaum()
    vp = TR.pfad_werte(er.baum, x[None, :])
    return EineZeile(zeile, x, option, name, P.shapley_aus_werten(v)[0], z.phi_k[zeile, :5], P.shapley_aus_werten(vp)[0], z.phi_i[zeile, :5],
                     float(er.baum.vorhersage(x[None, :])[0]), float(G.wahrheit(x[None, :])[0]), float(v[0, 0]), float(vp[0, 0]), T.loo_exakt())


def verdict_zeile(ez: EineZeile):
    """Eine Zeile: derselbe Wert, vier Antworten; die Summe ist in jedem Funktional Vorhersage minus Grundwert."""
    summe_i = float(ez.phi_i.sum())
    text = (f"Zeile {ez.zeile} der Menge 1, Hintergrund {ez.hintergrund}: der Beitrag der Art ist {fmt(float(ez.phi_i[0]), 3)} interventional, {fmt(float(ez.phi_k[0]), 3)} konditional und "
            f"{fmt(float(ez.phi_p[0]), 3)} pfadabhängig (Ersatzbaum), der des Vorlaufs {fmt(float(ez.phi_i[4]), 3)}, {fmt(float(ez.phi_k[4]), 3)} und {fmt(float(ez.phi_p[4]), 3)}. "
            f"Die Summe der Beiträge ist in jedem Funktional Vorhersage minus Grundwert (interventional {fmt(summe_i, 3)}, wahre Mitte {fmt(ez.mu, 3)} minus {fmt(ez.grundwert_i, 3)}). "
            f"Ein Beitrag ohne genanntes Wertfunktional ist keine Antwort.")
    ok = abs(summe_i - (ez.mu - ez.grundwert_i)) < 1e-9
    return ("info" if ok else "error"), text


@dataclass(frozen=True)
class WichtigkeitBev:
    interventional: np.ndarray
    konditional: np.ndarray
    rangkorrelation: float
    abstand: float
    rang1_i: int
    rang1_k: int


@lru_cache(maxsize=1)
def wichtigkeit_bevoelkerung() -> WichtigkeitBev:
    z = T.bevoelkerung_zeilen()
    wi, wk = wichtigkeit(z.phi_i, z.w), wichtigkeit(z.phi_k, z.w)
    return WichtigkeitBev(wi, wk, rangkorrelation(wi, wk), abstand(z.phi_i, z.phi_k, z.w), int(np.argmax(wi)), int(np.argmax(wk)))


def verdict_wichtigkeit(wb: WichtigkeitBev):
    text = (f"Mittel von |φ| über die Bevölkerung (exakt, Zeilenmenge 1 224 gewichtete Stützpunkte): Art {fmt(float(wb.interventional[0]), 3)} interventional gegen {fmt(float(wb.konditional[0]), 3)} konditional, "
            f"Vorlauf {fmt(float(wb.interventional[4]), 3)} gegen {fmt(float(wb.konditional[4]), 3)}; Rang 1 wechselt von {NAMEN5[wb.rang1_i]} zu {NAMEN5[wb.rang1_k]}. "
            f"Rangmaß: Rangkorrelation der beiden Wichtigkeiten {fmt(wb.rangkorrelation)}; Betragsmaß: Summe der Abweichungen je Zeile {fmt(wb.abstand, 3)}.")
    return "info", text


@dataclass(frozen=True)
class Ziehungen:
    m: int
    art_vor_vorlauf_i: float     # Anteil der Ziehungen mit mittlerem |φ_Art| > mittlerem |φ_Vorlauf| (interventional)
    vorlauf_vor_art_k: float
    art_q05: float
    art_q95: float
    n_ziehungen: int


@lru_cache(maxsize=8)
def ziehungen(m: int) -> Ziehungen:
    """ANZ_ZIEHUNGEN Ziehungen von je m Instanzen aus der exakten Bevölkerung (mit Zurücklegen nach den Gewichten): wie oft steht die Art vor dem Vorlauf?"""
    z = T.bevoelkerung_zeilen()
    n = C.ANZ_ZIEHUNGEN
    idx = np.minimum(np.searchsorted(np.cumsum(z.w) / z.w.sum(), G.uniforms(C.ZIEHUNG_SEED + m, n * m)), len(z.w) - 1).reshape(n, m)
    ai, vi = np.abs(z.phi_i[idx, 0]).mean(axis=1), np.abs(z.phi_i[idx, 4]).mean(axis=1)
    ak, vk = np.abs(z.phi_k[idx, 0]).mean(axis=1), np.abs(z.phi_k[idx, 4]).mean(axis=1)
    return Ziehungen(m, float(np.mean(ai > vi)), float(np.mean(vk > ak)), float(np.quantile(ai, 0.05)), float(np.quantile(ai, 0.95)), n)


def verdict_ziehungen(zs):
    teile = [f"{zi.m} Instanzen: Art vor Vorlauf in {fmt(zi.art_vor_vorlauf_i * 100, 1)} % der Ziehungen (Art {fmt(zi.art_q05, 3)} bis {fmt(zi.art_q95, 3)}), konditional Vorlauf vor Art in {fmt(zi.vorlauf_vor_art_k * 100, 1)} %"
             for zi in zs]
    return "info", ("Bei kleinen Instanzmengen schwankt schon der Rang (je " + f"{zs[0].n_ziehungen:,}".replace(",", " ") + " Ziehungen aus der exakten Bevölkerung, mit Zurücklegen): " + "; ".join(teile) + ".")


# ------------------------------------------------------------------------------------------------ Experiment 2: zwei Funktionen, gleiche Daten
@dataclass(frozen=True)
class ZweiFunktionen:
    menge: int
    n: int
    funktional: int              # 0: μ gegen Ersatzbaum, beide interventional; 1: pfadabhängig gegen interventional auf demselben Baum
    wichtigkeit_a: np.ndarray
    wichtigkeit_b: np.ndarray
    name_a: str
    name_b: str
    summe: float
    gleiches_top: float
    spanne_summe: tuple          # über die fünf Mengen zu 150 Zeilen: (min, max)
    spanne_art: tuple
    r2_baum: float
    n_blaetter: int


def _paar(funktional: int, i: int, n: int):
    z = T.e2_zeilen(i)
    er = T.ersatzbaum()
    sl = slice(150 * i, 150 * i + n)
    phi_baum_i = er.phi_i[sl]
    if funktional == 0:
        return z.phi_i[:n, :5], phi_baum_i, "μ interventional", "Ersatzbaum interventional"
    return T.pfad_baum_alle()[sl], phi_baum_i, "Ersatzbaum pfadabhängig", "Ersatzbaum interventional"


def zwei_funktionen(menge: int, n: int, funktional: int) -> ZweiFunktionen:
    a, b, na, nb = _paar(funktional, menge, n)
    summen, arts = [], []
    for i in range(5):
        pa, pb, _, _ = _paar(funktional, i, C.E2_N)
        summen.append(abstand(pa, pb))
        arts.append(float(np.abs(pa[:, 0] - pb[:, 0]).mean()))
    er = T.ersatzbaum()
    return ZweiFunktionen(menge, n, funktional, wichtigkeit(a), wichtigkeit(b), na, nb, abstand(a, b), gleiches_top(a, b), (min(summen), max(summen)), (min(arts), max(arts)), er.r2, er.n_blaetter)


def verdict_zwei(zf: ZweiFunktionen):
    if zf.funktional == 0:
        text = (f"Zwei Funktionen mit R² {fmt(zf.r2_baum, 4)} gegen die wahre Mitte auf den Daten (Ersatzbaum mit {zf.n_blaetter} Blättern): interventional um {fmt(zf.spanne_summe[0])} bis "
                f"{fmt(zf.spanne_summe[1])} auseinander (Art allein {fmt(zf.spanne_art[0])} bis {fmt(zf.spanne_art[1])}; Summe der Abweichungen je Instanz, fünf getrennte Mengen zu {C.E2_N} Instanzen). "
                f"Die gewählte Menge {zf.menge + 1} mit {zf.n} Instanzen: {fmt(zf.summe)}.")
    else:
        text = (f"Pfadabhängig gegen interventional auf demselben Baum ({zf.n_blaetter} Blätter): Summe der Abweichungen je Instanz {fmt(zf.spanne_summe[0], 3)} bis {fmt(zf.spanne_summe[1], 3)} "
                f"(fünf getrennte Mengen zu {C.E2_N} Instanzen); gewählt Menge {zf.menge + 1} mit {zf.n} Instanzen: {fmt(zf.summe, 3)}. Auf diesem Ersatzbaum geben beide Funktionale sehr ähnliche Antworten: der große Unterschied liegt hier nicht zwischen den Funktionalen, "
                f"sondern zwischen zwei Funktionen mit fast gleichem R² (Vergleich μ gegen Ersatzbaum); auf anderen Bäumen kann das pfadabhängige Funktional weit abweichen (Mini-Tabelle).")
    return "info", text


# ------------------------------------------------------------------------------------------------ Experiment 3: Modellklassen
@dataclass(frozen=True)
class Modelle:
    klasse: int
    menge: int                   # 0 = alle fünf Mengen, sonst 1 bis 5
    hintergrund: int             # 0 Bevölkerung, 1 derselben Art
    r2: float
    gleich_i: tuple              # Spanne über die gewählten Mengen: Modell interventional gegen Wahrheit interventional (Mittel über 8 Zufalls-Seeds)
    gleich_k: tuple              # Modell konditional gegen Wahrheit konditional
    int_gegen_wahrheit_k: tuple  # interventionaler Modellbeitrag gegen konditionale Wahrheit
    int_gegen_kond: tuple        # dasselbe Modell: interventional gegen konditional
    int_gegen_pfad: tuple | None
    kond_gegen_pfad: tuple | None
    gleiches_top: tuple          # interventional gegen konditional
    rangkorr: tuple
    rang1_treffer: tuple         # je Menge: Zufalls-Seeds (von 8), in denen der Rang 1 des interventionalen Modellbeitrags der des interventionalen Wahrheit ist
    wichtigkeit: dict            # Funktional -> (5) Mittel über Zufalls-Seeds und gewählte Mengen
    loo: np.ndarray              # (5) Mittel über Zufalls-Seeds
    loo_rang: tuple
    wahr_rang1: tuple            # je Menge: Rang-1-Merkmal der interventionalen Wahrheit


def _spanne(vals):
    return (float(min(vals)), float(max(vals)))


def modelle(klasse: int, menge: int, hint: int) -> Modelle:
    kl = T.klasse(klasse)
    sets = range(5) if menge == 0 else [menge - 1]
    mit_pfad = "pfadabhaengig" in kl.phi
    fi = "interventional_art" if hint == 1 else "interventional"
    cols = {k: [] for k in ("gi", "gk", "iwk", "ik", "ip", "kp", "top", "rk")}
    rang1, wahr_r1 = [], []
    wi_alle = {"interventional": [], "konditional": [], "pfadabhängig": []}
    for s in sets:
        z = T.e3_zeilen(s)
        sl = slice(C.E3_N * s, C.E3_N * (s + 1))
        wahr_i = z.phi_i_art if hint == 1 else z.phi_i
        per = {k: [] for k in cols}
        treffer = 0
        for sd in range(C.N_MODELL_SEEDS):
            pi, pk = kl.phi[fi][sd][sl], kl.phi["konditional"][sd][sl]
            pp = kl.phi["pfadabhaengig"][sd][sl] if mit_pfad else None
            per["gi"].append(abstand(pi, wahr_i))
            per["gk"].append(abstand(pk, z.phi_k))
            per["iwk"].append(abstand(pi, z.phi_k))
            per["ik"].append(abstand(pi, pk))
            if mit_pfad:
                per["ip"].append(abstand(pi, pp))
                per["kp"].append(abstand(pk, pp))
            per["top"].append(gleiches_top(pi, pk))
            per["rk"].append(rangkorrelation(wichtigkeit(pi), wichtigkeit(pk)))
            treffer += int(np.argmax(wichtigkeit(pi)) == np.argmax(wichtigkeit(wahr_i)))
            wi_alle["interventional"].append(wichtigkeit(pi))
            wi_alle["konditional"].append(wichtigkeit(pk))
            if mit_pfad:
                wi_alle["pfadabhängig"].append(wichtigkeit(pp))
        for k in cols:
            cols[k].append(float(np.nanmean(per[k])) if per[k] else float("nan"))
        rang1.append(treffer)
        wahr_r1.append(int(np.argmax(wichtigkeit(wahr_i))))
    loo = kl.loo.mean(axis=0)
    return Modelle(klasse, menge, hint, float(kl.r2.mean()), _spanne(cols["gi"]), _spanne(cols["gk"]), _spanne(cols["iwk"]), _spanne(cols["ik"]),
                   _spanne(cols["ip"]) if mit_pfad else None, _spanne(cols["kp"]) if mit_pfad else None, _spanne(cols["top"]), _spanne(cols["rk"]), tuple(rang1),
                   {k: np.mean(v, axis=0) for k, v in wi_alle.items() if v}, loo, tuple(int(i) for i in np.argsort(-loo)), tuple(wahr_r1))


def verdict_modelle(m: Modelle):
    name = C.KLASSEN[m.klasse]
    hg = "Hintergrund derselben Art" if m.hintergrund == 1 else "Hintergrund Bevölkerung"
    sets = "fünf Mengen zu 40 Instanzen" if m.menge == 0 else f"Menge {m.menge} mit 40 Instanzen"
    text = (f"{name} (R² gegen die wahre Mitte {fmt(m.r2, 3)}, {C.N_MODELL_SEEDS} Zufalls-Seeds, n = {C.N_TRAIN}, {sets}, {hg}): der interventionale Beitrag liegt {fmt(m.gleich_i[0])} bis {fmt(m.gleich_i[1])} von der "
            f"interventionalen Wahrheit entfernt, der konditionale {fmt(m.gleich_k[0])} bis {fmt(m.gleich_k[1])} von der konditionalen; der interventionale Beitrag liegt {fmt(m.int_gegen_wahrheit_k[0])} bis "
            f"{fmt(m.int_gegen_wahrheit_k[1])} von der konditionalen Wahrheit entfernt (Summe der Abweichungen je Instanz, Spanne über die Mengen). Rang 1 des interventionalen Beitrags wie der der Wahrheit in "
            f"{', '.join(str(t) for t in m.rang1_treffer)} von {C.N_MODELL_SEEDS} Zufalls-Seeds (je Menge): der Rang 1 der Wahrheit wechselt mit der Instanzmenge und mit ihm die Trefferzahl (Betrag und Rangkorrelation je Menge: Tabelle des Experiments 3 in der README).")
    return "info", text


# ------------------------------------------------------------------------------------------------ Experiment 4: Zwillinge
def verdict_zwilling(fall: int, rho: float, q: float):
    """fall 0 Gauß (rho), fall 1 binär (q)."""
    if fall == 0:
        a_k, b_k = TW.gauss_wichtigkeit_kond(rho)
        anteil = TW.gauss_anteil_kond(rho)
        if rho < 0.05:
            zusatz = " Bei ρ = 0 ist der Zwilling kein Zwilling: alle Funktionale geben dasselbe."
        else:
            zusatz = ""
        text = (f"f = x_A mit einem Zwilling x_B, Korrelation ρ = {fmt(rho, 3)} (Gauß-Fall, geschlossen): konditional bekommt der Zwilling {fmt(anteil * 100, 1)} % der Wichtigkeit, interventional 0 %. "
                f"Mittleres |φ|: x_A {fmt(a_k, 3)}, x_B {fmt(b_k, 3)} konditional; x_A {fmt(TW.gauss_wichtigkeit_int()[0], 3)}, x_B 0 interventional.{zusatz}")
    else:
        a_k, b_k = TW.binaer_wichtigkeit_kond(q)
        text = (f"f = x_1 mit x_2 = x_1 mit Wahrscheinlichkeit q = {fmt(q, 3)} (binär, geschlossen): konditional bekommt x_2 {fmt(b_k / (a_k + b_k) * 100, 1)} % der Wichtigkeit "
                f"(E|φ_1| = {fmt(a_k, 3)}, E|φ_2| = {fmt(b_k, 3)}), interventional 0 % (E|φ_1| = 0,500, E|φ_2| = 0).")
        if q <= 0.5 + 1e-12:
            text += " Bei q = 1/2 ist x_2 unabhängig von x_1, kein Zwilling: alle Funktionale geben dasselbe."
    return "info", text


# ------------------------------------------------------------------------------------------------ Mini-Tabelle (Brüche)
def verdict_mini():
    from wfk_hand import mini
    m = mini()
    a = m.antworten
    w = m.wichtigkeit
    text = (f"Mini-Tabelle (Zoll nur bei Import), Zeile (Import, Zollhalt), Grundwert {bruch(m.grundwert)}, Vorhersage {bruch(m.vorhersage)}: Beitrag (Art, Zoll) interventional "
            f"({bruch(a['interventional (μ)'][0])}, {bruch(a['interventional (μ)'][1])}), konditional ({bruch(a['konditional (μ)'][0])}, {bruch(a['konditional (μ)'][1])}), pfadabhängig auf Baum 1 "
            f"({bruch(a['pfadabhängig, Baum 1'][0])}, {bruch(a['pfadabhängig, Baum 1'][1])}), auf Baum 2 ({bruch(a['pfadabhängig, Baum 2'][0])}, {bruch(a['pfadabhängig, Baum 2'][1])}). "
            f"Die Rangfolge von Art und Zoll in der Wichtigkeit kippt zwischen den beiden Bäumen mit gleicher Vorhersage auf den Daten: Baum 1 {bruch(w['pfadabhängig, Baum 1'][0])} gegen "
            f"{bruch(w['pfadabhängig, Baum 1'][1])}, Baum 2 {bruch(w['pfadabhängig, Baum 2'][0])} gegen {bruch(w['pfadabhängig, Baum 2'][1])}. Leave-one-out (Verlustanstieg): Art {bruch(m.loo['Art'])}, Zoll {bruch(m.loo['Zoll'])}.")
    return "warning", text
