"""Dieselbe Frage, andere Wahrheit: vier Wertfunktionale - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 2 der Reihe „Beitrag und Beleg“ (Feld „Modelle verstehen“). Vehikel: erfundener Containerstrom mit bekannter wahrer Mitte; Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import wfk_constants as C
import wfk_evaluation as E
import wfk_hand as H
import wfk_texts as TX
import wfk_visualization as V
from wfk_layout import dunkles_design, kopf_css
from wfk_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Dieselbe Frage, andere Wahrheit – Sebastian Hanisch", layout="wide")

# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in wfk_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⚖️ Dieselbe Frage, andere Wahrheit: vier Wertfunktionale")
st.markdown(TX.INTRO_MARKDOWN)

with st.expander("So funktioniert ein Beitrag je Wertfunktional", expanded=True):
    st.markdown(TX.SO_FUNKTIONIERT_MARKDOWN)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("1️⃣ Eine Zeile")
    st.slider("Zeile (aus Menge 1, 300 Zeilen)", *bounds("zeile_slider"), key="zeile_slider", help="Welche der 300 eingefrorenen Zeilen erklärt wird.")
    st.selectbox("Hintergrund des interventionalen Funktionals", list(range(len(C.HG_OPTIONEN))), format_func=lambda i: C.HG_OPTIONEN[i], key="hintergrund",
                 help="Aus welchen Zeilen die nicht festgehaltenen Merkmale gezogen werden. Die Stichproben sind feste Ziehungen (Zufalls-Seed 9100); „derselben Art“ schränkt die Bevölkerung auf die Art der Zeile ein.")
    st.header("2️⃣ Zwei Funktionen")
    st.radio("Vergleich", [0, 1], format_func=lambda i: ("μ gegen Ersatzbaum, beide interventional", "Baum: pfadabhängig gegen interventional")[i], key="e2_funktional")
    st.select_slider("Instanzen", options=list(C.E2_N_OPTIONEN), key="e2_n", help="Wie viele Zeilen der gewählten Menge in den Vergleich eingehen.")
    st.selectbox("Instanzmenge (150 Zeilen)", list(range(len(C.E2_SEEDS))), format_func=lambda i: f"Menge {i + 1} (Zufalls-Seed {C.E2_SEEDS[i]})", key="e2_menge")
    st.header("3️⃣ Modellklassen")
    st.selectbox("Klasse", list(range(len(C.KLASSEN))), format_func=lambda i: C.KLASSEN[i], key="klasse", help="Eingefrorene Modelle aus scikit-learn (acht Zufalls-Seeds, n = 1 000).")
    st.selectbox("Instanzmenge (40 Zeilen)", list(range(len(C.E3_MENGEN))), format_func=lambda i: C.E3_MENGEN[i], key="e3_menge")
    st.radio("Hintergrund", [0, 1], format_func=lambda i: ("Bevölkerung", "derselben Art")[i], key="e3_hintergrund")
    st.header("4️⃣ Zwillinge")
    st.slider("Korrelation ρ (Gauß-Fall)", *bounds("rho"), key="rho", step=0.001, format="%.3f")
    st.slider("Wahrscheinlichkeit q (binärer Fall)", *bounds("q"), key="q", step=0.001, format="%.3f")
    st.toggle("Mini-Tabelle in Brüchen zeigen", key="mini", help="Handbeispiel: dieselbe Funktion, zwei Bäume, vier Antworten.")

S = {k: st.session_state[k] for k in ("zeile_slider", "hintergrund", "e2_funktional", "e2_n", "e2_menge", "klasse", "e3_menge", "e3_hintergrund", "rho", "q", "mini")}
sync_query_params(S)

st.markdown("## ❓ Kernfragen")
st.markdown("""
1. **Welche Frage stellt das Funktional?** → Experiment 1
2. **Ändert sich der Beitrag derselben Funktion mit der Frage, und hängt er an der Funktion oder an der Baumform?** → Experiment 1 und 2
3. **Welche Wahrheit passt zu welchem Modell?** → Experiment 3
4. **Was passiert bei Zwillingen?** → Experiment 4
""")

# --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 1: Eine Zeile, vier Antworten")
st.caption("Zähleinheit: Instanzen. Eine Zeile (live: 32 Koalitionen mal Hintergrund), darunter die Wichtigkeit über die exakte Bevölkerung (eingefroren). Leave-one-out hat keinen Beitrag je Zeile, nur eine Wichtigkeit.")
ez = E.eine_zeile(int(S["zeile_slider"]), int(S["hintergrund"]))
x = ez.x
st.markdown(f"**Zeile {ez.zeile} der Menge 1:** Art {C.ART_LABELS[int(x[0])]}, Kunde {int(x[1])}, Zollhalt {'ja' if x[2] else 'nein'}, Wochenende {'ja' if x[3] else 'nein'}, Vorlauf {E.fmt(float(x[4]), 2)} Tage. "
            f"Wahre Mitte {E.fmt(ez.mu, 3)}, Ersatzbaum {E.fmt(ez.vorhersage_baum, 3)}.")
st.plotly_chart(V.build_zeile(ez), width="stretch", key="chart_zeile")
st.dataframe([{"Merkmal": E.NAMEN5[j], f"interventional ({ez.hintergrund})": E.fmt(float(ez.phi_i[j]), 4), "konditional (exakt)": E.fmt(float(ez.phi_k[j]), 4),
               "pfadabhängig (Ersatzbaum)": E.fmt(float(ez.phi_p[j]), 4), "interventional (Bevölkerung)": E.fmt(float(ez.phi_i_bev[j]), 4)} for j in range(5)], hide_index=True, width="stretch")
kind, text = E.verdict_zeile(ez)
getattr(st, kind)(text)
wb = E.wichtigkeit_bevoelkerung()
st.plotly_chart(V.build_wichtigkeit_bev(wb), width="stretch", key="chart_wichtigkeit")
kind, text = E.verdict_wichtigkeit(wb)
getattr(st, kind)(text)
zs = [E.ziehungen(m) for m in C.ZIEHUNG_GROESSEN]
kind, text = E.verdict_ziehungen(zs)
getattr(st, kind)(text)

# --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 2: Zwei Funktionen, gleiche Daten")
zf = E.zwei_funktionen(int(S["e2_menge"]), int(S["e2_n"]), int(S["e2_funktional"]))
st.caption(f"Zähleinheit: Instanzen ({zf.n} von {C.E2_N} der Menge {zf.menge + 1}); Funktionen: die wahre Mitte und ein Ersatzbaum auf der wahren Mitte mit {zf.n_blaetter} Blättern, R² gegen die Mitte {E.fmt(zf.r2_baum, 4)}. "
           "Der Baum ist als Modell nicht lesbar; er trennt nur Funktion und Funktional.")
st.plotly_chart(V.build_zwei(zf), width="stretch", key="chart_zwei")
st.markdown(f"Summe der Abweichungen je Instanz zwischen den beiden Beiträgen: **{E.fmt(zf.summe, 3)}**; gleiches größtes Merkmal je Instanz in {E.fmt(zf.gleiches_top * 100, 0)} % der Instanzen.")
kind, text = E.verdict_zwei(zf)
getattr(st, kind)(text)

# --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 3: Welche Wahrheit passt zu welchem Modell?")
m = E.modelle(int(S["klasse"]), int(S["e3_menge"]), int(S["e3_hintergrund"]))
st.caption(f"Zähleinheiten: {C.N_MODELL_SEEDS} Zufalls-Seeds (Trainingsdatensätze, n = {C.N_TRAIN}), Instanzen ({'fünf getrennte Mengen zu je 40' if m.menge == 0 else 'Menge ' + str(m.menge) + ' mit 40'}), Hintergrund: "
           f"{'nur Zeilen derselben Art' if m.hintergrund else 'Bevölkerung'}. Modelle eingefroren (scikit-learn, nur beim Vorrechnen). Summe der Abweichungen je Instanz, Spanne über die Mengen.")
st.plotly_chart(V.build_modelle(m), width="stretch", key="chart_modelle")
rows = [{"Maß": "Beitrag im gleichen Funktional gegen Wahrheit: interventional", "Wert": f"{E.fmt(m.gleich_i[0])} bis {E.fmt(m.gleich_i[1])}"},
        {"Maß": "Beitrag im gleichen Funktional gegen Wahrheit: konditional", "Wert": f"{E.fmt(m.gleich_k[0])} bis {E.fmt(m.gleich_k[1])}"},
        {"Maß": "interventionaler Modellbeitrag gegen konditionale Wahrheit", "Wert": f"{E.fmt(m.int_gegen_wahrheit_k[0])} bis {E.fmt(m.int_gegen_wahrheit_k[1])}"},
        {"Maß": "dasselbe Modell: interventional gegen konditional", "Wert": f"{E.fmt(m.int_gegen_kond[0])} bis {E.fmt(m.int_gegen_kond[1])}"}]
if m.int_gegen_pfad:
    rows += [{"Maß": "dasselbe Modell: interventional gegen pfadabhängig", "Wert": f"{E.fmt(m.int_gegen_pfad[0], 3)} bis {E.fmt(m.int_gegen_pfad[1], 3)}"},
             {"Maß": "dasselbe Modell: konditional gegen pfadabhängig", "Wert": f"{E.fmt(m.kond_gegen_pfad[0], 3)} bis {E.fmt(m.kond_gegen_pfad[1], 3)}"}]
rows += [{"Maß": "gleiches größtes Merkmal je Instanz (interventional gegen konditional)", "Wert": f"{E.fmt(m.gleiches_top[0] * 100, 0)} bis {E.fmt(m.gleiches_top[1] * 100, 0)} %"},
         {"Maß": "Rangkorrelation der Wichtigkeit (interventional gegen konditional)", "Wert": f"{E.fmt(m.rangkorr[0])} bis {E.fmt(m.rangkorr[1])}"},
         {"Maß": f"Rang 1 des interventionalen Modellbeitrags wie der der Wahrheit (Zufalls-Seeds von {C.N_MODELL_SEEDS}, je Menge)", "Wert": ", ".join(str(t) for t in m.rang1_treffer)},
         {"Maß": "Leave-one-out: Rangfolge der Merkmale (größter Verlustanstieg zuerst)", "Wert": " > ".join(E.NAMEN5[i] for i in m.loo_rang)}]
st.dataframe(rows, hide_index=True, width="stretch")
kind, text = E.verdict_modelle(m)
getattr(st, kind)(text)

# --- Experiment 4 ---------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 4: Zwillinge, das Funktional entscheidet")
st.caption("Geschlossene Formeln (Orakel in den Tests: Quadratur und Stichprobe). Verweis: die Streuung der Aufteilung im Wald steht in der `korrelierte-merkmale-demo`.")
st.plotly_chart(V.build_zwilling(float(S["rho"]), float(S["q"])), width="stretch", key="chart_zwilling")
for fall in (0, 1):
    kind, text = E.verdict_zwilling(fall, float(S["rho"]), float(S["q"]))
    getattr(st, kind)(text)

if S["mini"]:
    st.markdown("---")
    st.subheader("🧮 Mini-Tabelle in exakten Brüchen")
    mi = H.mini()
    st.markdown("Import (1) oder Export (0), Zollhalt nur bei Import; wahre Mitte μ = 1 + 2·Art + 2·Art·Zoll; Verteilung (Import, kein Zoll) 1/4, (Import, Zollhalt) 1/4, (Export, kein Zoll) 1/2. "
                "Baum 1 fragt zuerst die Art, Baum 2 zuerst den Zoll; beide treffen die drei Stützpunkte, außerhalb der Daten (Export mit Zollhalt) gibt Baum 1 "
                f"{H.bruch(mi.auswendig['Baum 1'])}, Baum 2 {H.bruch(mi.auswendig['Baum 2'])} (μ: 1).")
    st.dataframe([{"Antwort für (Import, Zollhalt)": k, "Art": H.bruch(v[0]), "Zoll": H.bruch(v[1]), "Wichtigkeit Art": H.bruch(mi.wichtigkeit[k][0]), "Wichtigkeit Zoll": H.bruch(mi.wichtigkeit[k][1])}
                  for k, v in mi.antworten.items()], hide_index=True, width="stretch")
    kind, text = E.verdict_mini()
    getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(TX.KONVENTION_MARKDOWN)
    st.markdown(TX.FORMEL_LESEHILFE)
    st.markdown(TX.FORMEL_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
