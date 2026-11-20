"""Plotly-Abbildungen (alle Achsen fest, fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import wfk_constants as C
import wfk_evaluation as E
import wfk_twin as TW

_LAYOUT = dict(margin=dict(l=50, r=20, t=50, b=50), legend=dict(orientation="h", y=-0.22), template="plotly_white")


def _fix(fig, **kw):
    fig.update_layout(**_LAYOUT, **kw)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_zeile(ez: E.EineZeile):
    fig = go.Figure()
    fig.add_bar(x=list(E.NAMEN5), y=[float(v) for v in ez.phi_i], name=f"interventional ({ez.hintergrund})", marker_color=C.COLOR_INT)
    fig.add_bar(x=list(E.NAMEN5), y=[float(v) for v in ez.phi_k], name="konditional (exakt)", marker_color=C.COLOR_KOND)
    fig.add_bar(x=list(E.NAMEN5), y=[float(v) for v in ez.phi_p], name="pfadabhängig (Ersatzbaum)", marker_color=C.COLOR_PFAD)
    fig.update_yaxes(title="Beitrag (Log-Einheiten)")
    return _fix(fig, title=f"Zeile {ez.zeile}: dieselbe Funktion, drei Antworten", barmode="group", height=360)


def build_wichtigkeit_bev(wb: E.WichtigkeitBev):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35], subplot_titles=("Wichtigkeit über die exakte Bevölkerung", "Leave-one-out (Verlustanstieg, exakt)"))
    fig.add_bar(x=list(E.NAMEN5), y=wb.interventional, name="interventional", marker_color=C.COLOR_INT, row=1, col=1)
    fig.add_bar(x=list(E.NAMEN5), y=wb.konditional, name="konditional", marker_color=C.COLOR_KOND, row=1, col=1)
    import wfk_truth as T
    fig.add_bar(x=list(E.NAMEN5), y=T.loo_exakt(), name="Leave-one-out", marker_color=C.COLOR_LOO, row=1, col=2)
    fig.update_yaxes(title_text="mittleres |φ|", row=1, col=1)
    fig.update_yaxes(title_text="Verlustanstieg (MSE gegen μ)", row=1, col=2)
    return _fix(fig, barmode="group", height=360)


def build_zwei(zf: E.ZweiFunktionen):
    fig = go.Figure()
    fig.add_bar(x=list(E.NAMEN5), y=zf.wichtigkeit_a, name=zf.name_a, marker_color=C.COLOR_INT if zf.funktional == 0 else C.COLOR_PFAD)
    fig.add_bar(x=list(E.NAMEN5), y=zf.wichtigkeit_b, name=zf.name_b, marker_color=C.COLOR_MODELL)
    fig.update_yaxes(title="mittleres |φ| über die Instanzen")
    return _fix(fig, title=f"Beiträge beider Funktionen (Menge {zf.menge + 1}, {zf.n} Instanzen)", barmode="group", height=360)


def build_modelle(m: E.Modelle):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35], subplot_titles=(f"Wichtigkeit je Funktional ({C.KLASSEN[m.klasse]})", "Leave-one-out (Mittel über Zufalls-Seeds)"))
    farben = {"interventional": C.COLOR_INT, "konditional": C.COLOR_KOND, "pfadabhängig": C.COLOR_PFAD}
    for fn, w in m.wichtigkeit.items():
        fig.add_bar(x=list(E.NAMEN5), y=w, name=fn, marker_color=farben[fn], row=1, col=1)
    fig.add_bar(x=list(E.NAMEN5), y=m.loo, name="Leave-one-out", marker_color=C.COLOR_LOO, row=1, col=2)
    fig.update_yaxes(title_text="mittleres |φ|", row=1, col=1)
    fig.update_yaxes(title_text="Verlustanstieg", row=1, col=2)
    return _fix(fig, barmode="group", height=360)


def build_zwilling(rho: float, q: float):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Gauß-Zwilling: Anteil von x_B", "binärer Zwilling: Anteil von x_2"))
    rs = np.linspace(0.0, C.RHO_MAX, 100)
    fig.add_trace(go.Scatter(x=rs, y=[TW.gauss_anteil_kond(float(r)) for r in rs], mode="lines", name="konditional", line=dict(color=C.COLOR_KOND)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, C.RHO_MAX], y=[0, 0], mode="lines", name="interventional", line=dict(color=C.COLOR_INT)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[rho], y=[TW.gauss_anteil_kond(rho)], mode="markers", marker=dict(size=10, color="black"), name="gewählt", showlegend=False), row=1, col=1)
    qs = np.linspace(0.5, C.Q_MAX, 100)
    fig.add_trace(go.Scatter(x=qs, y=[TW.binaer_anteil_kond(float(x)) for x in qs], mode="lines", line=dict(color=C.COLOR_KOND), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0.5, C.Q_MAX], y=[0, 0], mode="lines", line=dict(color=C.COLOR_INT), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[q], y=[TW.binaer_anteil_kond(q)], mode="markers", marker=dict(size=10, color="black"), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="Korrelation ρ", row=1, col=1)
    fig.update_xaxes(title_text="Wahrscheinlichkeit q", row=1, col=2)
    fig.update_yaxes(title_text="Anteil an der Wichtigkeit", range=[-0.02, 0.55])
    return _fix(fig, height=340)
