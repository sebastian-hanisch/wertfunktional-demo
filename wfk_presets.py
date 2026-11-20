"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

from dataclasses import dataclass

import streamlit as st

import wfk_constants as C


def _bool(text) -> bool:
    return str(text).strip().lower() in ("1", "true", "ja")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: object = None
    hi: object = None


SETTING_SPECS = {
    "zeile_slider": SettingSpec("zeile", int, C.DEFAULT_ZEILE, C.ZEILE_MIN, C.ZEILE_MAX),
    "hintergrund": SettingSpec("hg", int, C.HG_BEVOELKERUNG, 0, len(C.HG_OPTIONEN) - 1),
    "e2_funktional": SettingSpec("e2f", int, 0, 0, 1),
    "e2_n": SettingSpec("e2n", int, C.DEFAULT_E2_N, min(C.E2_N_OPTIONEN), max(C.E2_N_OPTIONEN)),
    "e2_menge": SettingSpec("e2m", int, 0, 0, len(C.E2_SEEDS) - 1),
    "klasse": SettingSpec("klasse", int, C.KLASSE_FOREST, 0, len(C.KLASSEN) - 1),
    "e3_menge": SettingSpec("e3m", int, 0, 0, len(C.E3_MENGEN) - 1),
    "e3_hintergrund": SettingSpec("e3h", int, 0, 0, 1),
    "rho": SettingSpec("rho", float, C.DEFAULT_RHO, C.RHO_MIN, C.RHO_MAX),
    "q": SettingSpec("q", float, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "mini": SettingSpec("mini", _bool, False, False, True),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def einrasten(value: int) -> int:
    """Nächste Instanzzahl der Auswahl (Permalink mit krummer Zahl)."""
    return min(C.E2_N_OPTIONEN, key=lambda o: abs(o - value))


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt (n eingerastet)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = max(spec.lo, min(spec.hi, value))
            if state_key == "e2_n":
                value = einrasten(int(value))
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt JEDEN Regler: den im Preset genannten Wert, sonst den Standardwert (ein Preset meint für nicht genannte Regler die Voreinstellung)."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


