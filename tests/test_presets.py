"""Presets: vollständig, Werte innerhalb der Reglergrenzen, Standard-Preset gleich Standardeinstellung, paarweise verschieden, Hilfetexte; Permalink-Einrasten."""

import pytest

import wfk_constants as C
import wfk_presets as P


def test_jedes_preset_hat_einen_hilfetext_und_nur_bekannte_regler():
    assert set(C.PRESET_HELP) == set(C.PRESETS)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name


def test_preset_werte_liegen_in_den_reglergrenzen():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            assert spec.lo <= value <= spec.hi, (name, key, value)


def test_erstes_preset_ist_die_standardeinstellung():
    assert C.PRESETS[next(iter(C.PRESETS))] == {}


def test_presets_sind_paarweise_verschieden_aufgeloest_gegen_die_standardwerte():
    voll = {name: tuple(preset.get(k, s.default) for k, s in P.SETTING_SPECS.items()) for name, preset in C.PRESETS.items()}
    assert len(set(voll.values())) == len(voll)


def test_standardwerte_liegen_in_den_grenzen_und_e2_n_ist_eine_option():
    for key, spec in P.SETTING_SPECS.items():
        assert spec.lo <= spec.default <= spec.hi, key
    assert C.DEFAULT_E2_N in C.E2_N_OPTIONEN


def test_url_parameter_sind_eindeutig():
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


@pytest.mark.parametrize("roh,erwartet", [(120, 100), (60, 50), (140, 150), (17, 10), (18, 25), (1000, 150)])
def test_permalink_rastet_e2_n_auf_die_naechste_option_ein(roh, erwartet):
    assert P.einrasten(roh) == erwartet


def test_bool_caster_des_permalinks():
    assert P._bool("True") and P._bool("1") and not P._bool("False") and not P._bool("0")


def test_mengen_und_klassen_der_auswahl():
    assert len(C.E3_SEEDS) == 5 and len(C.E2_SEEDS) == 5 and len(C.KLASSEN) == 3 and len(C.HG_OPTIONEN) == 5
    assert C.PRESETS["Boosting"]["klasse"] == C.KLASSE_BOOSTING and C.PRESETS["Netz"]["klasse"] == C.KLASSE_NETZ
    assert P.SETTING_SPECS["klasse"].default == C.KLASSE_FOREST                       # die Voreinstellung ist der Wald: die Presets Boosting und Netz unterscheiden sich von ihr
