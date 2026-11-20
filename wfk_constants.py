"""Konstanten: Vehikel (erfundener Container-Generator), Zeilenmengen, Modelle, Regler-Grenzen, Farben und Presets.

Alles ist erfunden (kein Echtdatenbezug). Der Generator ist aus der verweilzeit-demo KOPIERT (nicht importiert; tests/test_generator.py hält die Kopie an eingefrorene Werte der Quelle).
Die zwölf Kundeneffekte stehen als Literal (Quelle: numpy default_rng(12345).normal(0, 0,30, 12) der verweilzeit-demo)."""

# --- Vehikel ------------------------------------------------------------------------------------------------------------------------------
ART_LABELS = ("Import", "Export", "Umschlag", "Leer")
ART_P = (0.35, 0.30, 0.25, 0.10)
K_KUNDEN = 12
UK = (-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
      -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771)
U_SCALE = (1.0, 0.3, 0.3, 1.0)
S_BASE = (0.55, 0.25, 0.35, 0.90)
P_ZOLL = 0.25
P_WOCHENENDE = 2 / 7
NAMEN = ("Art", "Kunde", "Zoll", "Wochenende", "Vorlauf", "Gewicht")
D = 6
D_WIRKUNG = 5
DRAWS_PER_ROW = 10
TRAIN_OFFSET = 1_000_000
SET_OFFSET = 2_000_000
TEST_SEED = 777
TEST_N = 5000
Q_POP = 24
N_POP = 1224

# --- Zeilenmengen ----------------------------------------------------------------------------------------------------------------------------
N_ZEILEN = 300                                # Menge 1 (Zufalls-Seed 4242): 300 natürliche Zeilen für Experiment 1
SEED_ZEILEN = 4242
E2_SEEDS = (7777, 7778, 7779, 7780, 7781)     # fünf getrennte Instanzmengen zu 150 Zeilen (Experiment 2)
E2_N = 150
E3_SEEDS = (4242, 5301, 5302, 5303, 5304)     # fünf getrennte Instanzmengen zu 40 Zeilen (Experiment 3)
E3_N = 40
HG_STICHPROBE_SEED = 9100                     # feste Ziehung der Hintergrund-Stichproben (Experiment 1)
HG_OPTIONEN = ("Bevölkerung (1 224 Stützpunkte, exakt)", "Stichprobe von 20 Zeilen", "Stichprobe von 100 Zeilen", "Stichprobe von 1 000 Zeilen", "derselben Art wie die Zeile")
HG_BEVOELKERUNG, HG_S20, HG_S100, HG_S1000, HG_ART = range(5)
HG_GROESSEN = {HG_S20: 20, HG_S100: 100, HG_S1000: 1000}

# --- Modelle (Experiment 3), eingefroren aus scikit-learn (tools/wfk_make_models.py) -------------------------------------------------------------
KLASSEN = ("Boosting", "Random Forest", "Kleines Netz")
KLASSE_BOOSTING, KLASSE_FOREST, KLASSE_NETZ = range(3)
N_TRAIN = 1000
N_MODELL_SEEDS = 8                            # Zufalls-Seeds (Trainingsdatensätze) je Klasse
BOOST_TREES, BOOST_DEPTH, BOOST_LR = 200, 3, 0.1
FOREST_TREES, FOREST_MIN_LEAF = 60, 5
NET_HIDDEN = 16
ERSATZ_N, ERSATZ_MIN_LEAF = 100_000, 100      # Ersatzbaum: auf der wahren Mitte, 100 000 natürliche Zeilen, mindestens 100 Zeilen je Blatt
FUNKTIONALE = ("interventional", "konditional", "pfadabhängig", "Leave-one-out")

# --- Regler ----------------------------------------------------------------------------------------------------------------------------------
ZEILE_MIN, ZEILE_MAX, DEFAULT_ZEILE = 0, N_ZEILEN - 1, 0
E2_N_OPTIONEN = (10, 25, 50, 100, 150)
DEFAULT_E2_N = 150
E3_MENGEN = ("alle fünf Mengen", "Menge 1 (Zufalls-Seed 4242)", "Menge 2 (5301)", "Menge 3 (5302)", "Menge 4 (5303)", "Menge 5 (5304)")
RHO_MIN, RHO_MAX, DEFAULT_RHO = 0.0, 0.999, 0.5
Q_MIN, Q_MAX, DEFAULT_Q = 0.5, 0.999, 0.9
MINI_LABELS = ("aus", "an")
ANZ_ZIEHUNGEN = 20_000                        # Ziehungen aus der Bevölkerung für die Spanne bei kleinen Instanzmengen
ZIEHUNG_GROESSEN = (20, 40, 100, 150)
ZIEHUNG_SEED = 31_000_000

COLOR_INT = "#c77700"
COLOR_KOND = "#7a3fb0"
COLOR_PFAD = "#2F6B65"
COLOR_LOO = "#5b6470"
COLOR_MODELL = "#4c78a8"

PRESETS = {
    "Eine Zeile, vier Antworten": {},
    "Zwei Funktionen, gleiche Daten": {"e2_funktional": 1},
    "Zwilling (Gauß)": {"rho": 0.9},
    "Mini-Tabelle (Zoll)": {"mini": True},
    "Boosting": {"klasse": KLASSE_BOOSTING},
    "Netz": {"klasse": KLASSE_NETZ},
    "Hintergrund derselben Art": {"hintergrund": HG_ART, "e3_hintergrund": 1},
}
PRESET_HELP = {
    "Eine Zeile, vier Antworten": "Dieselbe Funktion, eine Zeile, mehrere Wertfunktionale: der Beitrag hängt an der Frage, nicht nur an der Funktion. Voreinstellung.",
    "Zwei Funktionen, gleiche Daten": "Wahre Mitte und ein Ersatzbaum mit fast gleichem R² auf den Daten: pfadabhängig und interventional auf demselben Baum liegen dicht beieinander.",
    "Zwilling (Gauß)": "f = x_A mit einem Zwilling x_B bei Korrelation 0,9: konditional bekommt der Zwilling einen Anteil, interventional keinen.",
    "Mini-Tabelle (Zoll)": "Handbeispiel in Brüchen: dieselbe Funktion, zwei Bäume, vier Antworten; die Rangfolge von Art und Zoll kippt zwischen den Bäumen.",
    "Boosting": "Experiment 3 für Boosting (Voreinstellung ist der Random Forest).",
    "Netz": "Experiment 3 für das kleine Netz: es extrapoliert glatt, sein interventionaler Beitrag liegt nahe an der interventionalen Wahrheit.",
    "Hintergrund derselben Art": "Der Hintergrund wird auf Zeilen derselben Art eingeschränkt (Experiment 1 und 3): das Modell wird nicht außerhalb der Daten befragt.",
}

# Zitat in Stück 1 (beitrag-linear-demo) aus diesem Stück: siehe tests/test_claims.py (Spannen der Experiment-3-Tabelle)
