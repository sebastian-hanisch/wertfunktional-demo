# Dieselbe Frage, andere Wahrheit: vier Wertfunktionale – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-wertfunktional-demo.streamlit.app/)** -->

Stück 2 der Reihe **„Beitrag und Beleg“** (Feld „Modelle verstehen“, Bereich Lernen) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Kante: **`fix` auf Stück 1**
(`beitrag-linear-demo`). Dort ist das Wertfunktional festgehalten (interventional) und der Beitrag eines linearen Modells Koeffizient mal Abweichung; die Annahme, die hier aufgehoben wird, lautet: *es gibt den Beitrag eines Merkmals,
und die Zerlegung des linearen Modells gilt allgemein.* Hier wird das Wertfunktional zum Regler, und dieselbe Funktion bekommt je Frage andere Beiträge. Daneben gibt es einen ungerichteten Verweis auf das Stück über den dünnen Baum
(erklärbar von Natur gegen nachträglich erklärt).

Vehikel ist derselbe **erfundene Containerstrom** wie in Stück 1 (Art, Kunde, Zollhalt, Wochenende, Vorlauf, dazu ein Rauschmerkmal Gewicht ohne Wirkung; Ziel ln Verweilzeit, wahre Mitte μ(x) bekannt). Zollhalt und Vorlauf gibt es nur
bei bestimmten Arten, das erzeugt die Abhängigkeit, an der sich die Wertfunktionale trennen. **Es gibt keine reale Datenquelle**; in den gelesenen Quellen gab es keine Messung, ob Beitragszahlen im Betrieb, in der Logistik oder im Recht
zu besseren Entscheidungen führen. Gezählt werden Instanzen, Zufalls-Seeds und Modellauswertungen; was die Zahlen für einen Betrieb heißen, wird hier nicht behauptet.

## Warum dieses Problem

Ein Beitrag ohne genanntes **Wertfunktional** ist keine Antwort. Das Wertfunktional ist die Vorschrift, nach der Merkmale „weggelassen“ werden: *interventional* (Merkmale auf den Wert der Zeile setzen, die übrigen aus einem Hintergrund),
*konditional* (bedingte Erwartung unter der gemeinsamen Verteilung), *pfadabhängig* (Blattanteile eines Baums, hängt an der Baumform) und *Leave-one-out* (Merkmal weglassen, neu trainieren; keine Zeile, nur eine Wichtigkeit).
**Kernfragen:** Welche Frage stellt das Funktional? Ändert sich der Beitrag derselben Funktion mit der Frage? Welche Wahrheit passt zu welchem Modell? Neu ist an alledem nichts; Quellen und Vorbilder (XAI-Bench, OpenXAI, Hooker und
Mentch) stehen unten. Hier ist es eine nachrechenbare Vorführung mit Zähleinheiten und exakten Orakeln auf einem Vehikel, in dem sogar die konditionale Wahrheit **exakt** vorliegt.

## Konventionen (für die ganze Reihe)

| Begriff | Bedeutung in dieser Demo |
|---|---|
| **Zeile** | ein Container mit Art, Kunde, Zoll, Wochenende, Vorlauf, Gewicht; Ziel ist ln(Verweilzeit in Tagen) |
| **Beitrag** φ_j(x) | lokaler, vorzeichenbehafteter Anteil des Merkmals j an der Vorhersage für *eine* Zeile; **Wichtigkeit** = globales Mittel von \|φ_j\| über die Instanzen |
| **Wertfunktional** | die Frage hinter einem Beitrag: interventional, konditional, pfadabhängig oder Leave-one-out |
| **Hintergrund** | die Bevölkerung des Generators: 1 224 gewichtete Stützpunkte, exakt (oder eine Stichprobe natürlicher Zeilen, oder nur Zeilen derselben Art) |
| **Wahrheit je Funktional** | Shapley-Wert der wahren Mitte μ unter genau diesem Wertfunktional |
| **Instanzen** | Zeilen, für die ein Beitrag gerechnet wird; die **Zeilenmenge** nennt Zahl und Art (exakte Bevölkerung oder natürliche Zeilen) |

**Shapley-Wert** heißt die Aufteilung (Mittel der Grenzbeiträge über alle Reihenfolgen, gerechnet als Summe über Koalitionen mit Gewicht |S|!(d − |S| − 1)!/d!); „SHAP“ hieße die Software mit ihren Näherungen und kommt hier nicht vor.
Das Rauschmerkmal Gewicht ist Nullspieler und fällt weg; die Modelle sind auf den fünf übrigen Merkmalen trainiert. Rangmaß **und** Betragsmaß stehen immer nebeneinander: Rangkorrelation der Wichtigkeit (Mittelränge bei Gleichstand) und
gleiches größtes Merkmal je Instanz; Betragsmaß = Summe über die fünf Merkmale des mittleren |φ_A − φ_B| je Instanz („Summe der Abweichungen je Instanz“).

## Modell

Generator (aus der `verweilzeit-demo` kopiert, nicht importiert; der Kopie-Test hält ihn an eingefrorene Werte der Quelle; eigener Zufallsgenerator SplitMix64) wie in Stück 1; die Formeln stehen im Expander
„📐 Mathematische Formulierung“. **Wahrheit je Funktional:** Shapley-Wert der wahren Mitte unter dem interventionalen Funktional (Hintergrund: die exakte Bevölkerung, Gauß-Legendre über den Vorlauf) und unter dem konditionalen
(exakt unter der Verteilung des Generators); beides ist vorgerechnet und eingefroren (`wfk_truth.json`, `tools/wfk_make_importance.py` mit `--check`).
**Modelle** (Experiment 3, eingefroren aus scikit-learn, `tools/wfk_make_models.py` mit `--check`, nur im Entwicklungs-venv): Boosting (200 Bäume, Tiefe 3, Lernrate 0,1), Random Forest (60 Bäume, mindestens 5 Zeilen je Blatt) und ein kleines
Netz (16 tanh-Einheiten, Art und Kunde als Dummies), je 8 Zufalls-Seeds (Trainingsdatensätze mit n = 1 000, nur die fünf Merkmale mit Wirkung). **Ersatzbaum** (Experiment 2): ein Baum auf der wahren Mitte, 100 000 natürliche Zeilen,
mindestens 100 Zeilen je Blatt. Das pfadabhängige Funktional (Blattanteile, Lundberg u. a. 2020, Algorithmus 1) ist eigener Code (`wfk_tree.py`), geprüft gegen Knoten-Rekursion; `shap` und `lime` sind nicht im Spiel.

## Methodik

* **Zähleinheiten.** Instanzen (Zeilen, für die ein Beitrag gerechnet wird), Zufalls-Seeds (Trainingsdatensätze), Ziehungen (Stichproben aus der Bevölkerung), Koalitionen (32 je Zeile bei fünf Spielern), Modellauswertungen.
  Jede Aussage über Beiträge je Zeile nennt die Zeilenmenge und gibt eine Spanne über **fünf getrennte Instanzmengen** an, keine Punktzahl aus einer festen Ziehung; die Testwerte (Wichtigkeit der Wahrheit) kommen aus der **exakten Bevölkerung**.
* **Instanzmengen.** Experiment 1: Menge 1 (300 Zeilen, Zufalls-Seed 4242); Experiment 2: fünf Mengen zu 150 Zeilen (Zufalls-Seeds 7777 bis 7781); Experiment 3: fünf Mengen zu 40 Zeilen (4242, 5301 bis 5304).
* **Gleichstände der Bäume.** Die Bäume sind aus scikit-learn eingefroren; es löst Gleichstände beim Aufteilen nach seiner (mit `random_state` festen) Zufallsreihenfolge der Merkmale. Die Konvention der Reihe (kleinster Index in fester Aufzählungsreihenfolge, eine zweite Regel daneben) ist hier **nicht** umgesetzt und nicht gerechnet (🚧 unten).
* **Zwei Modellklassen mindestens** in jeder Aussage zu Modellen: Boosting, Random Forest **und** Netz; das lineare Modell steht in Stück 1.

## Befunde (gemessen, keine Behauptungen)

**Experiment 1: eine Funktion, mehrere Antworten.** Zeilenmenge: exakte Bevölkerung (1 224 gewichtete Stützpunkte). Mittel von |φ| über die Bevölkerung: Art 0,651 interventional gegen 0,168 konditional, Vorlauf 0,561 gegen 0,275;
Rang 1 wechselt von Art zu Vorlauf. Rangmaß: Rangkorrelation der beiden Wichtigkeiten 0,70; Betragsmaß: Summe der Abweichungen je Zeile 1,235. Kunde und Wochenende (unabhängige Merkmale) haben in beiden Funktionalen denselben Wert.
Leave-one-out (exakt, Verlustanstieg des besten Vorhersagers ohne Merkmal): Vorlauf 0,140, Kunde 0,066, Art 0,037, Zoll 0,032, Wochenende 0,008.
**Bei kleinen Instanzmengen schwankt der Rang 1 der Wichtigkeit; die Größe des Hintergrunds ist hier nicht variiert.** Je 20 000 Ziehungen von m Instanzen aus der exakten Bevölkerung (mit Zurücklegen; die Zeilenmenge ist die Bevölkerung, nicht ein Pool natürlicher Zeilen):

| Instanzen m | interventional: Art vor Vorlauf in … der Ziehungen (Art, 5 bis 95 %) | konditional: Vorlauf vor Art in … der Ziehungen |
|---|---|---|
| 20 | 84,5 % (0,558 bis 0,750) | 93,5 % |
| 40 | 92,4 % (0,585 bis 0,720) | 98,3 % |
| 100 | 98,8 % (0,608 bis 0,694) | 100,0 % |
| 150 | 99,7 % (0,616 bis 0,687) | 100,0 % |

**Experiment 2: zwei Funktionen, gleiche Daten** (Ersatzbaum auf der wahren Mitte mit 462 Blättern, R² 0,9965 gegen die Mitte auf 5 000 frischen Zeilen; fünf getrennte Mengen zu 150 Instanzen). Interventional liegen μ und der
Ersatzbaum um 1,31 bis 1,39 auseinander (Art allein 0,64 bis 0,69). Pfadabhängig gegen interventional **auf demselben Baum**: 0,061 bis 0,066. Zwei Funktionen, die auf den Daten fast übereinstimmen, haben also ebenso verschiedene
Beiträge wie interventionales und konditionales Funktional derselben Funktion (1,31 bis 1,39 gegen 1,235); das pfadabhängige Funktional liegt auf diesem Ersatzbaum dicht am interventionalen, auf anderen Bäumen kann es weit abweichen (Mini-Tabelle). Die Erklärung „außerhalb der Daten“ ist eine Deutung, nicht getrennt geprüft.

**Experiment 3: welche Wahrheit passt zu welchem Modell?** 8 Zufalls-Seeds (n = 1 000), Hintergrund Bevölkerung, fünf getrennte Instanzmengen zu je 40 Zeilen; Summe der Abweichungen je Instanz, Mittel über die Zufalls-Seeds, **Spanne über die
fünf Mengen**:

| Modell (R² gegen die Mitte, Mittel über 8 Zufalls-Seeds) | im gleichen Funktional gegen die Wahrheit: interventional / konditional | interventional gegen die **konditionale** Wahrheit | dasselbe Modell: interventional gegen konditional / gegen pfadabhängig / konditional gegen pfadabhängig | gleiches größtes Merkmal (interventional gegen konditional) | Rangkorrelation der Wichtigkeit (interventional gegen konditional) |
|---|---|---|---|---|---|
| Boosting (0,930) | 1,03 bis 1,18 / 0,15 bis 0,17 | 0,31 bis 0,36 | 0,24 bis 0,28 / 0,061 bis 0,075 / 0,247 bis 0,278 | 76 bis 92 % | 0,92 bis 0,97 |
| Random Forest (0,911) | 1,15 bis 1,34 / 0,18 bis 0,21 | 0,41 bis 0,43 | 0,30 bis 0,33 / 0,083 bis 0,086 / 0,293 bis 0,334 | 66 bis 81 % | 0,80 bis 0,95 |
| Kleines Netz (0,922) | 0,44 bis 0,50 / 0,15 bis 0,18 | 1,53 bis 1,65 | 1,48 bis 1,58 / nicht anwendbar | 32 bis 54 % | 0,70 bis 0,92 |

Der interventionale Beitrag der Bäume liegt weit von der interventionalen Wahrheit und nah an der konditionalen (Boosting und Wald: 1,03 bis 1,34 gegen 0,31 bis 0,43), beim Netz ist es umgekehrt (0,44 bis 0,50 gegen 1,53 bis 1,65): **welche
Wahrheit dem Beitrag näher liegt, unterscheidet sich bei diesen drei Modellfassungen (Boosting und Wald näher an der konditionalen, das Netz an der interventionalen); ob es an der Klasse oder an den Einstellungen liegt, ist nicht getrennt.** Der konditionale Beitrag liegt bei allen drei Klassen nah an der konditionalen Wahrheit (0,15 bis 0,21). **Rang 1** des interventionalen
Modellbeitrags wie der der interventionalen Wahrheit, in Zufalls-Seeds von 8 je Menge (Mengen 1 bis 5): Boosting 0, 0, 8, 8, 2; Random Forest 0, 1, 8, 8, 1; Netz 5, 8, 8, 8, 8. In den Mengen 3 und 4 ist der Rang 1 der Wahrheit der Vorlauf, in den Mengen 1, 2 und 5 die Art: der Rang 1 der Wahrheit wechselt mit der Instanzmenge und mit ihm die Trefferzahl; der Betrag liegt beim Boosting in 1,03 bis 1,18, die Rangkorrelation bei Wald und Netz in 0,80 bis 0,95 und 0,70 bis 0,92 (Spannen über die Mengen). **Leave-one-out** (Mittel über 8 Zufalls-Seeds, Verlustanstieg gegen die wahre Mitte auf 5 000 frischen Zeilen) ordnet
Vorlauf vor Kunde und Wochenende zuletzt bei Boosting und Wald (Vorlauf 0,149, Kunde 0,055 und 0,046); beim Netz liegt die Art an zweiter Stelle der Rangfolge (0,061) und das Wochenende bei −0,001 (das Netz ohne Wochenende ist im Mittel nicht schlechter).
**Hintergrund derselben Art** (nur Zeilen der Art der Instanz; die Wahrheit im selben Hintergrund): der Abstand sinkt bei allen drei Klassen stark, Boosting 0,136 bis 0,157, Wald 0,158 bis 0,191, Netz 0,138 bis 0,169 (gegen die Spannen der Tabelle). Das vergleicht
zwei verschiedene Fragen und sagt nicht, welcher Hintergrund richtig ist: mit Hintergrund derselben Art wird das Modell nicht außerhalb der Daten befragt (Deutung, nicht getrennt geprüft).

**Experiment 4: Zwillinge, das Funktional entscheidet** (geschlossene Formeln, Orakel: Quadratur und Stichprobe). Gauß-Fall f = x_A: bei ρ = 0,9 bekommt der Zwilling konditional 41,8 % der Wichtigkeit, bei ρ = 0,99 49,0 %, interventional 0 %. Binärer Fall
(Janzing u. a., Beispiel 1) mit q = 9/10: konditional (E|φ_1|, E|φ_2|) = (17/50, 1/5), Anteil des Zwillings 10/27, interventional (1/2, 0); an (1, 1) (3/10, 1/5), an (1, 0) (7/10, −1/5). Bei ρ = 0 und bei q = 1/2 ist der Zwilling kein Zwilling:
alle Funktionale geben dasselbe.

**Mini-Tabelle (Handbeispiel in Brüchen)**: Zoll nur bei Import, μ = 1 + 2 Art + 2 Art Zoll, Verteilung (1, 0) 1/4, (1, 1) 1/4, (0, 0) 1/2; für (Import, Zollhalt) Grundwert 5/2, Vorhersage 5. Beitrag (Art, Zoll): interventional (3/2, 1), konditional (3/4, 7/4),
pfadabhängig auf Baum 1 (7/4, 3/4), auf Baum 2 (1/2, 2); beide Bäume geben auf den drei Stützpunkten dasselbe wie μ, außerhalb der Daten (Export mit Zollhalt) 1 und 5 gegen 1. Wichtigkeit (Art, Zoll): interventional (5/4, 1/2), konditional (13/12, 7/8),
pfadabhängig Baum 1 (11/8, 1/2), Baum 2 (17/24, 9/8): **die Rangfolge von Art und Zoll kippt zwischen zwei Bäumen mit gleicher Vorhersage auf den Daten.** Leave-one-out (Verlustanstieg): Art 2/3, Zoll 1/2.

### Zahl → Test (Band oder exakt)

| Zahl in dieser README | Test |
|---|---|
| Wichtigkeit der Bevölkerung, Ziehungen, Experiment 2, Tabelle des Experiments 3, Hintergrund derselben Art, Leave-one-out, Zwillinge, Mini-Tabelle | `tests/test_claims.py::test_zahl_der_readme_stimmt_mit_der_rechnung_ueberein`: der Wortlaut steht in der README, und die Rechnung (dieselben Funktionen wie die App) erzeugt genau diese Stellen |
| Wahrheit: Koalitionsformel gegen alle 120 Permutationen, Effizienz, Nullspieler, Bevölkerung gegen andere Quadratur und Stichprobe, bedingte Erwartung von Hand | `tests/test_oracle_wahrheit.py` |
| Pfadabhängig: Blattrechnung gegen Knoten-Rekursion, v(alle) = Vorhersage, Blattanteile gegen scikit-learn, Mini-Bäume gegen Brüche | `tests/test_tree.py`, `tests/test_hand.py` |
| Zwillinge: geschlossene Formeln gegen Quadratur, Stichprobe, Koalitionsformel und Brüche | `tests/test_twin.py` |
| Eingefrorene Modelle: Effizienz-Konsistenz, Neu-Trainieren (Boosting, Wald) als Band, Form | `tests/test_evaluation.py` |
| Rechenzeiten, Laufzeit der Suite, Erwartungen des Plans (unten) | Messung dieser README, nicht im Test |

## Befunde und Korrekturen gegenüber dem Plan

Die Zahlen des Plans stammen aus einer Vormessung mit anderem Zufallsgenerator und eigenen Fassungen der Modelle; **alle Zahlen sind neu gemessen**. Was sich verschoben hat (Erwartungen des Plans, Messung dieser README, nicht im Test):

* **Netz.** Der Plan erwartete für das Netz (R² 0,943) 0,26 bis 0,34 gegen die interventionale und 1,13 bis 1,17 gegen die konditionale Wahrheit; hier (R² 0,922) 0,44 bis 0,50 und 1,53 bis 1,65. Die Aussage bleibt (das Netz liegt näher an der interventionalen Wahrheit als die Bäume
  und weit von der konditionalen), die Zahlen nicht. Stück 1 (`beitrag-linear-demo`) zitiert bereits die Zahlen dieser Tabelle.
* **Boosting und Wald.** Boosting 1,03 bis 1,18 und 0,31 bis 0,36 (Plan 1,05 bis 1,17 und 0,31 bis 0,34); Wald 1,15 bis 1,34 und 0,41 bis 0,43 (Plan 1,01 bis 1,14 und 0,41 bis 0,44); R² 0,930 und 0,911 (Plan 0,925 und 0,895).
* **Rang 1.** Im Plan hatte die Menge mit dem Zufalls-Seed 4242 den Vorlauf vorn; hier sind es die Mengen 3 und 4 (Zufalls-Seeds 5302 und 5303). Der Plan-Satz „der Rang 1 ist eine Eigenschaft der Instanzmenge“ gilt für den Rang 1 der Wahrheit unverändert; ob das Modell ihn trifft, hängt zusätzlich am Modell.
* **Ersatzbaum.** 462 Blätter bei R² 0,9965 (Plan 473 Blätter bei 0,9965); μ gegen Baum interventional 1,31 bis 1,39 (Plan 1,30 bis 1,42), Art allein 0,64 bis 0,69 (0,64 bis 0,70); pfadabhängig gegen interventional auf demselben Baum 0,061 bis 0,066 (0,059 bis 0,067).
* **Instanzzahl.** Der Plan zog Instanzmengen aus einem Pool von 12 000 natürlichen Zeilen; hier werden die Ziehungen aus der exakten Bevölkerung mit Zurücklegen gemacht (die Zeilen sind dann Stützpunkte, keine stetigen Vorlauf-Werte). Gemessen 84,5 / 92,4 / 98,8 / 99,7 % bei 20 / 40 / 100 / 150
  Instanzen (Plan: 85,8 / 93,3 / 99,2 / 99,9 %).

## Ehrliche Grenzen

* Alles gilt für **dieses Vehikel** (erfunden, fünf Merkmale mit Wirkung), diese Modellklassen, Zufalls-Seeds und Instanzmengen. **Nutzen von Beitragszahlen in Betrieb, Logistik oder Recht ist nicht belegt** und nicht Gegenstand.
* Die App prüft **funktionsbezogen** (ohne Menschen, Doshi-Velez und Kim 2017) und zeigt **nicht**, ob Menschen etwas verstehen.
* Die konditionale Wahrheit ist nur exakt, weil der Generator die Verteilung kennt; bei echten Daten wäre sie zu schätzen.
* Die Rechenzeiten sind unter Last gemessen; die Zahlen sind unter Windows mit Python 3.14 gerechnet, Linux-Lauf (Docker, Python 3.12) am 2026-11-20 grün; die Modelle hängen von der Version von scikit-learn ab (die Tests prüfen Boosting und Wald als Band).

### 🚧 Wo die Annahmen enden

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein Vehikel** (erfundener Containerstrom, ln Verweilzeit, fünf Merkmale mit Wirkung) | Alle Zahlen gelten für dieses Vehikel, diese Modellklassen, Zufalls-Seeds und Instanzmengen. Das Rauschmerkmal Gewicht ist Nullspieler und fällt weg. | - |
| **Konditionale Wahrheit exakt** | Sie ist nur exakt, weil der Generator die gemeinsame Verteilung kennt; bei echten Daten müsste sie geschätzt werden. | - |
| **Eigene Modellfassungen eingefroren** (scikit-learn, nur beim Vorrechnen) | Die App rechnet kein Boosting, keinen Wald und kein Netz; ihre Zahlen stammen aus der Vorrechnung mit festen Hyperparametern. | - |
| **Ersatzbaum als Funktion** | Ein Baum mit rund 470 Blättern ist als Modell nicht lesbar; er trennt nur Funktion und Funktional. | Stück über den dünnen Baum |
| **Hintergrund = Bevölkerung** | Interventional wertet das Funktional außerhalb der Daten aus; ein Hintergrund derselben Art vermeidet das, beantwortet aber eine andere Frage. | Folgestück zu unmöglichen Kombinationen |
| **Zwillinge geschlossen** (Gauß und binär) | Der Fall mit einem Wald und fast gleichen Spalten ist hier nicht gemessen. | Verweis `korrelierte-merkmale-demo` |
| **Funktionsbezogene Prüfung** (ohne Menschen; Doshi-Velez und Kim 2017) | Ob Menschen mit Beitragszahlen besser entscheiden, ist nicht Gegenstand; ein Nutzen in Betrieb, Logistik oder Recht ist nicht belegt. | Lehrseite zu Menschenstudien |

**Nicht gemessen oder nicht gebaut:** die Größe des Hintergrunds als eigener Befund (die Option steht in Experiment 1, der Rang bei 40 bis 1 000 Hintergrundzeilen über fünf Mengen zu 100 Instanzen ist nicht neu gemessen); die Tabelle der Rangfolgen aller Funktionale
auf dem Ersatzbaum (nur Abstände sind gerechnet); die Gleichstandsregel der Reihe für die Bäume (kleinster Index) mit einer zweiten Regel daneben; die Baumform-Aufzählung und KernelSHAP (Folgestück zu den Näherungen); Integrated Gradients; Menschenstudien. Das Häkchen „Funktionale“ des Plans
für Experiment 1 ist nicht gebaut (die Abbildung zeigt immer interventional, konditional und pfadabhängig).

## Tests

Die Suite hat 192 Tests (Laufzeit rund 100 Sekunden auf diesem Rechner, mit `-n 4`, unter paralleler Last durch andere Läufe; ein Wert dieser README, nicht im Test). Zusammensetzung:

* **Handbeispiele** (`test_hand.py`, `test_twin.py`): Brüche des Mini-Vehikels (vier Antworten, zwei Bäume, Leave-one-out durch Neuberechnung, sechs Permutationen mit Münzspieler), binärer und Gauß-Zwilling.
* **Orakel** (`test_oracle_wahrheit.py`, `test_tree.py`, `test_twin.py`): Koalitionsformel gegen alle 120 Permutationen; Effizienz und Nullspieler; Bevölkerung gegen Gauß-Legendre mit 48 Punkten, Mittelpunktsregel mit 400 und eine Stichprobe;
  bedingte Erwartung von Hand und gegen Funktionen unabhängiger Merkmale; Blattrechnung gegen Knoten-Rekursion, Blattanteile gegen scikit-learn, Ensemble gegen die Modellvorhersage; Zwilling gegen `dblquad` und 400 000 Paare.
* **Eigenschaften und Randfälle** (`test_edge_cases.py`, `test_eckfaelle.py`): Baum mit einem Blatt, ρ und q an den Rändern, jeder Regler am Minimum und Maximum (einzeln und zugleich).
* **Eingefrorenes** (`test_evaluation.py`): Effizienz-Konsistenz der Modellbeiträge, Neu-Trainieren (Boosting, Wald) und Neu-Rechnen der Wahrheit als Band, Vergleichswerkzeuge der Tools.
* **Claims, Verdikte (je Preset), Presets, Generator, Layout, App (AppTest), Footer, Laufzeit-Importe.**

Messungen dieser README, nicht im Test: Vorrechnen der Modelle (`tools/wfk_make_models.py`, 13 Minuten bei geteilter Maschine), Wahrheit (`tools/wfk_make_importance.py`, 13 Sekunden); ein Seitenlauf der App im AppTest etwa 6 Sekunden beim ersten Aufruf (mit Streamlit-Start), danach je Preset 0,2 bis 1,2 Sekunden.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore`, `.streamlit/config.toml` | venv, Caches, Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Kopf, Kernfragen, vier Experimente, Mini-Tabelle, Grenzen, Formeln) |
| `wfk_constants.py` | Vehikel, Zeilenmengen, Modelle, Regler, Farben, Presets |
| `wfk_rng.py`, `wfk_generator.py` | SplitMix64 (Vorlage unverändert), erfundener Containerstrom und wahre Mitte |
| `wfk_pop.py` | Bevölkerung, interventionale und konditionale Wertfunktion, Shapley-Wert (Koalitionen, Permutationen) |
| `wfk_tree.py` | Bäume und pfadabhängiges Funktional (Blattanteile), Ensemble |
| `wfk_twin.py`, `wfk_hand.py` | Zwillinge (geschlossen), Mini-Vehikel in Brüchen |
| `wfk_truth.py`, `wfk_truth.json`, `wfk_models.json`, `wfk_pfad.json` | eingefrorene Wahrheit, Modelle und Ersatzbaum, pfadabhängiger Beitrag des Ersatzbaums |
| `wfk_evaluation.py` | Messungen, Maße, Verdikte als reine Funktionen |
| `wfk_visualization.py`, `wfk_presets.py`, `wfk_layout.py`, `wfk_texts.py` | Abbildungen, Permalink und Presets, fester Kopf, feste Texte |
| `tools/wfk_make_importance.py`, `tools/wfk_make_models.py`, `tools/wfk_make_pfad.py` | Vorrechnen (je mit `--check`) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, plotly), Entwicklung (pytest, scikit-learn, scipy) |
| `tests/` | siehe Abschnitt Tests |

## Bewusst nicht umgesetzt

* Baumform-Aufzählung, Baum-SHAP und KernelSHAP (Folgestück zu den Näherungen); Integrated Gradients; ein eigenes Training in der App.
* PDF-Export und eine Seite „in einfachen Worten“ (kommen mit der Einbindung in die Website).
* Klassifikation; eine reale Datenquelle; jede Aussage über Menschen oder Betrieb.

## Quellen

Stufe: **W** = gelesen (wo angegeben, die Fassung), **A** = nur Zusammenfassung, „nicht gelesen“ = nur dem Titel nach bekannt.

* Lundberg und Lee 2017, *A Unified Approach to Interpreting Model Predictions*: SHAP als Shapley-Wert der bedingten Erwartung, **W**.
* Lundberg u. a. 2020, *From local explanations to global understanding with explainable AI for trees*: Algorithmus 1 (Blattanteile), **W** (arXiv v1).
* Sundararajan und Najmi 2020, *The many Shapley values for model explanation*: Abschnitte 1 bis 3, **W**.
* Janzing u. a., *Feature relevance quantification in explainable AI: a causal problem*, arXiv v2 (2019), Beispiel 1: **W** (Fassung AISTATS 2020 nicht gelesen); Kumar u. a. 2020, *Problems with Shapley-value-based explanations as feature importance measures*: **W**.
* Doshi-Velez und Kim 2017: Prüfen ohne Menschen heißt „funktionsbezogen“, **W**.
* Nur als Einordnung, **A**: Merrick und Taly 2020, Chen u. a. 2020, Aas u. a. 2019; Vorbilder für Messungen gegen eine bekannte Wahrheit (keine Neuheitsaussage): XAI-Bench, OpenXAI, Hooker und Mentch, **A**. Shapley 1953: nicht lesbar.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Getestet mit Python 3.14 lokal; die CI läuft mit Python 3.12 (die Python-Version der Streamlit Cloud ist in den App-Einstellungen zu wählen).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
