"""Feste Texte der App; die README enthält dieselben Fassungen (tests/test_claims.py vergleicht sie)."""

INTRO_MARKDOWN = """
Ein **Beitrag** sagt, wie viel ein Merkmal zu einer Vorhersage beiträgt. Welche Zahl das ist, hängt an einer **Frage**: dem **Wertfunktional**, der Vorschrift, nach der Merkmale „weggelassen“ werden.
Diese Demo hält die Funktion fest, nämlich die bekannte wahre Mitte eines **erfundenen Containerstroms**, und wechselt die Frage: **interventional**, **konditional**, **pfadabhängig** (Blattanteile eines Baums) und
**Leave-one-out** (neu trainieren ohne das Merkmal). Dieselbe Funktion bekommt dann nicht um Rundungen andere Beiträge. Das ist das zweite Stück der Reihe „Beitrag und Beleg“ und hebt die Annahme des ersten auf,
es gebe *den* Beitrag eines Merkmals.

Weiter unten: **🔬 vier Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Alles hier ist erfunden (keine reale Datenquelle); die Demo zeigt **nicht**, ob Menschen mit
Beitragszahlen besser entscheiden.
"""

SO_FUNKTIONIERT_MARKDOWN = """
1. **Spiel.** Spieler sind die Merkmale; das Spiel einer Zeile x ordnet jeder Koalition S von Merkmalen einen Wert v(S) zu. Der **Shapley-Wert** φ_j ist das Mittel der Grenzbeiträge v(S ∪ {j}) − v(S) über alle Reihenfolgen.
2. **Wertfunktional** = die Vorschrift für v(S): *interventional* (die Merkmale in S auf den Wert der Zeile setzen, die übrigen aus einem Hintergrund, Mittel über den Hintergrund), *konditional* (bedingte Erwartung gegeben x_S unter der
   gemeinsamen Verteilung), *pfadabhängig* (Blattanteile eines Baums; hängt an der Baumform) und *Leave-one-out* (Merkmal weglassen, neu trainieren; hat keinen Beitrag je Zeile, nur eine Wichtigkeit).
3. **Wahrheit je Funktional** heißt der Shapley-Wert der wahren Mitte μ unter genau diesem Wertfunktional. Hier sind sogar die konditionalen Werte **exakt**, weil der Generator seine Verteilung kennt.
4. **Maße.** Immer Rangmaß *und* Betragsmaß: Rangkorrelation der Wichtigkeit und gleiches größtes Merkmal je Instanz, daneben die Summe der Abweichungen je Instanz. Ein Maß allein täuscht.
"""

ASSUMPTIONS_MARKDOWN = """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein Vehikel** (erfundener Containerstrom, ln Verweilzeit, fünf Merkmale mit Wirkung) | Alle Zahlen gelten für dieses Vehikel, diese Modellklassen, Zufalls-Seeds und Instanzmengen. Das Rauschmerkmal Gewicht ist Nullspieler und fällt weg. | - |
| **Konditionale Wahrheit exakt** | Sie ist nur exakt, weil der Generator die gemeinsame Verteilung kennt; bei echten Daten müsste sie geschätzt werden. | - |
| **Eigene Modellfassungen eingefroren** (scikit-learn, nur beim Vorrechnen) | Die App rechnet kein Boosting, keinen Wald und kein Netz; ihre Zahlen stammen aus der Vorrechnung mit festen Hyperparametern. | - |
| **Ersatzbaum als Funktion** | Ein Baum mit rund 470 Blättern ist als Modell nicht lesbar; er trennt nur Funktion und Funktional. | Stück über den dünnen Baum |
| **Hintergrund = Bevölkerung** | Interventional wertet das Funktional außerhalb der Daten aus; ein Hintergrund derselben Art vermeidet das, beantwortet aber eine andere Frage. | Folgestück zu unmöglichen Kombinationen |
| **Zwillinge geschlossen** (Gauß und binär) | Der Fall mit einem Wald und fast gleichen Spalten ist hier nicht gemessen. | Verweis `korrelierte-merkmale-demo` |
| **Funktionsbezogene Prüfung** (ohne Menschen; Doshi-Velez und Kim 2017) | Ob Menschen mit Beitragszahlen besser entscheiden, ist nicht Gegenstand; ein Nutzen in Betrieb, Logistik oder Recht ist nicht belegt. | Lehrseite zu Menschenstudien |
"""

KONVENTION_MARKDOWN = """
| Begriff | Bedeutung in dieser Demo |
|---|---|
| **Zeile** | ein Container mit Art, Kunde, Zoll, Wochenende, Vorlauf, Gewicht; Ziel ist ln(Verweilzeit in Tagen) |
| **Beitrag** φ_j(x) | lokaler, vorzeichenbehafteter Anteil des Merkmals j an der Vorhersage für *eine* Zeile; **Wichtigkeit** = globales Mittel von \\|φ_j\\| über die Instanzen |
| **Wertfunktional** | die Frage hinter einem Beitrag: interventional, konditional, pfadabhängig oder Leave-one-out |
| **Hintergrund** | die Bevölkerung des Generators: 1 224 gewichtete Stützpunkte, exakt (oder eine Stichprobe natürlicher Zeilen, oder nur Zeilen derselben Art) |
| **Wahrheit je Funktional** | Shapley-Wert der wahren Mitte μ unter genau diesem Wertfunktional |
| **Instanzen** | Zeilen, für die ein Beitrag gerechnet wird; die **Zeilenmenge** nennt Zahl und Art (exakte Bevölkerung oder natürliche Zeilen) |
"""

FORMEL_LESEHILFE = """
**Lesehilfe.** *Gegeben:* eine Funktion f (wahre Mitte, Baum, Boosting, Wald oder Netz) und eine Zeile x mit fünf Merkmalen; *gesucht:* der Beitrag φ_j(x) je Merkmal unter einem Wertfunktional.

| Symbol | Bedeutung |
|---|---|
| $x$ | eine Zeile (Art, Kunde, Zoll, Wochenende, Vorlauf) |
| $f$ | die Funktion, die erklärt wird |
| $S$ | Koalition: Menge der Merkmale, die der Zeile „gehören“ |
| $v(S)$ | Wert der Koalition (Wertfunktional) |
| $\\varphi_j(x)$ | Shapley-Wert (Beitrag) des Merkmals $j$ |
| $b$ | Zeile des Hintergrunds (Bevölkerung) |
| $d$ | Zahl der Spieler (hier 5) |
| $\\rho$ | Korrelation des Zwillings im Gauß-Fall |
| $q$ | Wahrscheinlichkeit $x_2 = x_1$ im binären Fall |
"""

FORMEL_MARKDOWN = r"""
**Shapley-Wert** (Mittel der Grenzbeiträge über alle Reihenfolgen, als Summe über Koalitionen):
$$\varphi_j(x) = \sum_{S \subseteq \{1,\dots,d\} \setminus \{j\}} \frac{|S|!\,(d - |S| - 1)!}{d!}\,\big(v(S \cup \{j\}) - v(S)\big), \qquad \sum_j \varphi_j(x) = v(\{1,\dots,d\}) - v(\emptyset) .$$

**Vier Wertfunktionale:**
$$v^{\text{int}}(S) = \mathbb{E}_b\, f(x_S, b_{\neg S}), \qquad v^{\text{kond}}(S) = \mathbb{E}\,[\,f(X) \mid X_S = x_S\,] .$$
Pfadabhängig (Baum $T$): $v^{\text{pfad}}(S)$ ist die Erwartung des Baumwerts, wenn an jedem Knoten mit Merkmal in $S$ dem Wert von $x$ gefolgt wird und an jedem anderen Knoten beide Äste nach ihren Trainingsanteilen gemischt werden
(Lundberg u. a. 2020, Algorithmus 1); es hängt an der **Baumform**. Leave-one-out: Verlustanstieg $L(f_{-j}) - L(f)$ des ohne Merkmal $j$ neu trainierten Modells; es hat keinen Beitrag je Zeile.

**Zwillinge, Gauß-Fall** ($f(x) = x_A$, $(x_A, x_B)$ standardnormal mit Korrelation $\rho$): konditional $\varphi_A = x_A - \rho x_B / 2$ und $\varphi_B = \rho x_B / 2$, interventional $\varphi_A = x_A$ und $\varphi_B = 0$. Anteil des Zwillings an der Wichtigkeit:
$$\frac{\mathbb{E}\lvert\varphi_B\rvert}{\mathbb{E}\lvert\varphi_A\rvert + \mathbb{E}\lvert\varphi_B\rvert} = \frac{\rho / 2}{\rho / 2 + \sqrt{1 - 3\rho^2 / 4}} .$$
**Binär** ($x_1$ gleichverteilt auf $\{0,1\}$, $x_2 = x_1$ mit Wahrscheinlichkeit $q \ge 1/2$, $f = x_1$): $\mathbb{E}\lvert\varphi_1\rvert = q\,(3/4 - q/2) + (1 - q)\,(1/4 + q/2)$ und $\mathbb{E}\lvert\varphi_2\rvert = (q - 1/2)/2$ konditional; interventional $(1/2, 0)$.

**Maße.** Betragsmaß: $\sum_j \operatorname{mean}_x \lvert\varphi_j^{A}(x) - \varphi_j^{B}(x)\rvert$ über die fünf Merkmale; Rangmaß: Rangkorrelation (Mittelränge) der Wichtigkeiten $\operatorname{mean}_x \lvert\varphi_j\rvert$.

Der Formelblock beschreibt die Beitragsrechnung vollständig; die wahre Mitte $\mu$ und die Verteilung der Merkmale des erfundenen Generators stehen in der README des Vorgängerstücks `beitrag-linear-demo` und im Code (`wfk_generator.py`),
die Hyperparameter der Modelle in `wfk_constants.py`.

**Quellen** (Stufe W = gelesen): Lundberg und Lee 2017 (SHAP als Shapley-Wert der bedingten Erwartung, W); Lundberg u. a. 2020 (Algorithmus 1, W, arXiv v1); Janzing u. a. (arXiv v2, 2019, Beispiel 1, W; Fassung AISTATS 2020 nicht gelesen);
Kumar u. a. 2020 (W); Sundararajan und Najmi 2020 (W, Abschnitte 1 bis 3); Doshi-Velez und Kim 2017 (W). Nur als Einordnung (Zusammenfassung, A): Merrick und Taly 2020, Chen u. a. 2020, Aas u. a. 2019. Vorbilder für Messungen gegen
eine bekannte Wahrheit: XAI-Bench, OpenXAI, Hooker und Mentch; diese Demo ist keine neue Methode.
"""
