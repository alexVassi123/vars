# Onderzoeksprotocol

Dit document legt het experimentele protocol vast voor de bachelorscriptie over multi-view foul herkenning met de SoccerNet-MVFoul dataset. Het protocol is opgesteld vooraf en wordt niet gewijzigd na het zien van de resultaten, om post-hoc aanpassingen (p-hacking) te voorkomen.

## Onderzoeksvraag

Produceert cross-attention tussen cameraperspectieven betere overtreding classificatie dan eenvoudigere aggregatie strategieën?

## Vergeleken methodes

Drie aggregatie strategieën worden vergeleken onder identieke trainingscondities:

1. **Mean pooling** (`--pooling_type mean`). Parametervrije, elementgewijze gemiddelde over views. Dient als zero-parameter baseline.
2. **Weighted attention** (`--pooling_type attention`). De originele VARS baseline. Aangeleerde (512x512) gewichtsmatrix, input-onafhankelijke view weging. Ongeveer 262k parameters.
3. **Cross-attention** (`--pooling_type cross_attention`). Voorgestelde methode. Aangeleerde task-query tokens met multi-head attention waarbij Q van de tokens komt en K/V van de views. Ongeveer 1M parameters.

## Seeds

Elke methode wordt getraind met drie random seeds: `1`, `2`, `3`. Dit geeft in totaal 9 trainingsruns (3 methodes x 3 seeds).

## Hyperparameters (identiek voor alle runs)

| Parameter | Waarde |
|---|---|
| Backbone | R(2+1)D-18, voorgetraind op Kinetics-400 |
| Batch size | 8 |
| Max epochs | 40 |
| Early stopping patience | 12 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| LR schedule | LinearLR warmup (3 epochs) gevolgd door CosineAnnealingLR |
| Loss | Focal loss (gamma=2.0) met class weights, som over beide heads |
| Views tijdens training | 2 per sample |
| Views tijdens evaluatie | 2 per sample (zelfde als training voor consistentie) |
| Frame range | start=20, end=100, fps=10 (32 frames gecentreerd op het contactmoment) |
| Data augmentatie | RandomAffine, RandomPerspective, Rotation, ColorJitter, HorizontalFlip |
| Temporele augmentatie | Random shift van plus/min 10 frames, consistent over views |
| Head dropout | 0.4 in zowel fc_offence als fc_action |

## Metrieken

### Primaire metriek
**Test leaderboard waarde (LB) bij de beste validation epoch.** De LB is het gemiddelde van de twee balanced accuracies:

LB = 0.5 * balanced_accuracy_offence_severity + 0.5 * balanced_accuracy_action

De checkpoint wordt geselecteerd op basis van de beste validation LB (niet test LB), om model selectie leakage te voorkomen.

### Secundaire metrieken
- Balanced accuracy per head (offence severity en action type afzonderlijk).
- Aantal parameters per aggregatie methode.
- Training wall-clock tijd.
- Beste validation epoch per run.

## Statistische toets

**Paarsgewijze Wilcoxon signed-rank toets**, gepaard per seed. Dit is een niet-parametrische paartoets die geen normaalverdeling van de verschillen veronderstelt en geschikt is bij kleine N.

Drie paarsgewijze vergelijkingen worden uitgevoerd:
- mean versus attention
- mean versus cross_attention
- attention versus cross_attention

### Beslissingsregel

Methode A wordt geacht beter te zijn dan methode B als aan beide voorwaarden wordt voldaan:
1. mean(A) > mean(B) over de drie seeds
2. p < 0.05 in de Wilcoxon toets

Anders wordt het resultaat gerapporteerd als "niet significant verschillend".

### Beperking van de statistische power

Met N=3 seeds is de minimaal haalbare p-waarde in de Wilcoxon toets gelijk aan 0.25. Dit betekent dat significantie bij alpha=0.05 mathematisch onbereikbaar is, ongeacht de grootte van het effect. De toets wordt opgenomen voor methodologische volledigheid. Resultaten worden beschreven als descriptief in plaats van confirmatief.

## Uitvoeringsplan

### Fase 0: Omgeving controleren
- GPU beschikbaar met minimaal 20 GB VRAM.
- Dataset volledig uitgepakt in `data/SoccerNet/mvfouls/{Train,Valid,Test,Chall}`.
- Python omgeving met PyTorch, torchvision en SoccerNet package.

### Fase 1: Smoke tests (1 epoch per methode)
Voor elke aggregatie methode wordt een run van 1 epoch gestart om te verifieren dat:
- Er geen Python exceptions of CUDA OOM optreden.
- De training loss daalt over iteraties binnen de epoch.
- Een `best_model.pth.tar` wordt opgeslagen.
- De validation LB na 1 epoch positief is (typisch 20-30).

### Fase 2: Volledige trainingsruns
Voor elke methode worden drie volledige runs uitgevoerd met seeds 1, 2 en 3. Per run wordt gecontroleerd:
- Training log toont dalende loss en stijgende validation LB.
- Beste checkpoint opgeslagen bij een plausibele epoch (typisch 10 tot 25).
- Beste validation LB is geen NaN en blijft niet steken op een laag plateau.

### Fase 3: Evaluatie en analyse
Voor elke van de 9 runs wordt uit de per-epoch prediction JSONs de test LB bij de beste validation epoch gehaald. Per methode worden gemiddelde en standaarddeviatie berekend over de drie seeds. Paarsgewijze Wilcoxon toetsen worden uitgevoerd op zowel de totale LB als op de balanced accuracies van beide heads afzonderlijk.

### Fase 4: Visualisaties en verdieping
- Bar chart met test LB per methode (gemiddelde plus/min std).
- Per-seed scatter plot om consistentie over seeds te tonen.
- Per-head grouped bar chart (offence versus action).
- Training curves met gemiddelde en gearceerde std band per methode.
- Confusion matrices per methode (beste seed) voor beide taken.
- Tabel met parameter count en beste epoch per methode.

## Data gebruik

- Training: alleen het officiele Train split (2319 samples na filtering).
- Model selectie: alleen het Valid split (321 samples).
- Eindrapportage: alleen het Test split (251 samples), en uitsluitend bij de epoch die op basis van Valid werd gekozen.
- Het Chall split wordt niet gebruikt omdat de labels niet publiek beschikbaar zijn.

Het Test split wordt niet gebruikt voor model selectie, hyperparameter tuning of checkpoint keuze. Alleen validation data informeert de training beslissingen.

## Reproduceerbaarheid

- Alle runs gebruiken dezelfde dataset, dezelfde code versie en dezelfde hyperparameters op seed na.
- Seeds worden expliciet gezet voor Python `random`, NumPy, PyTorch CPU en PyTorch CUDA via een `set_seed()` helper in `main.py`.
- Volledige CUDA determinisme wordt niet afgedwongen (dit zou training ongeveer 10 procent vertragen), maar seed variantie wordt expliciet als bron van ruis gerapporteerd.

## Wat NIET wordt gedaan

Om de integriteit van het protocol te bewaken worden de volgende aanpassingen niet uitgevoerd na het zien van de resultaten:
- Toevoegen van extra seeds om significantie te bereiken.
- Aanpassen van hyperparameters voor een specifieke methode om deze beter te laten presteren.
- Veranderen van de primaire metriek of de beslissingsregel.
- Selecteren van een andere validation epoch dan de epoch met de beste validation LB.
- Ensemble van de drie seeds voor rapportage (kan als aparte aanvullende analyse, maar vervangt niet de per-seed resultaten).

## Verwachte uitkomsten

Het protocol staat drie soorten uitkomsten toe als geldig resultaat:

1. **Cross-attention wint significant.** Positieve bevestiging van de hypothese.
2. **Cross-attention presteert vergelijkbaar of slechter.** Negatief resultaat dat evengoed wetenschappelijke waarde heeft, mits goed geinterpreteerd (bijvoorbeeld in termen van capaciteit-versus-data mismatch).
3. **Niet significant verschillend.** Gegeven de lage power bij N=3 is dit de meest waarschijnlijke uitkomst. In dit geval worden descriptieve resultaten (gemiddelde, std, ranking consistentie) gerapporteerd.

Alle drie de uitkomsten zijn legitiem en worden niet als falen beschouwd. De scriptie discussieert de bevindingen op basis van de werkelijke resultaten.
