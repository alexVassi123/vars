# VARS - Thesis

## Introductie

Dit project bouwt voort op het originele VARS (Video Assistant Referee System) onderzoek van Held et al. (2023). In hun paper introduceren zij de SoccerNet-MVFoul dataset en een model dat overtredingen in het voetbal probeert te classificeren. Per actie voorspelt het model wat voor type overtreding het is (tackle, duw, hoge been, enzovoort) en hoe zwaar de overtreding is (geen overtreding, geen kaart, gele kaart of rode kaart). Wat hun aanpak onderscheidt is dat er per actie meerdere camerahoeken beschikbaar zijn, omdat een overtreding er vanuit verschillende perspectieven anders uit kan zien. Het originele model combineert de features van die camerahoeken via een aggregatiemethode (mean, max of een simpele attention).

De repository van het originele werk staat hier: https://github.com/SoccerNet/sn-mvfoul. Ik heb hun code en dataset als startpunt gebruikt voor deze thesis. De data heb ik via het officiele SoccerNet pip-package gedownload met [download.py](download.py). Vervolgens heb ik een aantal aanpassingen in het model en de training gemaakt om een specifieke onderzoeksvraag te kunnen beantwoorden: presteert cross-attention tussen camerahoeken beter dan de eenvoudigere aggregatiemethoden uit het originele werk?

## Belangrijke aanpassingen

### Nieuwe aggregatiemethoden
De originele code had drie aggregatie-opties (mean, max, attention). Ik heb in [VARS model/mvaggregate.py](VARS model/mvaggregate.py) een nieuwe methode toegevoegd: `CrossAttentionAggregate`, de hoofdmethode van mijn thesis. Deze gebruikt leerbare task-query tokens die via multi-head attention bepalen welke camerahoek belangrijk is voor welke taak. De originele attention-methode gebruikt een vaste weging die niet afhangt van de input. Cross-attention kan per sample en per taak kiezen welke view belangrijk is. Het idee is dat dit beter werkt als de relevantie van een camerahoek per overtreding verschilt.

### Diepere classificatie-heads
De originele code had een lineaire laag per taak. Ik heb deze vervangen door een kleine MLP met een tussenlaag, ReLU en dropout (0.4). Met een enkele lineaire laag begon het model snel te overfitten op de trainingsdata, vooral omdat sommige klassen zeldzaam zijn. De klasse "rode kaart" komt bijvoorbeeld maar 27 keer voor in de training set, tegenover meer dan 1000 keer voor "standing tackling". De dropout helpt om die minderheidsklassen niet te memoriseren.

### Focal loss in plaats van gewogen cross-entropy
Door de sterke klasse-imbalans was de originele gewogen cross-entropy loss niet ideaal. In eerdere testruns zag ik dat het model de minderheidsklassen memoriseerde op de training set maar niet generaliseerde. Focal loss met gamma=2.0 legt automatisch meer gewicht op moeilijke voorbeelden, zonder dat je expliciet moet oversamplen.

### Learning rate schedule
De originele `StepLR` is vervangen door 3 epochs lineaire warmup gevolgd door cosine annealing. Dit geeft stabielere training in de eerste epochs en voorkomt grote loss-sprongen bij het begin.

### Early stopping en reproduceerbaarheid
Ik heb early stopping toegevoegd op basis van de validation leaderboard score (patience 12). Verder is er een `set_seed()` functie die Python, NumPy en PyTorch seeds zet, zodat runs reproduceerbaar zijn. Dit was nodig omdat ik elke methode met drie seeds (1, 2, 3) train om de variantie over runs te kunnen meten.

### Overige kleinere aanpassingen
Gradient clipping (max norm 1.0) tegen exploderende gradients, lichte temporele data-augmentatie (willekeurige shift van plus/min 10 frames, consistent over views) en wat DataLoader-optimalisaties (`persistent_workers`, `prefetch_factor=4`) om de GPU-bezetting hoger te krijgen.

## Training

Alle modellen zijn getraind op [Vast.ai](https://vast.ai) met een NVIDIA RTX 4090 (24 GB VRAM). Per methode (mean, attention, cross_attention) zijn drie volledige runs uitgevoerd met seeds 1, 2 en 3, dus in totaal 9 trainingsruns van ongeveer 6 uur elk.

Een trainings-commando ziet er als volgt uit:

```bash
python "VARS model/main.py" \
  --path /workspace/sn-mvfoul/data/SoccerNet/mvfouls \
  --num_views 2 --pooling_type cross_attention --pre_model r2plus1d_18 \
  --batch_size 8 --max_epochs 40 --max_num_worker 16 \
  --LR 1e-4 --weight_decay 0.01 --patience 12 --GPU 0 \
  --start_frame 20 --end_frame 100 --fps 10 \
  --model_name THESIS_cross_attention_s1 --seed 1
```

Voor de andere methoden wissel je `--pooling_type` naar `mean` of `attention`, en voor de andere seeds wissel je `--seed` en `--model_name`. De volledige runbook met alle gebruikte commando's staat in [runbook.md](runbook.md).

## Evaluatie en resultaten

Evaluatie gebeurt met de officiele SoccerNet evaluator ([Evaluate/evaluatMV_Foul.py](Evaluate/evaluatMV_Foul.py)). De primaire metriek is de leaderboard-waarde (LB), het gemiddelde van de balanced accuracy voor offence+severity en voor action type. Per run selecteer ik de checkpoint met de beste validation LB en rapporteer ik de test LB op diezelfde epoch.

* Samengevatte resultaten per run: [results.csv](results.csv)
* Trainingslogs per run: [logs/](logs/)
* Plots (training curves, confusion matrices, per-head vergelijkingen, attention heatmaps): [plots/](plots/)
* Analyses en statistische toetsen (Wilcoxon signed-rank): [notebooks/](notebooks/)

Het volledige experimentele protocol (methoden, seeds, hyperparameters, metrieken, beslissingsregel) staat in [protocol.md](protocol.md). Dit protocol is vooraf opgesteld en niet aangepast na het zien van de resultaten, om post-hoc keuzes te vermijden.

### Vergelijking met de leaderboard

Mijn beste methode (weighted attention) haalt gemiddeld een test LB van 35.26. De hoogste score op de SoccerNet-MVFoul leaderboard ligt rond de 48. Dat is duidelijk hoger, maar ook de rest van de leaderboard zit in een vergelijkbaar bereik als mijn scores. De dataset is klein (2916 trainingsacties), veel klassen zijn zeldzaam en het verschil tussen bijvoorbeeld geel en rood is vaak subtiel.

De topscores gebruiken doorgaans een zwaardere backbone (MViT-v2 of VideoMAE in plaats van R(2+1)D-18) die end-to-end wordt fijngetuned. Bij mij staat de backbone bevroren om de training binnen het budget van een RTX 4090 te houden. Een beter backbone en end-to-end fine-tunen zou de absolute scores waarschijnlijk omhoog trekken, maar dat was niet het doel van deze thesis. Ik vergelijk aggregatiemethoden onderling bij dezelfde backbone, dus de absolute waarde van de score is minder belangrijk dan het verschil tussen methoden.

## Referentie

Held, J., Cioppa, A., Giancola, S., Hamdi, A., Ghanem, B., & Van Droogenbroeck, M. (2023). *VARS: Video Assistant Referee System for Automated Soccer Decision Making From Multiple Views*. CVPR Workshop on Computer Vision in Sports.
