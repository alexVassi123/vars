# Resultaten ( nog niet af, in de notebooks is het wat uitgebreider)

## Samenvatting

Gemiddeld over 3 seeds, per methode:

| Methode | Val LB | Test LB |
|---|---|---|
| mean | 38.55 | 34.45 |
| attention | 39.46 | **35.26** |
| cross_attention | **39.95** | 33.13 |

Per seed test LB:

- mean: 32.89 / 38.41 / 32.04 (std ongeveer 3.5)
- attention: 36.77 / 34.16 / 34.84 (std ongeveer 1.3)
- cross_attention: 35.99 / 31.57 / 31.82 (std ongeveer 2.5)

## Interpretatie

Cross_attention wint op validation maar verliest op test. De extra capaciteit (ongeveer 1M parameters tegenover 262k bij attention en 0 bij mean) past het model beter aan op de validation set, maar die winst generaliseert niet naar test. Attention is gemiddeld het beste op test en daarnaast het meest consistent over seeds.

De beste individuele run komt van `mean` met seed 2 (test LB 38.41). De variantie van die methode is dus hoog: dezelfde methode scoort ook 32.04 en 32.89 op andere seeds.

## Statistische toets

Volgens het protocol doe ik een paarsgewijze Wilcoxon signed-rank toets per seed. Met N=3 seeds is de laagst mogelijke p-waarde in deze toets 0.25. Significantie bij alpha=0.05 is dus wiskundig onbereikbaar, hoe groot het verschil tussen methoden ook is. De toets is daarom vooral opgenomen voor de volledigheid, en ik beschrijf de resultaten als descriptief in plaats van confirmatief.

## Waarom niet meer seeds of een zwaardere backbone

Om significantie mogelijk te maken zou ik minstens 6 seeds per methode moeten draaien (minimale p-waarde 0.03 bij N=6). Dat zijn 9 extra trainingsruns van ongeveer 6 uur op een RTX 4090. Op Vast.ai valt die kost buiten het budget van deze thesis.

Een vergelijkbare afweging geldt voor een grotere backbone zoals MViT-v2 of VideoMAE. Die geven rijkere features en zouden cross_attention waarschijnlijk meer ruimte geven om zich te onderscheiden, maar ze zijn zwaarder om te trainen. Een volledige vergelijking met een video transformer als backbone zou nog eens 9 tot 18 runs betekenen, wat opnieuw buiten het budget valt.

Met de middelen die ik heb is de conclusie dus descriptief: cross_attention levert bij deze dataset-grootte en deze backbone geen verbetering op test ten opzichte van de originele attention-methode. Of dat met meer data of een zwaardere backbone wel zou gebeuren, kan op basis van deze resultaten niet uitgesloten maar ook niet bevestigd worden.
