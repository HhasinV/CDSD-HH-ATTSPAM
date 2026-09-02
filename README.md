# AT&T Spam Detector 

Détecteur automatique de spam SMS basé uniquement sur le contenu des messages.
Projet Deep Learning — certification Jedha.

## Contexte

Les utilisateurs d'AT&T sont exposés en permanence aux SMS indésirables. Le
flagging manuel ne passe pas à l'échelle : l'objectif est de construire un
modèle qui classe automatiquement chaque SMS entrant en **spam** ou **ham**
(message légitime) à partir de son seul texte.

## Dataset

- Source : [spam.csv](https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Deep+Learning/project/spam.csv)
- 5 572 SMS étiquetés, encodage `latin-1`
- Classes déséquilibrées : **86,6 % ham / 13,4 % spam** → l'accuracy seule est
  insuffisante ; les métriques principales sont le **F1 et le recall sur la
  classe spam**
- Split train/test 80/20 **stratifié**, effectué avant tout préprocessing pour
  éviter toute fuite de données

## Approche

Trois modèles de complexité croissante, pour mesurer l'apport du transfer
learning sur un petit dataset :

| # | Modèle | Idée |
|---|--------|------|
| 1 | Embedding + LSTM (from scratch) | Baseline : tout est appris sur les ~4 500 SMS d'entraînement |
| 2 | GloVe gelé + LSTM | Transfer learning léger : vecteurs de mots pré-entraînés sur 6 Mds de tokens (glove.6B.100d), couche gelée |
| 3 | DistilBERT fine-tuné | Transfer learning complet : modèle de langue contextuel pré-entraîné, fine-tuning 3 époques, lr = 5e-5 |

Préprocessing (modèles 1-2) : tokenisation Keras (vocab 10 000, jeton `<OOV>`),
padding post à `maxlen = 33` (95ᵉ percentile des longueurs). Le modèle 3
utilise son propre tokenizer WordPiece (sous-mots, aucun mot inconnu).
Early stopping (patience 3, restore best weights) sur les modèles 1-2.

## Résultats (test set, 1 115 SMS jamais vus)

| Modèle | Accuracy | Precision spam | Recall spam | F1 spam |
|--------|---------:|---------------:|------------:|--------:|
| 1. LSTM from scratch | 0.9857 | 0.9854 | 0.9060 | 0.9441 |
| 2. GloVe gelé + LSTM | 0.9839 | 0.9517 | 0.9262 | 0.9388 |
| 3. **DistilBERT fine-tuné** | **0.9919** | 0.9730 | **0.9664** | **0.9697** |

**Modèle retenu : DistilBERT.** Meilleur F1 spam (0.97) et surtout meilleur
recall (0.966) : il attrape ~97 % des spams tout en bloquant très peu de
messages légitimes. La baseline LSTM, malgré une precision élevée, laisse
passer près d'1 spam sur 10 (recall 0.906) — un écart invisible si l'on ne
regardait que l'accuracy.

Enseignements : (1) sur un petit corpus, le facteur limitant est la donnée,
pas l'architecture — le LSTM from scratch overfitte dès la 2ᵉ époque ;
(2) les embeddings pré-entraînés stabilisent l'entraînement ; (3) le modèle
contextuel pré-entraîné domine, avec seulement 3 époques de fine-tuning.

## Limites et pistes

- Dataset en anglais uniquement ; l'argot SMS évolue → réentraînement périodique
- Seuil de décision ajustable selon la tolérance métier d'AT&T
  (faux positifs vs spams manqués)
- Piste : distillation ou quantification de DistilBERT pour une inférence
  temps réel à moindre coût

## Auteur
**Henintsoa HASINAVALONA**
