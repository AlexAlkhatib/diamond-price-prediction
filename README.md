# 💎 Diamond Price Prediction

*Régression, feature engineering et modèles d'ensemble*

## 📌 Description du Projet

Ce projet prédit le **prix d'un diamant** en dollars à partir de ses caractéristiques physiques
(poids, qualité de la coupe, couleur, pureté, dimensions) et **explique** ce qui détermine ce prix.

L'ensemble de l'analyse tient dans un **notebook unique, déjà exécuté** (figures et résultats visibles
sans rien relancer) :
 [`diamond-price-prediction.ipynb`](diamond-price-prediction.ipynb)

Le projet met l'accent sur :

- ✔️ Un **nettoyage fondé sur la physique** du diamant (dimensions, densité, cohérence de la profondeur)
- ✔️ L'**encodage ordinal** des variables de qualité plutôt que le one-hot
- ✔️ Le **feature engineering** géométrique, mesuré par ablation
- ✔️ La **comparaison honnête** de 7 modèles en validation croisée
- ✔️ L'**interprétation** de chaque figure et de chaque résultat (dont un paradoxe de Simpson)
- ✔️ Un modèle **utilisable en production** (pipeline complet sauvegardé, avec garde-fou physique)

---

## 📂 Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `dataset.csv` | Jeu de données source (53 940 diamants) |
| `diamond-price-prediction.ipynb` | **Livrable principal** : analyse complète, exécutée |
| `diamond_price_pipeline.joblib` | Pipeline entraîné (feature engineering + modèle), généré par le notebook |
| `diamond-price-prediction_v1_original.ipynb` | Version initiale conservée pour comparaison |

> ⚠️ **Il n'y a pas de `train.py` ni de `evaluate.py`.** Tout l'entraînement et l'évaluation sont dans le
> notebook, qui produit lui-même le fichier `.joblib`. Un `requirements.txt` n'est pas fourni : les
> dépendances sont listées ci-dessous.

---

## 📊 Dataset

**Source :** [Kaggle — Diamond Price Prediction Dataset](https://www.kaggle.com/datasets/ronil8/diamond-price-prediction-dataset)
(équivalent du jeu *diamonds* de ggplot2).

53 940 lignes, 10 colonnes utiles (la première colonne, `Unnamed: 0`, n'est qu'un identifiant de ligne).

| Variable | Signification | Type |
|---|---|---|
| `price` | prix en dollars US (**cible**) | numérique |
| `carat` | poids (0,2 – 5,01 ct) | numérique |
| `cut` | Fair → Good → Very Good → Premium → Ideal | **ordinale** |
| `color` | J (la moins blanche) → D (la plus blanche) | **ordinale** |
| `clarity` | I1 → SI2 → SI1 → VS2 → VS1 → VVS2 → VVS1 → IF | **ordinale** |
| `depth` | profondeur totale en % | numérique |
| `table` | largeur du plateau en % | numérique |
| `x`, `y`, `z` | longueur, largeur, hauteur en mm | numérique |

Aucune valeur manquante. Les vrais défauts sont ailleurs : **146 doublons** et **62 lignes physiquement
impossibles** (dimensions nulles, `y = 58,9 mm`, `z = 31,8 mm`).

---

## 🔬 Démarche

### 1. Audit et nettoyage (`100 % tracé`)

53 940 → **53 732 lignes** (99,6 % conservées), avec un rapport étape par étape dans le notebook :

1. suppression de l'identifiant ;
2. **146 doublons** exacts ;
3. géométrie valide (`0 < x, y ≤ 11 mm`, `0 < z ≤ 8 mm`) ;
4. `table` et `depth` dans des bornes physiques (43–75 % / 45–75 %) ;
5. **cohérence de la profondeur** : on recalcule `depth_est = 200·z/(x+y)` et on écarte les écarts > 3 points.

Ce dernier contrôle est le plus utile : il détecte les erreurs de saisie que les statistiques descriptives
ne voient pas. La **densité** `carat / (x·y·z)` sert de seconde vérification — elle doit valoir ≈ **0,0061**.

### 2. Encodage ordinal (et non one-hot)

| Variable | Encodage |
|---|---|
| `cut` | Fair 1 → Good 2 → Very Good 3 → Premium 4 → Ideal 5 |
| `color` | J 1 → I 2 → H 3 → G 4 → F 5 → E 6 → D 7 |
| `clarity` | I1 1 → SI2 2 → SI1 3 → VS2 4 → VS1 5 → VVS2 6 → VVS1 7 → IF 8 |

Ces trois variables sont **ordonnées** : le one-hot détruit cette information et éclate 3 colonnes en 20.
Résultat mesuré : **527 $ de RMSE en ordinal contre 541 $ en one-hot** (16 variables contre 32).

### 3. Feature engineering

| Feature | Formule | Intérêt |
|---|---|---|
| `volume` | `x · y · z` | volume réel de la pierre |
| `face_area` | `x · y` | surface visible du dessus |
| `density` | `carat / volume` | contrôle physique (≈ 0,0061) |
| `xy_ratio` | `x / y` | symétrie (1 = ronde) |
| `depth_est` | `200·z / (x+y)` | profondeur recalculée |
| `depth_gap` | `depth − depth_est` | erreur de mesure / taille atypique |
| `quality_score` | `color_ord + clarity_ord` | score de qualité global (2 à 15) |

**Ablation (même modèle, validation croisée 3-fold) :**

| Jeu de variables | RMSE CV | R² |
|---|---|---|
| A. Numériques brutes | 1 353 $ | 0,885 |
| B. + qualités ordinales | **536 $** | 0,982 |
| C. + géométrie (retenu) | **527 $** | **0,983** |
| C′. Géométrie + one-hot | 541 $ | 0,982 |

L'encodage ordinal est le gain décisif (**−60 %** de RMSE) ; la géométrie apporte −1,7 % supplémentaire.

### 4. Modélisation

Découpage **80 / 20** (`random_state=42`), puis validation croisée **3-fold sur le train uniquement**.
Le jeu de test n'est utilisé qu'une fois, à la fin.

- `DummyRegressor` (référence « prix moyen »)
- **Régression linéaire** et **Ridge** (`StandardScaler`)
- **KNN** (k=7, distance pondérée)
- **Arbre de décision**
- **Random Forest** (bagging, 100 arbres)
- **HistGradientBoosting** (boosting) — réglé par `RandomizedSearchCV` (10 tirages, 3 folds)

### 5. Cible logarithmique

Le prix est fortement asymétrique (skewness **1,62** contre **0,12** pour `log(prix)`). Le modèle retenu
apprend `log(prix)` via `TransformedTargetRegressor` : la validation croisée est donc notée **en dollars**,
directement comparable au modèle brut.

---

## 📈 Résultats

### Validation croisée (train)

| Modèle | RMSE ↓ | R² ↑ | MAE ↓ |
|---|---|---|---|
| Dummy (prix moyen) | 3 990 $ | 0,000 | 3 035 $ |
| Régression linéaire | 1 137 $ | 0,919 | 764 $ |
| Ridge (alpha=1) | 1 153 $ | 0,916 | 765 $ |
| KNN (k=7) | 752 $ | 0,964 | 399 $ |
| Arbre de décision | 764 $ | 0,963 | 371 $ |
| Random Forest | 531 $ | 0,982 | 263 $ |
| **HistGradientBoosting** | **527 $** | **0,983** | **273 $** |

Points saillants :

- **Le non-linéaire est indispensable** : 1 137 $ (linéaire) → 527 $ (boosting).
- **L'arbre seul est un mauvais choix** : 5ᵉ au classement, avec un écart-type de 14 $ entre folds (instable).
  La version initiale du projet le retenait pourtant, sur un seul split.
- **Random Forest est excellent mais bien plus lent** (≈ 10–20×) que le boosting sur ce volume de données.

### Jeu de test (résultats du modèle retenu)

| Métrique | Valeur |
|---|---|
| **RMSE** | **515 $** |
| **MAE** | **256 $** |
| Erreur médiane | 91 $ |
| **R²** | **0,983** |
| **MAPE** | **5,9 %** |
| Prédictions à ± 10 % du prix réel | **82,9 %** |
| Prédictions à ± 20 % du prix réel | **96,5 %** |
| Biais moyen | +0,13 % |

À titre de comparaison, la régression linéaire obtient 1 126 $ de RMSE et **41,3 %** de MAPE sur le même test.

L'erreur **absolue** croît avec le prix (27 $ sur les pierres les moins chères, 713 $ sur les plus chères)
mais l'erreur **relative** reste stable (4,3 % à 6,7 %). Les segments les plus difficiles sont les pierres
rares : `I1` (10,3 % de MAPE), `Fair` (9,8 %), plus de 2 ct (8,3 %).

---

## 🧠 Interprétation

- **La taille domine** : `carat` est corrélé à 0,92 au prix, et la taille représente **81 %** de l'importance
  des variables, la qualité **18 %**, les proportions **1 %**. Le prix suit une loi de puissance du poids.
- **La pureté est le premier critère de qualité** : à poids comparable (0,9–1,1 ct), le prix au carat va de
  **2 736 $** (I1) à **9 871 $** (IF), soit un facteur **3,6**. La couleur suit (J → D : +42 %).
- **Paradoxe de Simpson** : sur données brutes, `Fair` (3 281 $) semble plus cher que `Ideal` (1 813 $), ce qui
  est absurde. La cause est le poids (1,04 ct en moyenne pour `Fair` contre 0,70 ct pour `Ideal`). À poids
  comparable, la hiérarchie s'inverse : `Ideal` 5 103 $/ct, `Fair` 3 763 $/ct. **Toute analyse qui ne contrôle
  pas le carat tire une conclusion commerciale fausse.**
- **Profondeur et table** : effet non linéaire (optimum ≈ 61–62 % et 56–58 %), invisible en corrélation.
- **`cut` a l'effet le plus faible**, et `Premium` se comporte comme `Very Good` : l'échelle ordinale est une
  approximation assumée (signalée dans les limites du notebook).

---

## 🔮 Utilisation du modèle

Le fichier `.joblib` contient le **pipeline complet** : il accepte des colonnes **brutes**, sans reproduire
l'encodage ni le réalignement des colonnes.

```python
import joblib
import pandas as pd

artefact = joblib.load("diamond_price_pipeline.joblib")
pipeline = artefact["pipeline"]

diamant = pd.DataFrame([{
    "carat": 1.00, "cut": "Ideal", "color": "D", "clarity": "VS1",
    "depth": 61.5, "table": 57.0, "x": 6.40, "y": 6.43, "z": 3.94,
}])

prix = pipeline.predict(diamant)[0]
print(f"Prix estimé : {prix:,.0f} $")   # ≈ 9 464 $
```

L'artefact contient aussi `features`, `residual_calibration` (intervalles à 90 % par tranche de prix),
`metrics_test` et `train_carat_range`.

⚠️ **Garde-fou** : le notebook fournit une fonction de validation physique (densité, bornes de poids et de
dimensions, plages de `depth`/`table`). Un diamant hors domaine est **refusé** plutôt que prédit par
extrapolation. Exemple de résultat :

| Diamant | Prix prédit | Intervalle 90 % | Référence marché |
|---|---|---|---|
| 1,0 ct Ideal D VS1 | 9 464 $ | 7 582 – 11 072 $ | 8 700 $ (22 pierres comparables) |
| 0,51 ct Good J SI2 | 960 $ | 827 – 1 115 $ | trop peu de comparables |
| 10 ct, 20 × 20 × 30 mm | — | — | **rejeté** (5 incohérences physiques) |

---

## 🚀 Technologies Utilisées

- **Python 3.12**
- `pandas` 3.1.1 · `numpy` 2.5.5
- `scikit-learn` 1.9.1
- `matplotlib` / `seaborn` (19 figures)
- `joblib` (sérialisation du pipeline)

---

## ▶️ Exécuter le Projet

### 1. Cloner le dépôt

```bash
git clone https://github.com/username/diamond-regression.git
cd diamond-regression
```

### 2. Installer les dépendances

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### 3. Ouvrir le notebook

```bash
jupyter notebook diamond-price-prediction.ipynb
```

Le notebook est **déjà exécuté** : les figures et résultats sont visibles sans rien lancer.
Pour tout régénérer : *Kernel → Restart & Run All* (≈ 2 minutes sur 2 cœurs).
Il relit `dataset.csv` (même dossier) et réécrit `diamond_price_pipeline.joblib`.

---

## ⚠️ Limites

- **Données déclaratives et incomplètes** : ni fluorescence, ni polissage, ni symétrie, ni certificat, ni lieu
  ou date de vente. D'où un plancher d'erreur d'environ 5 %.
- **Segments rares mal couverts** (`I1`, `Fair`, > 2 ct) : MAPE jusqu'à 10–19 %.
- **Extrapolation interdite** : au-delà de 5,01 ct ou 10,74 mm, aucune prédiction n'est valide.
- **Encodage ordinal approximatif** : l'échelle `cut` n'est pas régulière.
- **Un seul jeu de test** : une validation croisée imbriquée serait plus rigoureuse.

---

## ✨ Améliorations Futures

- **XGBoost / LightGBM / CatBoost** et comparaison avec le HistGradientBoosting actuel
- **Target encoding** pour `cut` / `color` / `clarity`, ou intégration de la grille Rapaport
- **Régression quantile** (prédire P10 / P50 / P90) plutôt qu'un intervalle constant par tranche
- **Optuna** pour une recherche d'hyperparamètres plus large
- **SHAP** pour une interprétabilité locale par diamant
- Déploiement via **FastAPI** ou **Streamlit**
- *(Non implémenté à ce jour : `VotingRegressor` / `StackingRegressor`)*

---

## 👤 Auteur

**Alex Alkhatib**
Projet de Data Science — Modélisation prédictive en Machine Learning

## 📄 Licence

MIT License
Copyright (c) 2025 Alex Alkhatib