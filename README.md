# 💎 Diamond Price Prediction

*Modèles de régression & Ensemble Learning*

## 📌 Description du Projet

Ce projet a pour objectif de **prédire le prix des diamants** à partir de différentes caractéristiques (carat, coupe, couleur, clarté, dimensions, etc.).
Pour améliorer la performance, plusieurs modèles de machine learning sont implémentés :

* **Régression Linéaire**
* **Arbre de Décision**
* **Random Forest**
* **Ensemble Learning (moyenne ou stacking)**

Le projet met l’accent sur :
- ✔️ La préparation des données
- ✔️ La comparaison des performances
- ✔️ L’amélioration de la précision grâce aux modèles d’ensemble
- ✔️ L’interprétation des résultats


## 📂 Dataset

Le dataset utilisé contient plusieurs colonnes telles que :

* `carat` — poids du diamant
* `cut` — qualité de la coupe
* `color` — couleur
* `clarity` — pureté
* `x, y, z` — dimensions
* `price` — variable cible

Dataset typique : **diamonds.csv** (Kaggle).


## 🧹 Prétraitement

Les étapes de préparation incluent :

* Gestion des valeurs manquantes
* Encodage des variables catégorielles (`cut`, `color`, `clarity`) via **OneHotEncoder**
* Normalisation/standardisation des variables numériques
* Séparation train/test


## 🧠 Modèles Implémentés

### 🔹 1. Régression Linéaire

Modèle simple permettant d’obtenir une baseline.

**Points forts :**

* Interprétable
* Rapide

### 🔹 2. Arbre de Décision

Capture les relations non linéaires.

**Hyperparamètres typiques :**

* `max_depth`
* `min_samples_split`

### 🔹 3. Forêt Aléatoire (Random Forest)

Modèle robuste basé sur l’agrégation de plusieurs arbres.

**Avantages :**

* Excellent compromis biais/variance
* Gère bien les features hétérogènes
* Réduit fortement l’overfitting

### 🔹 4. Ensemble Learning

Combinaison de modèles pour augmenter la précision.
Deux approches possibles :

* **Voting Regressor** (moyenne des prédictions)
* **Stacking Regressor** (modèle méta-apprenant)


## 📊 Évaluation des Modèles

Les métriques utilisées sont :

* **RMSE** (Root Mean Squared Error)
* **MAE** (Mean Absolute Error)
* **R² Score**

Exemple de code pour comparer :

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate(y_true, y_pred):
    return {
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
```


## 📈 Résultats (exemple)

| Modèle              | RMSE ↓  | R² ↑     |
| ------------------- | ------- | -------- |
| Régression Linéaire | 1100    | 0.88     |
| Arbre de Décision   | 900     | 0.93     |
| Random Forest       | **650** | **0.97** |
| Ensemble Learning   | **620** | **0.98** |

*(Les chiffres peuvent varier selon l’entraînement.)*


## 🚀 Technologies Utilisées

* **Python 3.x**
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib` / `seaborn`


## ▶️ Exécuter le Projet

### 1. Cloner le dépôt

```bash
git clone https://github.com/username/diamond-regression.git
cd diamond-regression
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l’entraînement

```bash
python train.py
```

### 4. Visualiser les résultats

```bash
python evaluate.py
```


## ✨ Améliorations Futures

* Ajout de **XGBoost / LightGBM / CatBoost**
* Technique avancée : **hyperparameter tuning (GridSearchCV, Optuna)**
* Deployment via **FastAPI** ou **Streamlit**
* Interprétabilité via **SHAP values**


## 👤 Auteur

**Alex Alkhatib**
Projet de Data Science — Modélisation prédictive en Machine Learning


## 📄 Licence
MIT License
Copyright (c) 2025 Alex Alkhatib
