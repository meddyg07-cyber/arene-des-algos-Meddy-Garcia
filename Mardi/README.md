# Data pour le Machine Learning — Collecte et Preprocessing

Notebook de préparation de données ML appliqué au dataset **Telco Customer Churn** (IBM/Kaggle). L'objectif est de passer d'un CSV brut à un DataFrame propre, entièrement numérique, et prêt pour l'entraînement d'un modèle.

---

## Description

Le dataset contient des informations sur 7 043 clients d'un opérateur télécom, avec 21 colonnes (données démographiques, services souscrits, facturation, résiliation). La variable cible est `Churn` : est-ce que le client a résilié ou non.

Le notebook couvre toutes les étapes de preprocessing : audit, nettoyage, encodage, traitement des outliers et analyse de multicolinéarité.

---

## Structure du notebook

### Phase 0 — Récupérer et ouvrir la donnée

Téléchargement du dataset via `kagglehub`, chargement avec pandas. Le DataFrame se charge correctement : 7 043 lignes, 21 colonnes, aucune valeur manquante apparente.

### Phase 1 — Audit qualité

Vérification de la forme, des types, des valeurs manquantes et de l'équilibre de la cible. Résultat : aucun NaN visible, mais la colonne `TotalCharges` est de type `object` alors qu'elle devrait être numérique — signe qu'il y a des valeurs cachées à traiter. La cible est légèrement déséquilibrée : 73,5 % de non-résiliation vs 26,5 % de résiliation.

### Phase 2 — La colonne piégée (types incohérents et trous cachés)

`TotalCharges` contient 11 valeurs à espace (` `) qui empêchent la conversion automatique. La fonction `reparer_total_charges` :

- détecte les espaces
- convertit la colonne en `float64` avec `pd.to_numeric(..., errors='coerce')`, ce qui transforme les espaces en `NaN`
- impute les NaN avec la médiane (1 397,47)

Note : avant conversion, on peut aussi rechercher les virgules avec `str.contains` pour détecter d'éventuelles corruptions de format.

### Phase 3 — Encoder les variables catégorielles

Le DataFrame contient beaucoup de colonnes texte. La fonction `encoder_features` :

- supprime `customerID` (identifiant sans valeur prédictive)
- encode les colonnes binaires Yes/No en 0/1
- applique un One-Hot Encoding (`get_dummies`) sur les colonnes nominales restantes

Résultat : 31 colonnes, toutes numériques.

### Phase 4 — Traiter les valeurs aberrantes

Détection des outliers via la méthode IQR sur les colonnes numériques continues (`tenure`, `MonthlyCharges`, `TotalCharges`). Aucun outlier détecté dans ce dataset, ce qui est cohérent avec des données de facturation réelles.

### Phase 5 — Multicolinéarité

Analyse des corrélations entre variables numériques via une heatmap et calcul du VIF (Variance Inflation Factor). Les variables avec un VIF > 5 sont signalées comme à risque. Cette étape permet de repérer les variables redondantes avant l'entraînement.

---

## Dépendances

```
pandas
kagglehub
seaborn
matplotlib
statsmodels
```

```bash
pip install pandas kagglehub seaborn matplotlib statsmodels
```

## Lancement

```bash
jupyter notebook data_pour_ml.ipynb
```

Il faut avoir un compte Kaggle et avoir configuré les credentials `kagglehub` pour que le téléchargement du dataset fonctionne.
