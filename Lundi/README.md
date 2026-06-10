# arene-des-algos-Meddy-Garcia

# L'Arène des Algorithmes

Notebook d'exploration du machine learning supervisé et non supervisé, sur le dataset Breast Cancer Wisconsin de scikit-learn.

---

## Description

Ce notebook suit les grandes étapes d'un pipeline ML classique : exploration des données, entraînement d'un modèle, comparaison de plusieurs algorithmes, et un peu de clustering non supervisé. Le tout sur un cas concret de classification de tumeurs (bénigne / maligne).

---

## Structure du notebook

### Phase 1 — Exploration du dataset

On commence par charger le dataset (569 cas, 30 features) et regarder comment les classes sont réparties : 37,3 % maligne contre 62,7 % bénigne. On teste aussi deux cas particuliers : un dataset filtré sur la classe maligne uniquement, et un dataset volontairement déséquilibré pour montrer pourquoi l'accuracy seule peut être trompeuse.

### Phase 2 — Entraînement et évaluation

Split train/test classique (80/20, `random_state=42` pour avoir des résultats reproductibles), un `RandomForestClassifier`, et on obtient 96,49 % d'accuracy sur le jeu de test.

### Phase 3 — La comparaison de modèles

On met trois algorithmes en concurrence sur le même split :

| Modèle | Accuracy |
|--------|----------|
| Régression logistique | 95,61 % |
| KNN | 95,61 % |
| Arbre de décision | 94,74 % |

La régression logistique et le KNN ont le même score, ce qui est assez rare et mérite d'être vérifié. Note : `max_iter=5000` est nécessaire sur la régression logistique, la valeur par défaut ne suffit pas à faire converger le modèle sur ce dataset.

### Phase 4 — Clustering non supervisé

On applique un `KMeans` à 2 clusters sans donner les étiquettes au modèle. Il retrouve partiellement la structure réelle des classes, ce qui montre qu'il y a bien une séparation dans les données, même si les résultats restent moins fiables qu'un modèle supervisé.

---

## Dépendances

```
scikit-learn
numpy
```

```bash
pip install scikit-learn numpy
```

## Lancement

```bash
jupyter notebook arene_des_algos.ipynb
```
