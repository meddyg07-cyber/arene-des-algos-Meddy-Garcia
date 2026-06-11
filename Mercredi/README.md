Jour 3 : Les algorithmes du Machine Learning  et le Fight des IA

# Arène des algos — fight_ia

Ce notebook regroupe quatre exercices pratiques de machine learning, pensés pour comparer des approches différentes sur des problèmes variés et observer ce qui se passe quand les conditions ne sont pas idéales.

L'idée est simple : pour chaque problème, on teste le cas "tout va bien", un cas limite qui pousse le modèle dans ses retranchements, et un cas adversarial conçu pour le faire déraper.

---

## Ce qu'il y a dedans

### Phase A — Prédire les prix immobiliers

On travaille sur le dataset California Housing pour estimer les prix de l'immobilier. Deux modèles sont comparés : une régression linéaire classique et un Random Forest.

Le cas limite réduit volontairement le jeu d'entraînement à seulement 100 lignes pour voir l'effet sur les performances. Le cas adversarial injecte un quartier fictif avec des valeurs complètement hors plage (revenu médian à zéro, population à 9000 habitants) pour observer que la régression linéaire extrapole sans aucune garde-fou et peut retourner des prix négatifs ou aberrants.

---

### Phase B — Segmenter les annonces Airbnb

On charge un fichier de listings Airbnb (CSV zippé) et on applique un clustering K-Means sur cinq variables : prix, minimum de nuits, capacité, note et nombre de chambres.

Le k optimal est sélectionné automatiquement via le score silhouette. Les segments obtenus sont ensuite interprétés (budget, premium, longue durée, familial...).

Deux expériences complémentaires sont menées : d'abord sans standardisation, pour montrer que la variable "price" (0-1000) écrase complètement "bedrooms" (1-5) et rend les segments sans sens ; ensuite avec un outlier fictif à 100 000€ la nuit, pour observer comment un seul point aberrant déplace tous les centres du clustering.

---

### Phase C — Détection de spam

Le dataset SMS Spam Collection (~5500 messages) est utilisé pour construire un filtre anti-spam. Les textes sont vectorisés avec TF-IDF, puis deux classifieurs sont comparés : Naive Bayes multinomial et régression logistique.

L'exercice teste aussi un message vide (pour vérifier que le pipeline ne plante pas) et un spam déguisé formulé en langage courant, pour illustrer que regarder la probabilité de décision vaut mieux que la classe prédite seule.

---

### Phase D — Classifier des signaux sonar

Le dataset Sonar contient des signaux acoustiques qui correspondent soit à des mines, soit à des rochers. Trois modèles sont comparés sur 60 variables numériques : régression logistique, SVM à noyau RBF et Random Forest.

L'effet de la standardisation est mesuré directement : les modèles SVM et régression logistique perdent beaucoup de performance sans mise à l'échelle, contrairement au Random Forest qui s'en moque. Le cas adversarial simule un capteur complètement en panne (signal à 60 zéros) : tous les modèles prédisent quand même avec une grande assurance, ce qui illustre pourquoi il faut valider les entrées en production avant même d'appeler le modèle.

---

## Comment lancer

Cloner le dépôt et s'assurer que les fichiers de données sont présents dans le même répertoire que le notebook :

- `Listings.zip` (listings Airbnb)
- `SMSSpamCollection` (dataset spam SMS)
- `sonar.all-data` (signaux sonar UCI)

Les dépendances sont toutes dans scikit-learn, pandas, numpy et matplotlib — rien d'exotique.

```bash
pip install scikit-learn pandas numpy matplotlib
jupyter notebook fight_ia.ipynb
```

---

## Structure du repo

```
arene-des-algos-Meddy-Garcia/
└── Mercredi/
    ├── fight_ia.ipynb
    ├── Listings.zip
    ├── SMSSpamCollection
    └── sonar.all-data
```




