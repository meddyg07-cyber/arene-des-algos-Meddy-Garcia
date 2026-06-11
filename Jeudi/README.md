jeudi :Le Deep Learning, l'évaluation et la mise en production

L’idée de cet après-midi est simple : apprendre à vraiment juger un modèle de machine learning et à le rendre utilisable dans une vraie application.

En pratique, entraîner un modèle ne suffit pas. Il faut aussi vérifier s’il est fiable, comprendre comment l’évaluer correctement, et surtout savoir le transformer en outil accessible (API, application web, etc.).

Jeu de données utilisé

On travaille sur le dataset Telco Customer Churn, qui sert à prédire le départ (ou non) des clients dans une entreprise télécom.

7 043 lignes, 21 colonnes
Cible : Churn (Yes / No)
Variables mixtes :
numériques : tenure, MonthlyCharges, TotalCharges
catégorielles : Contract, PaymentMethod, InternetService, etc.

Ce dataset est intéressant pour l’exercice car il s’agit d’une classification binaire avec un vrai mélange de types de données, ce qui oblige à faire un encodage propre et réfléchi.

Déroulé de l’après-midi

On avance étape par étape :

Phase	Contenu
Phase 0	Mise en route, README, premier commit
Phase 1	Split train / validation / test avec stratification
Phase 2	Bootstrap et bagging pour analyser la stabilité du modèle
Phase 3	Validation croisée (k-fold)
Phase 4	Choix des métriques selon le coût métier (precision, recall, F1, coût total)
Phase 5	Sauvegarde du modèle + API Flask
Phase 6	Application Streamlit pour les prédictions
Phase 7	Comparaison finale et sélection du meilleur modèle
Structure du projet
jour4/
├── README.md
├── notebooks/
│   └── jour4_churn.ipynb       # notebook principal (une cellule par phase)
├── src/
│   ├── split.py                # séparation train/val/test
│   ├── bootstrap.py            # analyse bootstrap
│   ├── crossval.py             # validation croisée
│   ├── metrics.py              # analyse métier des résultats
│   ├── api.py                  # API Flask
│   └── app.py                  # interface Streamlit
├── models/
│   └── modele.joblib           # modèle + scaler sauvegardés
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
Suivi des commits

Chaque étape correspond à un commit clair :

[phase-0] Mise en place du projet + README
[phase-1] Fonction de split avec stratification
[phase-2] Bootstrap et analyse de stabilité
[phase-3] Validation croisée
[phase-4] Ajout du rapport métier (coûts FN/FP)
[phase-5] API Flask + sérialisation du modèle
[phase-6] Application Streamlit
[phase-7] Comparaison finale et choix du modèle
Dépendances
scikit-learn
pandas
numpy
joblib
flask
streamlit

Installation :

pip install scikit-learn pandas numpy joblib flask streamlit
Notes

Les commits sont faits proprement, étape par étape, avec un message clair à chaque fois.
Chaque fichier est ajouté individuellement (git add fichier.py) plutôt qu’en bloc, pour garder un historique propre et lisible.
