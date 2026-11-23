# Petola — Prédiction de Production Pétrolière Journalière (ML · XAI)

> Application Streamlit de prédiction de la production pétrolière journalière à partir de 7 paramètres de puits, avec **explicabilité SHAP** (XAI), scénarios de forecast et analyse de données intégrée.

## Problématique

Estimer la production journalière d'un puits nécessite de croiser des paramètres opérationnels (pressions, températures, dimensions de choke). Sans modèle, ces estimations reposent sur l'expérience de l'opérateur et restent difficilement transmissibles ou auditables.

## Solution

Un modèle **RandomForest** entraîné sur des données réelles de production (15 641 enregistrements), servi par une application Streamlit multi-modules — chaque prédiction est accompagnée de l'**importance des variables** (SHAP) pour comprendre ce qui conduit le chiffre.

## Fonctionnalités

### Prédiction & explicabilité
- **Prédiction instantanée** : saisie de 7 paramètres de puits (pression, température, ΔP tubing, annulus, choke, WHP, WHT)
- **Explications SHAP activables** : importance des variables affichée avec la prédiction (vérifié dans `modules/prediction.py`)
- **Analyse de sensibilité** et **historique des prédictions**

### Forecast & scénarios
- Prévisions futures avec **3 scénarios** (optimiste / nominal / pessimiste) + tableau récapitulatif

### Analyse de données
- Tableau de bord KPI, **matrice de corrélation**, analyse historique
- **Détection d'anomalies** (IQR / Z-score)

## KPI vérifiables

- **7 paramètres** d'entrée · **3 scénarios** de forecast
- **Dataset : 15 641 enregistrements** (`data/prediction.csv`)
- **SHAP** intégré (toggle d'explication)
- **Tests pytest** réels (`tests/test_helpers.py`)
- **4 modules** applicatifs (`dashboard`, `prediction`, `visualisations`, `apropos`)

## Stack

Python · Streamlit · scikit-learn (RandomForest) · **SHAP** · Plotly · statsmodels · seaborn · loguru · pytest

## Démarrage

```bash
# 1. Environnement
python -m venv venv && source venv/bin/activate   # Windows : .\venv\Scripts\activate

# 2. Dépendances
pip install -r requirements.txt

# 3. Lancement
streamlit run app.py    # http://localhost:8501
```

Structure du modèle : le classifieur entraîné est servi depuis `modeles/random_forest_model.pkl`.

## Structure du dépôt

| Chemin | Rôle |
|---|---|
| `app.py` | Point d'entrée Streamlit |
| `modules/` | Modules applicatifs (dashboard, prediction, visualisations, apropos) |
| `config/` | Paramètres et thème |
| `utils/` | Helpers, notifications |
| `modeles/` | Modèle RandomForest entraîné (`.pkl`) |
| `data/prediction.csv` | Données de production (15 641 lignes) |
| `tests/` | Tests pytest |

## Déploiement

L'application est déployée sur Streamlit Community Cloud — l'accès est **protégé par authentification** : [petoladata.streamlit.app](https://petoladata.streamlit.app)

## Statut

`COMPLET` — Modèle entraîné, app testée (pytest) et déployée. Projet Data Science à vocation démonstrative (XAI + forecast opérationnel).

> Note : le chiffre de performance exact du modèle n'est pas affiché ici tant qu'une validation croisée documentée n'a pas été committée dans le dépôt — les KPI ci-dessus sont structurels et vérifiables dans le code.
