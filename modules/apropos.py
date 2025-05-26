import streamlit as st

def afficher_apropos():
    st.title("ℹ️ À propos")
    st.markdown("""
### Projet : Prédiction de la Production Pétrolière par Intelligence Artificielle

Ce projet fournit un outil complet pour prédire et analyser la production pétrolière avec des fonctionnalités avancées de visualisation et d'analyse.

### Fonctionnalités principales :
- **Prédiction en temps réel** : Estimation de la production pétrolière basée sur différents paramètres opérationnels
- **Tableau de bord interactif** : Suivi des indicateurs clés de performance (KPI) et tendances
- **Visualisations avancées** : Analyses interactives et exploration approfondie des données
- **Explication des prédictions** : Compréhension des facteurs influençant les prédictions avec SHAP

### Technologies utilisées :
- **Backend** : Python, Pandas, NumPy
- **Machine Learning** : Scikit-learn, SHAP pour l'explicabilité
- **Visualisation** : Plotly, Matplotlib, Seaborn
- **Interface utilisateur** : Streamlit
- **Gestion des modèles** : Joblib

### Sources de données :
- **Données de production** : [Kaggle - Drilling Well Production Data](https://www.kaggle.com/datasets/sobhanmohammadids/drilling-well-production-data)
- **Données opérationnelles** : Données de terrain et capteurs

### Améliorations à venir :
- 🚀 Intégration de modèles plus avancés (Deep Learning)
- 🌍 Visualisation géospatiale des puits de production
- 📱 Application mobile pour le suivi en temps réel
- 🤖 API de prédiction pour une intégration avec d'autres systèmes

### Équipe :
Développé par l'équipe d'optimisation pétrolière avec l'aide de l'IA.

### Version : 2.0.0
Dernière mise à jour : Mai 2024
""")