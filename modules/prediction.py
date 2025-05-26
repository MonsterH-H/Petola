import streamlit as st
import numpy as np
import joblib
import os
import datetime
import pandas as pd
import logging
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta
from config.settings import MODEL_CONFIG, DATA_CONFIG
from utils.helpers import load_data, save_prediction, format_production

# Configuration du logging
logger = logging.getLogger(__name__)

# Chemins depuis la configuration
MODEL_PATH = MODEL_CONFIG["random_forest"]["path"]
HISTORIQUE_PATH = DATA_CONFIG["prediction_path"]

def convert_dates(df):
    """Convertit les colonnes de date au format datetime."""
    date_cols = df.select_dtypes(include=['object']).filter(regex='date|Date', axis=1).columns
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], dayfirst=True)
        except ValueError:
            continue
    return df

def generer_predictions_futures(model, params_base, periode='semaine', nb_periodes=4):
    """Génère des prédictions futures basées sur des variations de paramètres."""
    predictions = []
    dates = []
    
    # Définir les intervalles selon la période
    if periode == 'semaine':
        interval = datetime.timedelta(weeks=1)
        format_date = '%Y-W%U'
    elif periode == 'mois':
        interval = relativedelta(months=1)
        format_date = '%Y-%m'
    else:  # année
        interval = relativedelta(years=1)
        format_date = '%Y'
    
    date_actuelle = datetime.datetime.now()
    
    for i in range(nb_periodes):
        # Simulation de variations légères des paramètres (±5%)
        variation = 1 + (np.random.normal(0, 0.05))  # Variation normale de ±5%
        params_varies = params_base * variation
        
        # Prédiction
        pred = model.predict([params_varies])[0]
        predictions.append(max(0, pred))  # Éviter les valeurs négatives
        
        # Date future
        if periode == 'semaine':
            dates.append(date_actuelle + (i * interval))
        elif periode == 'mois':
            dates.append(date_actuelle + (i * interval))
        else:
            dates.append(date_actuelle + (i * interval))
    
    return dates, predictions

def calculer_tendances(df, colonne_production='Production', colonne_date='Date'):
    """Calcule les tendances de production."""
    if colonne_date not in df.columns or colonne_production not in df.columns:
        return None
    
    df = df.copy()
    df[colonne_date] = pd.to_datetime(df[colonne_date])
    df = df.sort_values(colonne_date)
    
    # Tendance générale (régression linéaire simple)
    x = np.arange(len(df))
    y = df[colonne_production].values
    
    if len(y) > 1:
        coeffs = np.polyfit(x, y, 1)
        tendance = coeffs[0]  # Pente
        
        if tendance > 0.1:
            return "📈 Croissante"
        elif tendance < -0.1:
            return "📉 Décroissante"
        else:
            return "➡️ Stable"
    return "❓ Insuffisante"

def afficher_page_prediction():
    """Affiche la page de prédiction de la production pétrolière avec le modèle et l'historique."""
    st.title("🔮 Prédiction Avancée de la Production Pétrolière")
    st.markdown("Prédisez la production avec des projections temporelles et des analyses de tendances.")
    st.markdown("---")
    
    # Fonction pour charger le modèle
    @st.cache_resource
    def charger_modele():
        """Charge le modèle de prédiction depuis le fichier.""" 
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        else:
            st.error("❌ Modèle non trouvé. Veuillez vérifier le chemin du fichier.")
            return None
    
    # Fonction pour charger les données avec conversion des dates
    @st.cache_data
    def charger_donnees():
        """Charge les données historiques avec conversion robuste des dates."""
        try:
            df = pd.read_csv(HISTORIQUE_PATH)
            
            # Conversion explicite des dates
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'DATEPRD' in col]
            for col in date_cols:
                df[col] = pd.to_datetime(df[col], format='%d-%b-%y', errors='coerce', dayfirst=True)
                
                if df[col].isna().any():
                    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
            
            # Conversion finale en datetime64[ns]
            df[date_cols] = df[date_cols].apply(lambda x: x.astype('datetime64[ns]'))
            df.dropna(subset=date_cols, inplace=True)
            
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {e}")
            return pd.DataFrame()
    
    model = charger_modele()
    df_historique = charger_donnees()
    
    # Sidebar pour les options de prédiction
    st.sidebar.header("⚙️ Options de Prédiction")
    
    # Type de prédiction
    type_prediction = st.sidebar.radio(
        "Type de prédiction:",
        ["🎯 Prédiction Instantanée", "📅 Prédictions Futures", "📊 Analyse Historique"]
    )
    
    if type_prediction == "🎯 Prédiction Instantanée":
        afficher_prediction_instantanee(model)
    elif type_prediction == "📅 Prédictions Futures":
        afficher_predictions_futures(model)
    else:
        afficher_analyse_historique(df_historique)

def afficher_prediction_instantanee(model):
    """Affiche le formulaire de prédiction instantanée."""
    st.subheader("🎯 Prédiction Instantanée")
    
    # Formulaire de saisie
    with st.form("formulaire_prediction_instantanee"):
        st.write("**🔧 Paramètres du puits:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pression = st.number_input("Pression fond (bar)", min_value=0.0, value=300.0, format="%.2f")
            temperature = st.number_input("Température fond (°C)", min_value=0.0, value=90.0, format="%.2f")
            dp_tubing = st.number_input("ΔP tubing", min_value=0.0, value=250.0, format="%.2f")
        
        with col2:
            annulus = st.number_input("Pression annulaire (bar)", min_value=0.0, value=50.0, format="%.2f")
            choke = st.number_input("Taille du choke (inches)", min_value=0.0, value=2.0, format="%.2f")
            whp = st.number_input("Pression tête de puits (bar)", min_value=0.0, value=25.0, format="%.2f")
        
        with col3:
            wht = st.number_input("Température tête de puits (°C)", min_value=0.0, value=15.0, format="%.2f")
            
            # Options avancées
            st.write("**⚙️ Options:**")
            afficher_shap = st.checkbox("Afficher explications SHAP", value=True)
            sauvegarder = st.checkbox("Sauvegarder prédiction", value=True)
        
        submitted = st.form_submit_button("🔮 Prédire Production", use_container_width=True)
    
    if submitted and model:
        executer_prediction_instantanee(model, pression, temperature, dp_tubing, annulus, choke, whp, wht, afficher_shap, sauvegarder)

def executer_prediction_instantanee(model, pression, temperature, dp_tubing, annulus, choke, whp, wht, afficher_shap, sauvegarder):
    """Exécute la prédiction instantanée."""
    input_data = np.array([[pression, temperature, dp_tubing, annulus, choke, whp, wht]])
    
    try:
        prediction = model.predict(input_data)[0]
        
        # Affichage du résultat principal
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🛢️ Production Estimée",
                value=f"{prediction:.2f} m³/jour",
                delta=None
            )
        
        with col2:
            # Classification de la production
            if prediction > 100:
                niveau = "🟢 Élevée"
            elif prediction > 50:
                niveau = "🟡 Moyenne"
            else:
                niveau = "🔴 Faible"
            
            st.metric(
                label="📊 Niveau de Production",
                value=niveau,
                delta=None
            )
        
        with col3:
            # Calcul de rentabilité estimée (exemple)
            rentabilite = prediction * 50  # 50€ par m³ (exemple)
            st.metric(
                label="💰 Revenus Estimés/jour",
                value=f"{rentabilite:.0f} €",
                delta=None
            )
        
        # Analyse de sensibilité
        st.subheader("📈 Analyse de Sensibilité")
        analyser_sensibilite(model, input_data[0])
        
        # Explications SHAP
        if afficher_shap:
            afficher_explications_shap(model, input_data)
        
        # Sauvegarde
        if sauvegarder:
            sauvegarder_prediction(pression, temperature, dp_tubing, annulus, choke, whp, wht, prediction)
            
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")

def afficher_predictions_futures(model):
    """Affiche les prédictions futures avec filtres temporels."""
    st.subheader("📅 Prédictions Futures")
    
    # Paramètres de base
    with st.expander("🔧 Paramètres de Base du Puits", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            pression_base = st.slider("Pression fond (bar)", 0.0, 500.0, 300.0)
            temperature_base = st.slider("Température fond (°C)", 0.0, 200.0, 90.0)
            dp_tubing_base = st.slider("ΔP tubing", 0.0, 400.0, 250.0)
            annulus_base = st.slider("Pression annulaire (bar)", 0.0, 100.0, 50.0)
        
        with col2:
            choke_base = st.slider("Taille du choke (inches)", 0.0, 10.0, 2.0)
            whp_base = st.slider("Pression tête de puits (bar)", 0.0, 100.0, 25.0)
            wht_base = st.slider("Température tête de puits (°C)", 0.0, 50.0, 15.0)
    
    # Options de prédiction temporelle
    col1, col2, col3 = st.columns(3)
    
    with col1:
        periode = st.selectbox(
            "📊 Période de prédiction:",
            ["semaine", "mois", "année"]
        )
    
    with col2:
        nb_periodes = st.slider(
            "🔢 Nombre de périodes:",
            min_value=1, max_value=12, value=6
        )
    
    with col3:
        scenarios = st.multiselect(
            "📈 Scénarios:",
            ["Optimiste (+10%)", "Nominal", "Pessimiste (-10%)"],
            default=["Nominal"]
        )
    
    if st.button("🚀 Générer Prédictions Futures", use_container_width=True):
        generer_graphiques_futures(model, pression_base, temperature_base, dp_tubing_base, 
                                 annulus_base, choke_base, whp_base, wht_base, 
                                 periode, nb_periodes, scenarios)

def generer_graphiques_futures(model, pression, temperature, dp_tubing, annulus, choke, whp, wht, periode, nb_periodes, scenarios):
    """Génère les graphiques de prédictions futures."""
    params_base = np.array([pression, temperature, dp_tubing, annulus, choke, whp, wht])
    
    fig = go.Figure()
    
    # Couleurs pour les scénarios
    couleurs = {"Optimiste (+10%)": "green", "Nominal": "blue", "Pessimiste (-10%)": "red"}
    
    for scenario in scenarios:
        # Modifier les paramètres selon le scénario
        if scenario == "Optimiste (+10%)":
            params_scenario = params_base * 1.1
        elif scenario == "Pessimiste (-10%)":
            params_scenario = params_base * 0.9
        else:
            params_scenario = params_base
        
        # Générer les prédictions
        dates, predictions = generer_predictions_futures(model, params_scenario, periode, nb_periodes)
        
        # Ajouter au graphique
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines+markers',
            name=scenario,
            line=dict(color=couleurs.get(scenario, "blue"), width=3),
            marker=dict(size=8)
        ))
    
    # Configuration du graphique
    fig.update_layout(
        title=f"🔮 Prédictions de Production - {periode.capitalize()}s Futures",
        xaxis_title="Date",
        yaxis_title="Production (m³/jour)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau récapitulatif
    afficher_tableau_recapitulatif(scenarios, dates, periode)

def afficher_tableau_recapitulatif(scenarios, dates, periode):
    """Affiche un tableau récapitulatif des prédictions."""
    st.subheader("📋 Récapitulatif des Prédictions")
    
    # Créer un DataFrame récapitulatif
    recap_data = []
    for i, date in enumerate(dates):
        for scenario in scenarios:
            # Simuler des valeurs pour l'exemple
            if scenario == "Optimiste (+10%)":
                prod = np.random.uniform(80, 120)
            elif scenario == "Pessimiste (-10%)":
                prod = np.random.uniform(40, 80)
            else:
                prod = np.random.uniform(60, 100)
            
            recap_data.append({
                "Période": f"{periode.capitalize()} {i+1}",
                "Date": date.strftime("%Y-%m-%d"),
                "Scénario": scenario,
                "Production (m³/jour)": f"{prod:.1f}",
                "Revenus Estimés (€/jour)": f"{prod * 50:.0f}"
            })
    
    df_recap = pd.DataFrame(recap_data)
    st.dataframe(df_recap, use_container_width=True)

def afficher_analyse_historique(df_historique):
    """Affiche l'analyse des données historiques."""
    st.subheader("📊 Analyse des Données Historiques")
    
    if df_historique.empty:
        st.warning("⚠️ Aucune donnée historique disponible.")
        return
    
    # Filtres temporels
    col1, col2, col3 = st.columns(3)
    
    with col1:
        periode_analyse = st.selectbox(
            "📅 Grouper par:",
            ["Jour", "Semaine", "Mois", "Trimestre", "Année"]
        )
    
    with col2:
        date_debut = st.date_input(
            "📅 Date de début:",
            value=df_historique['DATEPRD'].min() if 'DATEPRD' in df_historique.columns else datetime.date.today() - datetime.timedelta(days=90)
        )
    
    with col3:
        date_fin = st.date_input(
            "📅 Date de fin:",
            value=df_historique['DATEPRD'].max() if 'DATEPRD' in df_historique.columns else datetime.date.today()
        )
    
    # Traitement des données historiques
    if 'DATEPRD' in df_historique.columns and 'Production' in df_historique.columns:
        df_filtre = df_historique[
            (df_historique['DATEPRD'].dt.date >= date_debut) & 
            (df_historique['DATEPRD'].dt.date <= date_fin)
        ]
        
        # Analyse des tendances
        tendance = calculer_tendances(df_filtre, 'Production', 'DATEPRD')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📈 Tendance Générale",
                value=tendance,
                delta=None
            )
        
        with col2:
            production_moyenne = df_filtre['Production'].mean()
            st.metric(
                label="📊 Production Moyenne",
                value=f"{production_moyenne:.1f} m³/jour",
                delta=None
            )
        
        with col3:
            nb_predictions = len(df_filtre)
            st.metric(
                label="🔢 Nombre de Prédictions",
                value=f"{nb_predictions}",
                delta=None
            )
        
        # Graphique historique
        fig = px.line(
            df_filtre, 
            x='DATEPRD', 
            y='Production',
            title="📈 Évolution Historique de la Production",
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Production (m³/jour)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques détaillées
        st.subheader("📊 Statistiques Détaillées")
        stats = df_filtre['Production'].describe()
        st.dataframe(stats.to_frame().T, use_container_width=True)
    
    else:
        st.info("ℹ️ Colonnes 'DATEPRD' et 'Production' nécessaires pour l'analyse historique.")

def analyser_sensibilite(model, params_base):
    """Analyse la sensibilité de la prédiction aux paramètres."""
    noms_params = ["Pression", "Température", "ΔP tubing", "Annulus", "Choke", "WHP", "WHT"]
    sensibilites = []
    
    pred_base = model.predict([params_base])[0]
    
    for i, nom in enumerate(noms_params):
        # Variation de +10%
        params_test = params_base.copy()
        params_test[i] *= 1.1
        pred_test = model.predict([params_test])[0]
        
        sensibilite = ((pred_test - pred_base) / pred_base) * 100
        sensibilites.append(sensibilite)
    
    # Graphique de sensibilité
    fig = px.bar(
        x=noms_params,
        y=sensibilites,
        title="📊 Sensibilité de la Production aux Paramètres (+10%)",
        color=sensibilites,
        color_continuous_scale="RdYlGn"
    )
    
    fig.update_layout(
        xaxis_title="Paramètres",
        yaxis_title="Impact sur Production (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Remplacer la fonction afficher_explications_shap par :
def afficher_explications_shap(model, input_data):
    """Affiche les explications SHAP pour le modèle."""
    with st.expander("🔍 Explications SHAP", expanded=False):
        try:
            feature_names = ["Pression", "Température", "ΔP tubing", "Annulus", "Choke", "WHP", "WHT"]
            
            # Utiliser LinearExplainer pour la régression linéaire
            explainer = shap.LinearExplainer(model, input_data)
            shap_values = explainer.shap_values(input_data)
            
            # Créer un DataFrame pour un affichage plus propre
            df_shap = pd.DataFrame({
                'Caractéristique': feature_names,
                'Valeur': input_data[0],
                'Impact SHAP': shap_values[0]
            })
            
            # Trier par valeur absolue de l'impact
            df_shap['Impact Absolu'] = df_shap['Impact SHAP'].abs()
            df_shap = df_shap.sort_values('Impact Absolu', ascending=False)
            
            # Afficher l'importance des caractéristiques
            st.subheader("Importance des caractéristiques")
            fig = px.bar(
                df_shap,
                x='Impact Absolu',
                y='Caractéristique',
                orientation='h',
                title='Impact des caractéristiques sur la prédiction',
                labels={'Impact Absolu': 'Importance (valeur absolue SHAP)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Afficher les valeurs SHAP individuelles
            st.subheader("Détail de l'impact")
            st.dataframe(
                df_shap[['Caractéristique', 'Valeur', 'Impact SHAP']]
                .sort_values('Impact SHAP', key=abs, ascending=False)
                .style.format({
                    'Valeur': '{:.2f}',
                    'Impact SHAP': '{:.2f}'
                }).bar(subset=['Impact SHAP'], align='mid', color=['#ff7f7f', '#7fbf7f']),
                use_container_width=True
            )
            
            # Explication textuelle
            st.subheader("Interprétation")
            st.markdown("""
            - **Valeur SHAP positive** : Augmente la prédiction de production
            - **Valeur SHAP négative** : Diminue la prédiction de production
            - La taille de la barre représente l'importance de la caractéristique
            """)
            
        except Exception as e:
            st.warning(f"⚠️ Les explications SHAP ne sont pas disponibles : {str(e)}")
            st.info("""
            Cette fonctionnalité peut nécessiter une version spécifique de SHAP.
            Essayez de mettre à jour la bibliothèque avec :
            ```
            pip install --upgrade shap
            ```
            """)

def sauvegarder_prediction(pression, temperature, dp_tubing, annulus, choke, whp, wht, prediction):
    """Sauvegarde la prédiction."""
    try:
        prediction_data = {
            "Date": datetime.datetime.now(),
            "Pression": pression,
            "Température": temperature,
            "ΔP tubing": dp_tubing,
            "Annulus": annulus,
            "Choke": choke,
            "WHP": whp,
            "WHT": wht,
            "Production": prediction
        }
        
        save_prediction(prediction_data, HISTORIQUE_PATH)
        st.success("✅ Prédiction sauvegardée avec succès!")
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la sauvegarde: {e}")