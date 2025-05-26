import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from config.settings import DATA_CONFIG
from utils.helpers import load_data

# Chemin des données
HISTORIQUE_PATH = DATA_CONFIG["prediction_path"]

@st.cache_data
def charger_donnees_petrole():
    """Charge et prépare les données de production pétrolière."""
    if not os.path.exists(HISTORIQUE_PATH):
        st.error(f"❌ Fichier de données non trouvé: {HISTORIQUE_PATH}")
        return pd.DataFrame()
    
    # Chargement des données
    df = pd.read_csv(HISTORIQUE_PATH)
    
    # Conversion des dates
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'DATEPRD' in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
    
    # Nettoyage des données
    # Remplacer les valeurs manquantes par 0 pour les colonnes numériques
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def afficher_visualisations_avancees():
    """Affiche des visualisations avancées pour l'analyse des données pétrolières."""
    st.title("📊 Visualisations Avancées des Données Pétrolières")
    st.markdown("Explorez les relations entre différentes variables et découvrez des insights cachés dans vos données.")
    st.markdown("---")
    
    # Chargement des données
    df = charger_donnees_petrole()
    
    if df.empty:
        st.warning("⚠️ Aucune donnée disponible pour l'analyse.")
        return
    
    # Sélection des colonnes pour l'analyse
    cols_numeriques = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cols_temporelles = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # 1. Graphique interactif de l'évolution des paramètres au fil du temps
    st.subheader("📈 Évolution des paramètres au fil du temps")
    
    # Sélection des paramètres à afficher
    col1, col2 = st.columns([1, 3])
    with col1:
        col_date = st.selectbox("Sélectionner la colonne de date", cols_temporelles, index=0)
        parametres = st.multiselect(
            "Sélectionner les paramètres à visualiser",
            options=[col for col in cols_numeriques if col not in ['ON_STREAM_HRS', 'DP_CHOKE_SIZE']],
            default=["AVG_DOWNHOLE_PRESSURE", "AVG_WHP_P", "BORE_OIL_VOL"][:min(3, len(cols_numeriques))]
        )
    
    with col2:
        if parametres:
            # Préparation des données pour le graphique
            df_plot = df.sort_values(by=col_date)
            df_plot = df_plot.dropna(subset=[col_date] + parametres)
            
            # Création du graphique avec Plotly
            fig = px.line(df_plot, x=col_date, y=parametres, markers=True,
                         title=f"Évolution des paramètres au fil du temps",
                         labels={param: param.replace('_', ' ').title() for param in parametres},
                         height=500)
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Valeur",
                legend_title="Paramètres",
                hovermode="x unified",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Veuillez sélectionner au moins un paramètre à visualiser.")
    
    st.markdown("---")
    
    # 2. Matrice de corrélation interactive
    st.subheader("🔄 Matrice de corrélation interactive")
    
    # Sélection des colonnes pour la matrice de corrélation
    cols_corr = st.multiselect(
        "Sélectionner les colonnes pour la matrice de corrélation",
        options=cols_numeriques,
        default=[col for col in cols_numeriques if 'VOL' in col or 'PRESSURE' in col or 'TEMPERATURE' in col][:min(6, len(cols_numeriques))]
    )
    
    if cols_corr:
        # Calcul de la matrice de corrélation
        corr_matrix = df[cols_corr].corr().round(2)
        
        # Création de la heatmap avec Plotly
        fig = px.imshow(corr_matrix,
                       labels=dict(x="Variables", y="Variables", color="Corrélation"),
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       text_auto=True,
                       color_continuous_scale="RdBu_r",
                       aspect="auto")
        
        fig.update_layout(
            title="Matrice de corrélation entre les variables sélectionnées",
            height=600,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Veuillez sélectionner au moins deux colonnes pour la matrice de corrélation.")
    
    st.markdown("---")
    
    # 3. Analyse de la production par puits
    st.subheader("🛢️ Analyse de la production par puits")
    
    if 'NPD_WELL_BORE_NAME' in df.columns and any(col in df.columns for col in ['BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL']):
        # Sélection du type de production à analyser
        col1, col2 = st.columns([1, 3])
        with col1:
            production_type = st.selectbox(
                "Type de production",
                options=[col for col in df.columns if 'VOL' in col and col in cols_numeriques],
                index=0
            )
            
            aggregation = st.selectbox(
                "Méthode d'agrégation",
                options=["Somme", "Moyenne", "Maximum", "Minimum"],
                index=0
            )
            
            top_n = st.slider("Nombre de puits à afficher", min_value=3, max_value=20, value=10)
        
        with col2:
            # Agrégation des données par puits
            agg_func = {
                "Somme": "sum",
                "Moyenne": "mean",
                "Maximum": "max",
                "Minimum": "min"
            }[aggregation]
            
            df_wells = df.groupby('NPD_WELL_BORE_NAME')[production_type].agg(agg_func).sort_values(ascending=False).head(top_n)
            
            # Création du graphique à barres
            fig = px.bar(
                df_wells,
                x=df_wells.index,
                y=production_type,
                title=f"{aggregation} de {production_type} par puits (Top {top_n})",
                labels={"NPD_WELL_BORE_NAME": "Puits", production_type: production_type.replace('_', ' ').title()},
                color=df_wells.values,
                color_continuous_scale="Viridis",
                height=500
            )
            
            fig.update_layout(
                xaxis_title="Puits",
                yaxis_title=f"{production_type.replace('_', ' ').title()}",
                xaxis_tickangle=-45,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Les données nécessaires pour l'analyse par puits ne sont pas disponibles.")
    
    st.markdown("---")
    
    # 4. Analyse de distribution et détection d'anomalies
    st.subheader("📊 Distribution des paramètres et détection d'anomalies")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        param_dist = st.selectbox(
            "Sélectionner un paramètre pour l'analyse de distribution",
            options=cols_numeriques,
            index=0
        )
        
        outlier_method = st.radio(
            "Méthode de détection d'anomalies",
            options=["IQR", "Z-Score"],
            index=0
        )
    
    with col2:
        # Filtrer les valeurs non nulles
        values = df[param_dist].dropna()
        
        if len(values) > 0:
            # Détection des anomalies
            if outlier_method == "IQR":
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                non_outliers = values[(values >= lower_bound) & (values <= upper_bound)]
            else:  # Z-Score
                z_scores = (values - values.mean()) / values.std()
                outliers = values[abs(z_scores) > 3]
                non_outliers = values[abs(z_scores) <= 3]
            
            # Création de l'histogramme avec Plotly
            fig = make_subplots(rows=2, cols=1, 
                               shared_xaxes=True,
                               vertical_spacing=0.1,
                               subplot_titles=("Histogramme avec détection d'anomalies", "Boxplot"))
            
            # Histogramme
            fig.add_trace(
                go.Histogram(x=non_outliers, name="Valeurs normales", marker_color="blue", opacity=0.7),
                row=1, col=1
            )
            
            if len(outliers) > 0:
                fig.add_trace(
                    go.Histogram(x=outliers, name="Anomalies", marker_color="red", opacity=0.7),
                    row=1, col=1
                )
            
            # Boxplot
            fig.add_trace(
                go.Box(x=values, name=param_dist, marker_color="green"),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f"Distribution de {param_dist.replace('_', ' ').title()}",
                height=600,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Statistiques
            stats_text = f"""
            **Statistiques pour {param_dist.replace('_', ' ').title()}**:
            - **Moyenne**: {values.mean():.2f}
            - **Médiane**: {values.median():.2f}
            - **Écart-type**: {values.std():.2f}
            - **Min**: {values.min():.2f}
            - **Max**: {values.max():.2f}
            - **Nombre d'anomalies détectées**: {len(outliers)} ({len(outliers)/len(values)*100:.2f}%)
            """
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(stats_text)
        else:
            st.warning(f"Pas de données valides pour {param_dist}.")
    
    st.markdown("---")
    
    # 5. Analyse de corrélation entre deux variables
    st.subheader("🔍 Analyse de corrélation entre deux variables")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        var_x = st.selectbox("Variable X", options=cols_numeriques, index=0)
        var_y = st.selectbox("Variable Y", options=cols_numeriques, index=min(1, len(cols_numeriques)-1))
        
        color_var = st.selectbox(
            "Variable de couleur (optionnelle)",
            options=["Aucune"] + cols_numeriques,
            index=0
        )
        
        regression = st.checkbox("Ajouter une ligne de régression", value=True)
    
    with col2:
        # Préparation des données
        scatter_data = df[[var_x, var_y]].dropna()
        
        if not scatter_data.empty:
            # Création du scatter plot
            if color_var != "Aucune":
                scatter_data = scatter_data.join(df[color_var].dropna())
                fig = px.scatter(
                    scatter_data, x=var_x, y=var_y, color=color_var,
                    trendline="ols" if regression else None,
                    labels={var_x: var_x.replace('_', ' ').title(), var_y: var_y.replace('_', ' ').title()},
                    title=f"Relation entre {var_x.replace('_', ' ').title()} et {var_y.replace('_', ' ').title()}",
                    height=500,
                    color_continuous_scale="Viridis"
                )
            else:
                fig = px.scatter(
                    scatter_data, x=var_x, y=var_y,
                    trendline="ols" if regression else None,
                    labels={var_x: var_x.replace('_', ' ').title(), var_y: var_y.replace('_', ' ').title()},
                    title=f"Relation entre {var_x.replace('_', ' ').title()} et {var_y.replace('_', ' ').title()}",
                    height=500
                )
            
            # Calcul du coefficient de corrélation
            corr = scatter_data[[var_x, var_y]].corr().iloc[0, 1]
            
            fig.update_layout(
                annotations=[
                    dict(
                        x=0.5,
                        y=1.05,
                        xref="paper",
                        yref="paper",
                        text=f"Coefficient de corrélation: {corr:.3f}",
                        showarrow=False,
                        font=dict(size=14)
                    )
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interprétation de la corrélation
            if abs(corr) < 0.3:
                correlation_strength = "faible"
            elif abs(corr) < 0.7:
                correlation_strength = "modérée"
            else:
                correlation_strength = "forte"
                
            correlation_direction = "positive" if corr >= 0 else "négative"
            
            st.info(f"Il existe une corrélation {correlation_strength} {correlation_direction} entre {var_x.replace('_', ' ').title()} et {var_y.replace('_', ' ').title()}.")
        else:
            st.warning("Données insuffisantes pour l'analyse de corrélation.")
    
    st.markdown("---")
    
    # 6. Carte de chaleur temporelle
    st.subheader("🗓️ Carte de chaleur temporelle")
    
    if cols_temporelles:
        col1, col2 = st.columns([1, 3])
        with col1:
            date_col = st.selectbox("Colonne de date", options=cols_temporelles, index=0)
            metric_col = st.selectbox("Métrique à visualiser", options=cols_numeriques, index=0)
            
            time_agg = st.selectbox(
                "Agrégation temporelle",
                options=["Jour", "Semaine", "Mois", "Année"],
                index=2
            )
        
        with col2:
            # Préparation des données
            heatmap_data = df[[date_col, metric_col]].dropna().copy()
            
            # Extraction des composantes temporelles
            if time_agg == "Jour":
                heatmap_data['day'] = heatmap_data[date_col].dt.day_name()
                heatmap_data['week'] = heatmap_data[date_col].dt.isocalendar().week
                pivot_data = heatmap_data.pivot_table(index='week', columns='day', values=metric_col, aggfunc='mean')
                title = f"Moyenne de {metric_col} par jour de la semaine"
            elif time_agg == "Semaine":
                heatmap_data['week'] = heatmap_data[date_col].dt.isocalendar().week
                heatmap_data['year'] = heatmap_data[date_col].dt.year
                pivot_data = heatmap_data.pivot_table(index='year', columns='week', values=metric_col, aggfunc='mean')
                title = f"Moyenne de {metric_col} par semaine"
            elif time_agg == "Mois":
                heatmap_data['month'] = heatmap_data[date_col].dt.month_name()
                heatmap_data['year'] = heatmap_data[date_col].dt.year
                pivot_data = heatmap_data.pivot_table(index='year', columns='month', values=metric_col, aggfunc='mean')
                title = f"Moyenne de {metric_col} par mois"
            else:  # Année
                heatmap_data['year'] = heatmap_data[date_col].dt.year
                heatmap_data['quarter'] = heatmap_data[date_col].dt.quarter
                pivot_data = heatmap_data.pivot_table(index='year', columns='quarter', values=metric_col, aggfunc='mean')
                title = f"Moyenne de {metric_col} par trimestre"
            
            # Création de la carte de chaleur
            fig = px.imshow(
                pivot_data,
                labels=dict(x="Période", y="Année", color=f"{metric_col.replace('_', ' ').title()}"),
                title=title,
                color_continuous_scale="Viridis",
                aspect="auto",
                text_auto=True
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucune colonne de date disponible pour l'analyse temporelle.")

    # Ajouter un bouton pour télécharger les données filtrées
    st.markdown("---")
    st.subheader("📥 Exporter les données")
    st.download_button(
        label="Télécharger les données analysées",
        data=df.to_csv(index=False),
        file_name="donnees_petrolieres_analysees.csv",
        mime="text/csv"
    )
