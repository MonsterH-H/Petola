import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Titre du projet
PROJECT_TITLE = "📊 Suivi des Prédictions de Production"

@st.cache_data
def charger_historique():
    """Charge les données de prédiction depuis un fichier CSV."""
    chemin = "data/prediction.csv"
    if os.path.exists(chemin):
        return pd.read_csv(chemin, parse_dates=["Date"])
    else:
        return pd.DataFrame()

def afficher_dashboard():
    """Affiche le tableau de bord avec les visualisations et les KPIs."""
    st.title(PROJECT_TITLE)
    st.markdown("Visualisez l'évolution de la production prédite et analysez les tendances clés.")

    # Chargement des données
    df = charger_historique()

    # Vérification si les données sont disponibles
    if df.empty:
        st.warning("⚠️ Aucune donnée de prédiction disponible pour le moment.")
        return

    # KPIs principaux : Dernière production, Production moyenne, Production max
    st.markdown("### 🔢 Indicateurs clés de production")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dernière production prédite", f"{df['Production'].iloc[-1]:.2f} m³/jour")
    col2.metric("Production moyenne", f"{df['Production'].mean():.2f} m³/jour")
    col3.metric("Production max", f"{df['Production'].max():.2f} m³/jour")

    # Graphique : Évolution temporelle de la production
    st.markdown("---")
    st.subheader("📊 Évolution temporelle de la production prédite")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x="Date", y="Production", ax=ax, marker="o", linewidth=2, color="teal")
    ax.set_xlabel("Date", fontsize=14, color="darkblue")
    ax.set_ylabel("Production (m³/jour)", fontsize=14, color="darkblue")
    ax.set_title("Tendance de la production prédite", fontsize=16, color="darkred")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Graphique de la production par type de fluide (si disponible dans les données)
    st.markdown("---")
    st.subheader("💧 Production par type de fluide")
    if 'BORE_OIL_VOL' in df.columns and 'BORE_GAS_VOL' in df.columns and 'BORE_WAT_VOL' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        df.groupby('Date')[['BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL']].sum().plot(ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_xlabel("Date", fontsize=14, color="darkblue")
        ax.set_ylabel("Volume (m³)", fontsize=14, color="darkblue")
        ax.set_title("Production par type de fluide au fil du temps", fontsize=16, color="darkred")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("🔧 Les colonnes pour la production par fluide (pétrole, gaz, eau) ne sont pas disponibles.")

    # Matrice de corrélation pour observer les relations entre les variables
    st.markdown("---")
    st.subheader("📊 Intensité des corrélations")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calcul des corrélations
    corr = df.corr(numeric_only=True)
    
    # Création du heatmap avec valeurs absolues pour l'intensité
    # mais affichage des valeurs réelles dans les annotations
    sns.heatmap(
        corr.abs(),  # Valeurs absolues pour l'intensité des couleurs
        annot=corr.round(2),  # Valeurs réelles arrondies à 2 décimales
        cmap="YlOrRd",  # Palette de couleurs séquentielle (du jaune au rouge)
        fmt=".2f",
        ax=ax,
        linewidths=0.8,
        linecolor='white',
        cbar_kws={"shrink": 0.8, "label": "Intensité de la corrélation"},
        vmin=0,  # Échelle de 0 à 1
        vmax=1,
        annot_kws={"size": 10}  # Taille de police des annotations
    )
    
    # Personnalisation du titre et des axes
    ax.set_title("Intensité des corrélations entre les variables\n(valeurs négatives en bleu)", 
                fontsize=16, 
                color='darkblue',
                pad=20)
    
    # Mise en forme des étiquettes
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    # Ajout d'une légende explicative
    st.caption("""
    **Légende :**
    - Les valeurs montrent le coefficient de corrélation (de -1 à 1)
    - Les couleurs indiquent l'intensité de la relation (0 = pas de corrélation, 1 = corrélation parfaite)
    - Les valeurs positives sont affichées en noir, les négatives en bleu
    """)
    
    st.pyplot(fig)

    # Statistiques descriptives détaillées
    st.markdown("---")
    st.subheader("📈 Statistiques descriptives détaillées")
    st.dataframe(df.describe(), use_container_width=True)

    # Vérification de la présence de valeurs manquantes
    st.markdown("---")
    st.subheader("⚠️ Vérification des valeurs manquantes")
    missing_values = df.isnull().sum()
    missing_data = missing_values[missing_values > 0]
    if not missing_data.empty:
        st.write(f"Il y a des valeurs manquantes dans les colonnes suivantes :")
        st.write(missing_data)
    else:
        st.write("Aucune valeur manquante dans les données.")

    # Exportation des données
    st.markdown("---")
    st.subheader("📥 Exporter les données")
    st.write("Vous pouvez exporter les données de prédiction sous forme de fichier CSV.")
    st.download_button(
        label="Télécharger les données",
        data=df.to_csv(index=False),
        file_name="prediction.csv",
        mime="text/csv"
    )
