"""
Application principale pour l'optimisation de la production pétrolière par IA.

Ce module gère la configuration de l'application, le routage des pages
et l'application des styles personnalisés.
"""

import sys
import logging
from typing import Dict, Any, Callable, List, Optional
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

# Configuration et utilitaires
from config.settings import APP_CONFIG
from config.theme import get_theme_config
from utils.helpers import setup_logging

# Import des vues
from modules.prediction import afficher_page_prediction
from modules.dashboard import afficher_dashboard
from modules.apropos import afficher_apropos
from modules.visualisations import afficher_visualisations_avancees

# Configuration du logging
setup_logging()
logger = logging.getLogger(__name__)

# Type pour les pages de l'application
PageType = Dict[str, Any]

class OilProductionApp:
    """Classe principale de l'application d'optimisation pétrolière."""
    
    def __init__(self):
        """Initialise l'application avec la configuration de base."""
        self.config = self._load_config()
        self.pages = self._initialize_pages()
        self._setup_page_config()
        self._apply_custom_theme()
    
    def _load_config(self) -> Dict[str, Any]:
        """Charge et valide la configuration de l'application."""
        required_keys = ["title", "icon", "layout", "sidebar_state"]
        if not all(key in APP_CONFIG for key in required_keys):
            logger.error("Configuration manquante. Vérifiez config/settings.py")
            sys.exit(1)
        return APP_CONFIG
    
    def _initialize_pages(self) -> List[PageType]:
        """Initialise la liste des pages disponibles dans l'application."""
        return [
            {
                "title": "🏠 Tableau de bord",
                "function": afficher_dashboard,
                "description": "Vue d'ensemble des indicateurs clés"
            },
            {
                "title": "📈 Prédiction de la production",
                "function": afficher_page_prediction,
                "description": "Effectuez des prédictions de production"
            },
            {
                "title": "📊 Visualisations avancées",
                "function": afficher_visualisations_avancees,
                "description": "Explorez les données avec des graphiques interactifs"
            },
            {
                "title": "ℹ️ À propos",
                "function": afficher_apropos,
                "description": "En savoir plus sur cette application"
            },
            {
                "title": "🏡 Accueil",
                "function": self._show_home,
                "description": "Page d'accueil de l'application"
            }
        ]
    
    def _setup_page_config(self) -> None:
        """Configure les paramètres de base de la page Streamlit."""
        st.set_page_config(
            page_title=self.config["title"],
            page_icon=self.config["icon"],
            layout=self.config["layout"],
            initial_sidebar_state=self.config["sidebar_state"]
        )
    
    def _apply_custom_theme(self) -> None:
        """Applique le thème personnalisé à l'application."""
        theme = get_theme_config()
        custom_css = self._generate_custom_css(theme)
        st.markdown(custom_css, unsafe_allow_html=True)
    
    def _generate_custom_css(self, theme: Dict[str, Any]) -> str:
        """Génère le CSS personnalisé à partir de la configuration du thème."""
        return f"""
        <style>
            /* Styles de base */
            .stApp {{
                background-color: #f8f9fc;
                font-family: {theme['font_family']};
                line-height: 1.6;
            }}
            
            /* En-tête */
            .main .block-container {{
                padding-top: 2rem;
                padding-bottom: 2rem;
            }}
            
            /* Titres */
            h1, h2, h3, h4, h5, h6 {{
                color: #1a365d;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
            }}
            
            /* Boutons */
            .stButton>button {{
                background-color: {theme['primary_color']};
                color: white;
                border-radius: {theme['button_styles']['border_radius']};
                padding: {theme['button_styles']['padding']};
                transition: {theme['button_styles']['transition']};
                font-weight: 500;
                border: none;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            .stButton>button:hover {{
                background-color: #2e59d9;
                transform: scale({theme['button_styles']['hover_scale']});
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }}
            
            .stButton>button:active {{
                transform: scale({theme['button_styles']['active_scale']});
            }}
            
            /* Cartes et conteneurs */
            .stCard {{
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                background: white;
            }}
            
            /* Sidebar */
            .css-1d391kg, .css-1oe5cao {{
                padding: 2rem 1.5rem;
                background-color: #f8fafc;
                border-right: 1px solid #e2e8f0;
            }}
            
            /* Animations */
            {theme['animations']['fade_in']}
            {theme['animations']['slide_up']}
            {theme['animations']['pulse']}
            
            /* Transitions globales */
            * {{
                transition: all {theme['transition_speed']} {theme['animation_timing']};
            }}
            
            /* Personnalisation des onglets */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                padding: 10px 20px;
                border-radius: 4px;
                margin: 0 2px;
            }}
            
            .stTabs [aria-selected="true"] {{
                background-color: {theme['primary_color']};
                color: white;
            }}
        </style>
        """
    
    def _show_home(self) -> None:
        """Affiche la page d'accueil de l'application."""
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='color: #1a365d;'>🛢️ Système d'IA pour l'Optimisation de la Production Pétrolière</h1>
            <p style='font-size: 1.2rem; color: #4a5568;'>
                Optimisez votre production pétrolière grâce à la puissance de l'intelligence artificielle
            </p>
        </div>
        
        <div class='stCard'>
            <h2>Bienvenue sur notre plateforme d'optimisation</h2>
            <p>Cette application vous permet de :</p>
            <ul>
                <li>Prédire la production pétrolière avec précision</li>
                <li>Suivre les indicateurs clés de performance en temps réel</li>
                <li>Explorer des visualisations interactives de vos données</li>
                <li>Prendre des décisions éclairées basées sur l'analyse prédictive</li>
            </ul>
            
            <div style='margin-top: 2rem;'>
                <p>Commencez par sélectionner une section dans le menu de navigation à gauche.</p>
            </div>
        </div>
        
        <div class='stCard'>
            <h3>📊 Tableau de bord</h3>
            <p>Visualisez les indicateurs clés et les tendances de production.</p>
            
            <h3>📈 Prédiction de la production</h3>
            <p>Effectuez des prédictions basées sur différents scénarios opérationnels.</p>
            
            <h3>📊 Visualisations avancées</h3>
            <p>Explorez vos données avec des graphiques interactifs et des analyses détaillées.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_navigation(self) -> str:
        """Affiche la barre de navigation et retourne la page sélectionnée."""
        st.sidebar.title("🛠️ Menu de Navigation")
        
        # Afficher les options de navigation
        page_titles = [page["title"] for page in self.pages]
        selected_page = st.sidebar.radio(
            "Sélectionner une section",
            options=page_titles,
            format_func=lambda x: f"{x}",
            index=0
        )
        
        # Ajouter des informations supplémentaires dans la sidebar
        st.sidebar.markdown("---")
        st.sidebar.info(
            "ℹ️ Utilisez les différentes sections pour explorer les fonctionnalités de l'application."
        )
        
        return selected_page
    
    def run(self) -> None:
        """Exécute l'application."""
        try:
            # Afficher le titre principal
            st.markdown(
                f"<h1 style='text-align: center; margin-bottom: 1.5rem;'>"
                f"🛢️ {self.config['title']}</h1>",
                unsafe_allow_html=True
            )
            
            # Afficher la navigation et récupérer la page sélectionnée
            selected_page_title = self._render_navigation()
            
            # Trouver et exécuter la fonction de la page sélectionnée
            for page in self.pages:
                if page["title"] == selected_page_title:
                    page["function"]()
                    break
            
        except Exception as e:
            logger.error(f"Erreur inattendue: {str(e)}", exc_info=True)
            st.error(
                "⚠️ Une erreur inattendue s'est produite. "
                "Veuillez réessayer ou contacter le support technique."
            )
            st.exception(e)  # Afficher les détails de l'erreur en mode développement

# Point d'entrée de l'application
if __name__ == "__main__":
    app = OilProductionApp()
    app.run()
