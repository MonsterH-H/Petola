"""
Utils pour l'application IA Pétrolière
"""

import logging
import os
from datetime import datetime
import pandas as pd

# Configuration du logger
logger = logging.getLogger(__name__)


def setup_logging():
    """Configure le logging selon les paramètres du projet."""
    from config.settings import LOG_CONFIG
    
    logging.basicConfig(
        level=LOG_CONFIG["level"],
        format=LOG_CONFIG["format"],
        filename=LOG_CONFIG["file"]
    )


def load_data(file_path):
    """
    Charge les données depuis un fichier CSV avec gestion des erreurs.
    
    Args:
        file_path (str): Chemin vers le fichier CSV
        
    Returns:
        pd.DataFrame: DataFrame contenant les données
    """
    try:
        df = pd.read_csv(file_path, parse_dates=["Date"])
        logger.info(f"Données chargées depuis {file_path}")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()


def save_prediction(data, file_path):
    """
    Sauvegarde une prédiction dans un fichier CSV.
    
    Args:
        data (dict): Dictionnaire contenant les données à sauvegarder
        file_path (str): Chemin vers le fichier de destination
    """
    try:
        df = pd.DataFrame([data])
        
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            existing_df = pd.read_csv(file_path)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.to_csv(file_path, index=False)
            
        logger.info(f"Prédiction sauvegardée dans {file_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")


def format_production(value):
    """Formate une valeur de production pour l'affichage."""
    return f"{value:,.2f} m³/jour".replace(".", " ").replace(",", ".").replace(" ", ",")