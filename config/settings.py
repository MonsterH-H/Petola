"""
Configuration du projet IA Pétrolière
"""

import os
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).resolve().parent.parent

# Configuration des modèles
MODEL_CONFIG = {
    "random_forest": {
        "path": os.path.join(BASE_DIR, "modeles", "random_forest_model.pkl"),
        "version": "1.0.0",
        "features": ["pression", "temperature", "dp_tubing", "annulus", "choke", "whp", "wht"]
    }
}

# Configuration de l'application
APP_CONFIG = {
    "title": "Système d'IA Pétrolière",
    "icon": "🛢️",
    "layout": "wide",
    "sidebar_state": "expanded",
    "theme": {
        "primary_color": "#4e73df",
        "secondary_color": "#f4f7fb",
        "font_family": "Arial"
    }
}

# Configuration des données
DATA_CONFIG = {
    "prediction_path": os.path.join(BASE_DIR, "data", "prediction.csv"),
    "historical_path": os.path.join(BASE_DIR, "data", "historical.csv")
}

# Configuration du logging
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": os.path.join(BASE_DIR, "logs", "app.log")
}