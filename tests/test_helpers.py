"""
Tests pour les utilitaires de l'application IA Pétrolière
"""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
import os

from utils.helpers import load_data, save_prediction, format_production

# Setup
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_FILE = TEST_DATA_DIR / "test_predictions.csv"

@pytest.fixture
def clean_test_file():
    """Fixture pour nettoyer le fichier de test avant/après chaque test."""
    if TEST_FILE.exists():
        TEST_FILE.unlink()
    yield
    if TEST_FILE.exists():
        TEST_FILE.unlink()


def test_load_data_empty_file(clean_test_file):
    """Teste le chargement d'un fichier vide."""
    TEST_FILE.touch()
    df = load_data(str(TEST_FILE))
    assert df.empty


def test_save_prediction(clean_test_file):
    """Teste la sauvegarde d'une prédiction."""
    test_data = {
        "Date": datetime.now(),
        "Pression": 100.5,
        "Température": 45.2,
        "ΔP tubing": 10.3,
        "Annulus": 50.1,
        "Choke": 2.5,
        "WHP": 30.2,
        "WHT": 40.1,
        "Production": 1500.75
    }
    
    save_prediction(test_data, str(TEST_FILE))
    assert TEST_FILE.exists()
    
    df = pd.read_csv(TEST_FILE)
    assert not df.empty
    assert df.shape[0] == 1


def test_format_production():
    """Teste le formatage des valeurs de production."""
    assert format_production(1500.75) == "1,500.75 m³/jour"
    assert format_production(0) == "0.00 m³/jour"
    assert format_production(1234567.89) == "1,234,567.89 m³/jour"