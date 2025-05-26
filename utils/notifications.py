"""
Système de notifications pour l'application IA Pétrolière
"""

import streamlit as st
from typing import Literal
from config.theme import get_theme_config

NotificationType = Literal["success", "error", "warning", "info"]

def show_notification(
    message: str,
    notification_type: NotificationType = "info",
    duration: int = 5,
    closable: bool = True
) -> None:
    """
    Affiche une notification stylisée
    
    Args:
        message: Le message à afficher
        notification_type: Le type de notification (success, error, warning, info)
        duration: Durée d'affichage en secondes (0 pour permanent)
        closable: Si la notification peut être fermée manuellement
    """
    theme = get_theme_config()
    styles = theme["notification_styles"][notification_type]
    
    # HTML/CSS pour la notification
    notification_html = f"""
    <div class="notification {notification_type}" 
         style="
             position: fixed;
             bottom: 20px;
             right: 20px;
             padding: 15px;
             border-radius: {theme['border_radius']};
             background-color: {styles['background']};
             color: {styles['text']};
             box-shadow: {theme['box_shadow_sm']};
             z-index: 1000;
             transition: all {theme['transition_speed']} ease;
             opacity: 0;
             transform: translateY(20px);
             animation: fadeInSlideUp 0.5s ease-out forwards;
         ">
        {message}
        {"<span style='margin-left: 10px; cursor: pointer;' onclick='this.parentElement.remove()'>×</span>" if closable else ""}
    </div>
    
    <style>
    @keyframes fadeInSlideUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes fadeOutSlideDown {{
        from {{ opacity: 1; transform: translateY(0); }}
        to {{ opacity: 0; transform: translateY(20px); }}
    }}
    
    .notification {{
        animation-delay: 0.1s;
    }}
    
    .notification.fade-out {{
        animation: fadeOutSlideDown 0.5s ease-out forwards;
    }}
    </style>
    """
    
    # Afficher la notification
    notification = st.markdown(notification_html, unsafe_allow_html=True)
    
    # Gestion de la durée
    if duration > 0:
        import time
        time.sleep(duration)
        notification.empty()