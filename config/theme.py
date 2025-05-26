"""
Configuration du thème UI/UX pour l'application IA Pétrolière
"""

from typing import Dict, Any

def get_theme_config() -> Dict[str, Any]:
    """
    Retourne la configuration du thème pour l'application
    """
    return {
        # Couleurs principales
        "primary_color": "#4e73df",
        "secondary_color": "#1cc88a",
        "danger_color": "#e74a3b",
        "warning_color": "#f6c23e",
        "info_color": "#36b9cc",
        
        # Typographie
        "font_family": "'Arial', sans-serif",
        "font_size_base": "1rem",
        "line_height_base": 1.5,
        
        # Espacements
        "spacing_unit": "1rem",
        "border_radius": "0.35rem",
        
        # Animations
        "transition_speed": "0.3s",
        "animation_duration": "0.5s",
        "animation_timing": "cubic-bezier(0.4, 0, 0.2, 1)",
        
        # Ombres
        "box_shadow": "0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15)",
        "box_shadow_sm": "0 0.125rem 0.25rem 0 rgba(58, 59, 69, 0.2)",
        "box_shadow_lg": "0 0.5rem 1rem rgba(0, 0, 0, 0.15)",
        
        # Tailles
        "sidebar_width": "14rem",
        "sidebar_collapsed_width": "4rem",
        
        # Styles spécifiques aux composants
        "button_styles": {
            "padding": "0.375rem 0.75rem",
            "border_radius": "0.35rem",
            "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
            "hover_scale": "1.05",
            "active_scale": "0.98"
        },
        
        # Styles pour les notifications
        "notification_styles": {
            "success": {"background": "#1cc88a", "text": "#fff", "icon": "✓"},
            "error": {"background": "#e74a3b", "text": "#fff", "icon": "✗"},
            "warning": {"background": "#f6c23e", "text": "#000", "icon": "⚠"},
            "info": {"background": "#36b9cc", "text": "#fff", "icon": "ℹ"}
        },
        
        # Animations prédéfinies
        "animations": {
            "fade_in": "@keyframes fadeIn {from {opacity: 0;} to {opacity: 1;}}",
            "slide_up": "@keyframes slideUp {from {transform: translateY(20px); opacity: 0;} to {transform: translateY(0); opacity: 1;}}",
            "pulse": "@keyframes pulse {0% {transform: scale(1);} 50% {transform: scale(1.05);} 100% {transform: scale(1);}}"
        }
    }


def get_custom_css() -> str:
    """
    Retourne le CSS personnalisé pour l'application
    """
    theme = get_theme_config()
    return f"""
    <style>
    /* Styles globaux */
    body {{
        font-family: {theme['font_family']};
        font-size: {theme['font_size_base']};
        line-height: {theme['line_height_base']};
        transition: all {theme['transition_speed']} ease;
    }}
    
    /* Boutons */
    .stButton>button {{
        padding: {theme['button_styles']['padding']};
        border-radius: {theme['button_styles']['border_radius']};
        transition: {theme['button_styles']['transition']};
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    .fade-in {{
        animation: fadeIn {theme['animation_duration']} ease-out;
    }}
    
    /* Responsive design */
    @media (max-width: 768px) {{
        .main {{
            padding: 0.5rem;
        }}
        
        .sidebar {{
            width: 100%;
        }}
    }}
    </style>
    """