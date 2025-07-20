"""
Asset Manager
Handles CSS, JavaScript, and other asset management for HTML generation
"""

from typing import Dict, Any
from ...utils.logging_config import get_logger

logger = get_logger('html_generator.asset_manager')


class AssetManager:
    """
    Manages CSS, JavaScript, and other assets for HTML generation
    Future: Could handle external CSS/JS files, image optimization, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise asset manager
        
        Args:
            config: Asset configuration
        """
        self.config = config
        self.styling_config = config.get('styling', {})
        
    def get_custom_css(self) -> str:
        """Get custom CSS if configured"""
        custom_css_path = self.styling_config.get('custom_css_path')
        
        if custom_css_path:
            try:
                with open(custom_css_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Could not load custom CSS from {custom_css_path}: {e}")
        
        return ""
    
    def get_theme_variables(self) -> Dict[str, str]:
        """Get CSS theme variables based on configuration"""
        theme = self.styling_config.get('default_theme', 'modern_blue')
        dark_mode = self.styling_config.get('enable_dark_mode', False)
        
        themes = {
            'modern_blue': {
                'primary_color': '#667eea',
                'secondary_color': '#764ba2',
                'accent_color': '#f8f9ff',
                'text_color': '#333',
                'background_color': '#f5f7fa'
            },
            'modern_green': {
                'primary_color': '#4ade80',
                'secondary_color': '#22c55e',
                'accent_color': '#f0fdf4',
                'text_color': '#333',
                'background_color': '#f9fafb'
            }
        }
        
        base_theme = themes.get(theme, themes['modern_blue'])
        
        if dark_mode:
            base_theme.update({
                'text_color': '#e5e7eb',
                'background_color': '#1f2937',
                'accent_color': '#374151'
            })
        
        return base_theme