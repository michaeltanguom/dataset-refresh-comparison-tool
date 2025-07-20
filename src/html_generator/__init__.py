"""
HTML Generator Module
Generates interactive HTML dashboards from JSON comparison reports
"""

from .templates import TemplateFactory, BaseTemplate
from .renderers import HtmlRenderer, AssetManager

__version__ = "1.0.0"

__all__ = [
    'TemplateFactory',
    'BaseTemplate', 
    'HtmlRenderer',
    'AssetManager'
]