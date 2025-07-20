"""
HTML Templates Module
Provides template classes for generating different types of dashboards
"""

from .base_template import BaseTemplate, TemplateConfig
from .research_dashboard import ResearchDashboardTemplate
from .template_factory import TemplateFactory

__all__ = [
    'BaseTemplate',
    'TemplateConfig', 
    'ResearchDashboardTemplate',
    'TemplateFactory'
]