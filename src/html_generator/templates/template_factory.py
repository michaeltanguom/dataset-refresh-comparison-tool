"""
Template factory for creating template instances
"""

from typing import Dict, List, Any
from .base_template import BaseTemplate
from .research_dashboard import ResearchDashboardTemplate
from ...utils.exceptions import HtmlGenerationError
from ...utils.logging_config import get_logger

logger = get_logger('html_generator.template_factory')


class TemplateFactory:
    """Factory for creating template instances based on configuration"""
    
    _templates = {
        'research_dashboard': ResearchDashboardTemplate,
        # Add more templates here as you create them
        # 'comparison_summary': ComparisonSummaryTemplate,
    }
    
    @classmethod
    def create_template(cls, template_name: str, config: Dict[str, Any]) -> BaseTemplate:
        """
        Create template instance
        
        Args:
            template_name: Name of template to create
            config: Template configuration
            
        Returns:
            Template instance
        """
        if template_name not in cls._templates:
            available_templates = list(cls._templates.keys())
            raise HtmlGenerationError(
                f"Unknown template: {template_name}. Available templates: {available_templates}"
            )
        
        template_class = cls._templates[template_name]
        logger.info(f"Creating template: {template_name}")
        
        return template_class(config)
    
    @classmethod
    def get_available_templates(cls) -> List[str]:
        """Get list of available template names"""
        return list(cls._templates.keys())
    
    @classmethod
    def register_template(cls, name: str, template_class: type) -> None:
        """
        Register a new template class
        
        Args:
            name: Template name
            template_class: Template class (must inherit from BaseTemplate)
        """
        if not issubclass(template_class, BaseTemplate):
            raise HtmlGenerationError(f"Template class must inherit from BaseTemplate")
        
        cls._templates[name] = template_class
        logger.info(f"Registered new template: {name}")