"""
Abstract base template for HTML generation
Provides the foundation for all dashboard templates
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ...utils.logging_config import get_logger
from ...utils.exceptions import HtmlGenerationError

logger = get_logger('html_generator.templates')


@dataclass
class TemplateConfig:
    """Configuration for template rendering"""
    title_format: str
    colour_scheme: str
    show_cross_field_analysis: bool = True
    max_researchers_summary: int = 10
    enable_filtering: bool = True
    enable_sorting: bool = True


class BaseTemplate(ABC):
    """
    Abstract base class for all HTML templates
    Defines the interface and common functionality
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise template with configuration
        
        Args:
            config: Template configuration dictionary
        """
        self.config = TemplateConfig(**config)
        self.logger = get_logger(f'html_generator.templates.{self.__class__.__name__}')
        
        # Validate configuration
        self._validate_config()
        
    @abstractmethod
    def generate_html(self, data: Dict[str, Any]) -> str:
        """
        Generate HTML content from processed data
        
        Args:
            data: Processed data for rendering
            
        Returns:
            Complete HTML document as string
        """
        pass
        
    @abstractmethod
    def get_required_data_fields(self) -> List[str]:
        """
        Return list of required data fields for this template
        
        Returns:
            List of required field names
        """
        pass
    
    def _validate_config(self) -> None:
        """Validate template configuration"""
        required_fields = ['title_format', 'colour_scheme']
        
        for field in required_fields:
            if not hasattr(self.config, field):
                raise HtmlGenerationError(f"Missing required config field: {field}")
                
        self.logger.info(f"Template configuration validated: {self.config.title_format}")
    
    def _validate_data(self, data: Dict[str, Any]) -> None:
        """
        Validate that required data fields are present
        
        Args:
            data: Data to validate
        """
        required_fields = self.get_required_data_fields()
        missing_fields = []
        
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
        
        if missing_fields:
            raise HtmlGenerationError(
                f"Missing required data fields for {self.__class__.__name__}: {missing_fields}"
            )
    
    def _format_title(self, dataset_type: str) -> str:
        """
        Format template title with dataset information
        
        Args:
            dataset_type: Type of dataset being processed
            
        Returns:
            Formatted title string
        """
        return self.config.title_format.format(
            dataset_type=dataset_type.replace('_', ' ').title()
        )
    
    def _get_css_classes(self) -> Dict[str, str]:
        """
        Get CSS classes based on colour scheme
        
        Returns:
            Dictionary of CSS class mappings
        """
        colour_schemes = {
            'blue_gradient': {
                'header': 'bg-gradient-blue',
                'primary': 'text-blue-600',
                'secondary': 'text-blue-400',
                'accent': 'bg-blue-100'
            },
            'green_gradient': {
                'header': 'bg-gradient-green',
                'primary': 'text-green-600',
                'secondary': 'text-green-400',
                'accent': 'bg-green-100'
            }
        }
        
        return colour_schemes.get(self.config.colour_scheme, colour_schemes['blue_gradient'])
    
    def _generate_metadata(self, dataset_type: str) -> Dict[str, str]:
        """
        Generate metadata for HTML document
        
        Args:
            dataset_type: Type of dataset
            
        Returns:
            Metadata dictionary
        """
        return {
            'generated_at': datetime.now().isoformat(),
            'dataset_type': dataset_type,
            'template_name': self.__class__.__name__,
            'title': self._format_title(dataset_type)
        }