"""
HTML Renderer
Handles final HTML generation, file saving, and output management
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ...utils.logging_config import get_logger
from ...utils.exceptions import HtmlGenerationError
from ...utils.common import create_directory_if_not_exists, normalise_text

logger = get_logger('html_generator.renderer')


class HtmlRenderer:
    """
    Handles HTML file generation and output management
    Single responsibility: File output and asset management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise HTML renderer
        
        Args:
            config: HTML generation configuration
        """
        self.config = config
        self.output_config = config.get('output', {})
        self.logger = get_logger('html_renderer')
        
    def save_html(self, html_content: str, dataset_key: str, template_name: str) -> str:
        """
        Save HTML content to file
        
        Args:
            html_content: Complete HTML document
            dataset_key: Dataset identifier (e.g., "agricultural_sciences_highly_cited_only")
            template_name: Template name used
            
        Returns:
            Path to saved HTML file
        """
        try:
            # Generate output filename
            filename = self._generate_filename(dataset_key, template_name)
            
            # Ensure output directory exists
            output_dir = self._get_output_directory()
            create_directory_if_not_exists(output_dir)
            
            # Full file path
            file_path = Path(output_dir) / filename
            
            # Save HTML content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Saved HTML dashboard: {file_path}")
            self.logger.info(f"File size: {len(html_content):,} characters")
            
            return str(file_path)
            
        except Exception as e:
            raise HtmlGenerationError(f"Failed to save HTML file for {dataset_key}: {e}")
    
    def _generate_filename(self, dataset_key: str, template_name: str) -> str:
        """Generate output filename based on configuration"""
        naming_pattern = self.output_config.get('file_naming', '{dataset_type}_{template_name}_dashboard.html')
        
        # Extract dataset type from key
        dataset_type = self._extract_dataset_type(dataset_key)
        
        # Generate base filename
        filename = naming_pattern.format(
            dataset_type=dataset_type,
            template_name=template_name
        )
        
        # Add timestamp if enabled
        if self.output_config.get('include_timestamp', True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{timestamp}{ext}"
        
        return filename
    
    def _extract_dataset_type(self, dataset_key: str) -> str:
        """Extract clean dataset type from dataset key"""
        # Remove common suffixes
        suffixes_to_remove = ['_highly_cited_only', '_incites_researchers', '_comparison_report']
        
        clean_key = dataset_key
        for suffix in suffixes_to_remove:
            if clean_key.endswith(suffix):
                clean_key = clean_key[:-len(suffix)]
                break
        
        return clean_key
    
    def _get_output_directory(self) -> str:
        """Get output directory path"""
        base_dir = self.config.get('output_directory', 'html_reports')
        
        # Create timestamped subdirectory if configured
        if self.output_config.get('create_timestamped_folders', False):
            timestamp = datetime.now().strftime('%Y%m%d')
            return str(Path(base_dir) / timestamp)
        
        return base_dir
    
    def get_output_summary(self, saved_files: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate summary of output files
        
        Args:
            saved_files: Dictionary of dataset_key -> file_path
            
        Returns:
            Summary dictionary
        """
        total_size = 0
        
        for file_path in saved_files.values():
            try:
                if Path(file_path).exists():
                    total_size += Path(file_path).stat().st_size
            except Exception:
                pass
        
        return {
            'files_generated': len(saved_files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'output_directory': self._get_output_directory(),
            'files': saved_files
        }