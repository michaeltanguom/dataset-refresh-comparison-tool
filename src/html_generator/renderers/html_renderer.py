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
            Path(output_dir).mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
            
            # Full file path
            file_path = Path(output_dir) / filename
            
            # Save HTML content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"💾 Saved HTML dashboard: {file_path}")
            self.logger.info(f"📊 File size: {len(html_content):,} characters")
            self.logger.info(f"📂 Output directory: {output_dir}")
            
            return str(file_path)
            
        except Exception as e:
            raise HtmlGenerationError(f"Failed to save HTML file for {dataset_key}: {e}")
    
    def _generate_filename(self, dataset_key: str, template_name: str) -> str:
        """Generate output filename based on configuration - updated for unified dashboards"""
        naming_pattern = self.output_config.get('file_naming', '{dataset_type}_{template_name}_dashboard.html')
        
        # For unified dashboards, dataset_key might be something like "unified_highly_cited_only_25_subjects"
        if dataset_key.startswith('unified_'):
            # Extract the actual dataset type
            parts = dataset_key.split('_')
            if len(parts) >= 3:
                # unified_highly_cited_only_25_subjects -> highly_cited_only
                dataset_type = '_'.join(parts[1:-2]) if 'subjects' in parts[-1] else '_'.join(parts[1:])
            else:
                dataset_type = dataset_key.replace('unified_', '')
        else:
            # Fallback to original logic
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
        """Extract clean dataset type from dataset key - updated for unified dashboards"""
        # Handle unified dashboard keys
        if dataset_key.startswith('unified_'):
            clean_key = dataset_key.replace('unified_', '')
            # Remove subject count info if present
            if '_subjects' in clean_key:
                clean_key = clean_key.split('_subjects')[0]
                # Remove the number before subjects
                parts = clean_key.split('_')
                if parts[-1].isdigit():
                    clean_key = '_'.join(parts[:-1])
        else:
            # Original logic for individual dashboards
            clean_key = dataset_key
        
        # Remove common suffixes
        suffixes_to_remove = ['_highly_cited_only', '_incites_researchers', '_comparison_report']
        
        for suffix in suffixes_to_remove:
            if clean_key.endswith(suffix):
                clean_key = clean_key[:-len(suffix)]
                break
        
        return clean_key
    
    def _get_output_directory(self) -> str:
        """Get output directory path - defaults to html_dashboards in project root"""
        
        # Get configured output directory or use default
        base_dir = self.config.get('output_directory', 'html_dashboards')
        
        # 🔧 FIX: Ensure it's relative to project root, not current working directory
        if not Path(base_dir).is_absolute():
            # Get project root (where comparison_reports folder exists)
            project_root = self._find_project_root()
            base_dir = str(project_root / base_dir)
        
        # Create timestamped subdirectory if configured
        if self.output_config.get('create_timestamped_folders', False):
            timestamp = datetime.now().strftime('%Y%m%d')
            return str(Path(base_dir) / timestamp)
        
        return base_dir

    def _find_project_root(self) -> Path:
        """Find project root directory (where comparison_reports exists)"""
        # Start from current file location and work up
        current_path = Path(__file__).parent
        
        # Look for comparison_reports folder to identify project root
        for parent in [current_path] + list(current_path.parents):
            if (parent / 'comparison_reports').exists():
                self.logger.debug(f"Found project root: {parent}")
                return parent
        
        # Fallback to current working directory
        cwd = Path.cwd()
        self.logger.warning(f"Could not find project root, using current directory: {cwd}")
        return cwd
    
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