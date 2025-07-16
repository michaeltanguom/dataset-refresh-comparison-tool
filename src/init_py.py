"""
Dataset Comparison Pipeline Package
Initialisation file to make src a proper Python package
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Dataset Comparison Pipeline Team"
__description__ = "EtLT pipeline for dataset comparison with Prefect orchestration"

# Import key classes for easier access
try:
    from .pipeline_orchestrator import PipelineOrchestrator
    from .config.config_manager import ConfigManager
    from .config.config_validator import ConfigValidator
    from .utils.exceptions import (
        ConfigurationError, 
        ExtractionError, 
        NormalisationError,
        DatabaseError, 
        DataQualityError, 
        ComparisonError, 
        PipelineError
    )
    from .utils.logging_config import setup_logging, get_logger
    from .utils.database_manager import DatabaseManager
    
    # Make key components available at package level
    __all__ = [
        'PipelineOrchestrator',
        'ConfigManager', 
        'ConfigValidator',
        'DatabaseManager',
        'setup_logging',
        'get_logger',
        'ConfigurationError',
        'ExtractionError', 
        'NormalisationError',
        'DatabaseError',
        'DataQualityError',
        'ComparisonError',
        'PipelineError'
    ]
    
except ImportError as e:
    # Graceful handling of import errors during development
    import warnings
    warnings.warn(f"Some components could not be imported: {e}", ImportWarning)
    __all__ = []

# Package configuration
PACKAGE_INFO = {
    'name': 'dataset-comparison-pipeline',
    'version': __version__,
    'description': __description__,
    'components': {
        'config': 'Configuration management and validation',
        'extract': 'Data extraction and light transformation',
        'load': 'Database loading operations', 
        'transform': 'Data cleaning and comparison',
        'utils': 'Shared utilities and exceptions'
    }
}
