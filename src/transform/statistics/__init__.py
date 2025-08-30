# src/transform/statistics/__init__.py
"""
Statistical analysis module for research performance data transformation
Provides SQL-based statistical enhancements including Z-scores, percentiles, and outlier detection
"""

from .statistical_analyser import StatisticalAnalyser

__version__ = "1.0.0"
__description__ = "SQL-based statistical analysis for research performance data"

__all__ = [
    'StatisticalAnalyser'
]

# Module configuration
STATISTICAL_MODULE_INFO = {
    'name': 'transform.statistics',
    'version': __version__,
    'description': __description__,
    'components': {
        'StatisticalAnalyser': 'Main SQL-based statistical analysis engine'
    },
    'supported_methods': ['iqr', 'zscore'],
    'supported_fields': [
        'times_cited',
        'highly_cited_papers', 
        'hot_papers',
        'indicative_cross_field_score'
    ]
}