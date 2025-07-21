"""
Extract and Light Transform for Dataset Comparison Pipeline
Handles data extraction from various sources and normalisation of column names
"""

import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

from ..config.config_manager import ConfigManager
from ..utils.exceptions import ExtractionError, NormalisationError, ValidationError
from ..utils.logging_config import get_logger
from ..utils.common import (
    normalise_text, 
    generate_timestamp, 
    clean_dataframe_columns, 
    get_sample_values,
    log_dataframe_info
)

logger = get_logger('extract')


@dataclass
class DatasetMetadata:
    """Metadata about a processed dataset"""
    source_file: str
    subject: str
    period: str
    sheet_name: str
    normalised_sheet_name: str
    table_name: str
    row_count: int
    columns_mapped: Dict[str, str]
    processing_timestamp: str
    extraction_duration_seconds: float
    normalisation_duration_seconds: float = 0.0


class DataExtractor(ABC):
    """
    Abstract base class for data extraction
    Enables extension to different file formats (Excel, JSON, CSV, etc.)
    Single responsibility: Extract raw data from files
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialise with configuration manager
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.logger = get_logger(f'extract.{self.__class__.__name__}')
        
    @abstractmethod
    def extract_files(self, folder_path: str, period_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract files from folder and return DataFrames with metadata
        
        Args:
            folder_path: Path to folder containing files
            period_name: Period identifier (e.g., 'feb', 'july')
            
        Returns:
            Dict mapping table names to {'dataframe': df, 'metadata': metadata}
        """
        pass
    
    def generate_table_name(self, subject: str, period_name: str, sheet_name: str) -> str:
        """Generate standardised table name (without prefix for database tables)"""
        # Normalise all components
        norm_subject = normalise_text(subject)
        norm_period = normalise_text(period_name)
        norm_sheet = normalise_text(sheet_name)
        
        # Return clean table name without df_ prefix (matches POC approach)
        return f"{norm_subject}_{norm_period}_{norm_sheet}"