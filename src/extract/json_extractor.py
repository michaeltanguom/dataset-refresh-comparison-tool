"""
JSON-specific data extraction for comparison reports
Follows the established DataExtractor ABC pattern
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import re

from .base_extract import DataExtractor, DatasetMetadata
from ..utils.exceptions import ExtractionError
from ..utils.logging_config import get_logger
from ..utils.common import normalise_text, generate_timestamp

class JSONDataExtractor(DataExtractor):
    """
    JSON-specific data extraction for comparison reports
    Single responsibility: Extract data from JSON comparison report files
    """
    
    def extract_files(self, folder_path: str, period_name: str) -> Dict[str, Dict[str, Any]]:
        """Extract JSON comparison reports with metadata"""
        start_time = time.time()
        
        folder_path = Path(folder_path)
        extracted_data = {}
        
        self.logger.info(f"Starting JSON extraction from: {folder_path}")
        
        if not folder_path.exists():
            raise ExtractionError(f"Reports folder not found: {folder_path}", str(folder_path))
        
        # Get JSON files from subdirectories (comparison reports are organised by type)
        json_files = []
        for subdir in folder_path.iterdir():
            if subdir.is_dir():
                json_files.extend(list(subdir.glob("*_comparison_report.json")))
        
        if not json_files:
            self.logger.warning(f"No JSON comparison reports found in {folder_path}")
            return extracted_data
        
        self.logger.info(f"Found {len(json_files)} JSON reports to process")
        
        successful_extractions = 0
        failed_extractions = 0
        
        for json_file in json_files:
            file_start_time = time.time()
            
            try:
                self.logger.info(f"Processing JSON report: {json_file.name}")
                
                # Load and validate JSON data
                json_data = self._load_and_validate_json(json_file)
                
                # Extract subject and sheet type from filename/directory structure
                subject, sheet_type = self._parse_report_metadata(json_file)
                
                # Generate table name following existing pattern
                table_name = self.generate_table_name(subject, period_name, sheet_type)
                
                # Calculate extraction time
                file_duration = time.time() - file_start_time
                
                # Create metadata following existing pattern
                metadata = DatasetMetadata(
                    source_file=str(json_file),
                    subject=subject,
                    period=period_name,
                    sheet_name=sheet_type,
                    normalised_sheet_name=normalise_text(sheet_type),
                    table_name=table_name,
                    row_count=self._calculate_researcher_count(json_data),
                    columns_mapped={},  # JSON doesn't need column mapping
                    processing_timestamp=generate_timestamp(),
                    extraction_duration_seconds=file_duration
                )
                
                # Store JSON data with metadata (raw JSON, not DataFrame)
                extracted_data[table_name] = {
                    'json_data': json_data,  # Raw JSON for HTML generation
                    'metadata': metadata
                }
                
                self.logger.info(f"Successfully extracted {json_file.name} -> {table_name}")
                self.logger.info(f"  Report contains {metadata.row_count} researcher records")
                
                successful_extractions += 1
                
            except Exception as e:
                self.logger.error(f"Failed to extract {json_file.name}: {e}")
                failed_extractions += 1
                continue
        
        # Summary logging
        total_duration = time.time() - start_time
        self.logger.info(f"JSON extraction complete: {successful_extractions} successful, {failed_extractions} failed")
        self.logger.info(f"Total extraction time: {total_duration:.2f}s")
        
        if successful_extractions == 0:
            raise ExtractionError(f"No JSON reports successfully extracted from {folder_path}", str(folder_path))
        
        return extracted_data
    
    def _load_and_validate_json(self, json_file: Path) -> Dict[str, Any]:
        """Load and validate JSON file structure"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Validate required fields for comparison reports
            required_fields = ['comparison_id', 'summary_statistics', 'researcher_changes']
            missing_fields = []
            
            for field in required_fields:
                if field not in json_data:
                    missing_fields.append(field)
            
            if missing_fields:
                raise ExtractionError(
                    f"JSON file missing required fields: {missing_fields}",
                    str(json_file)
                )
            
            self.logger.debug(f"JSON validation passed for {json_file.name}")
            return json_data
            
        except json.JSONDecodeError as e:
            raise ExtractionError(f"Invalid JSON format in {json_file.name}: {e}", str(json_file))
        except Exception as e:
            raise ExtractionError(f"Failed to load JSON file {json_file.name}: {e}", str(json_file))
    
    def _parse_report_metadata(self, json_file: Path) -> tuple:
        """
        Parse subject and sheet type from filename and directory structure
        
        Expected structure: comparison_reports/highly_cited_only/subject_sheet_comparison_report.json
        """
        try:
            # Get sheet type from parent directory name
            parent_dir = json_file.parent.name
            sheet_type = normalise_text(parent_dir)
            
            # Parse subject from filename
            # Expected pattern: {subject}_{sheet_type}_comparison_report.json
            filename_without_ext = json_file.stem
            
            # Remove the standard suffix
            if filename_without_ext.endswith('_comparison_report'):
                base_name = filename_without_ext[:-len('_comparison_report')]
            else:
                base_name = filename_without_ext
            
            # Remove sheet type suffix if present
            sheet_type_normalised = normalise_text(sheet_type)
            if base_name.endswith(f'_{sheet_type_normalised}'):
                subject = base_name[:-len(f'_{sheet_type_normalised}')]
            else:
                subject = base_name
            
            self.logger.debug(f"Parsed metadata: subject='{subject}', sheet_type='{sheet_type}'")
            return subject, sheet_type
            
        except Exception as e:
            # Fallback parsing
            self.logger.warning(f"Could not parse metadata from {json_file}, using fallback: {e}")
            return json_file.parent.name, "comparison_report"
    
    def _calculate_researcher_count(self, json_data: Dict[str, Any]) -> int:
        """Calculate total number of researcher records in the report"""
        try:
            count = 0
            
            # Count researchers with changes
            count += len(json_data.get('researcher_changes', []))
            
            # Count unchanged researchers
            count += len(json_data.get('researchers_unchanged', []))
            
            # Count researchers only in dataset 1
            count += len(json_data.get('researchers_only_in_dataset_1', []))
            
            # Count researchers only in dataset 2  
            count += len(json_data.get('researchers_only_in_dataset_2', []))
            
            return count
            
        except Exception as e:
            self.logger.warning(f"Could not calculate researcher count: {e}")
            return 0