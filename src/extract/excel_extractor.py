"""
Excel-specific data extraction
Moved from extract_light_transform.py as part of module restructure
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time

from .base_extract import DataExtractor, DatasetMetadata
from ..utils.exceptions import ExtractionError
from ..utils.logging_config import get_logger
from ..utils.common import (
    normalise_text, 
    generate_timestamp, 
    clean_dataframe_columns
)

class ExcelDataExtractor(DataExtractor):
    """
    Excel-specific data extraction
    Single responsibility: Extract data from Excel files only
    """
    
    def extract_files(self, folder_path: str, period_name: str) -> Dict[str, Dict[str, Any]]:
        """Extract Excel files to DataFrames with comprehensive metadata and robust validation"""
        start_time = time.time()
        
        folder_path = Path(folder_path)
        extracted_data = {}
        
        self.logger.info(f"Starting Excel extraction from: {folder_path}")
        
        if not folder_path.exists():
            raise ExtractionError(f"Folder not found: {folder_path}", str(folder_path))
        
        # Get Excel files, filter out temporary files
        all_excel_files = list(folder_path.glob("*.xlsx"))
        excel_files = [f for f in all_excel_files if not f.name.startswith('~$')]
        
        if not excel_files:
            self.logger.warning(f"No Excel files found in {folder_path}")
            if all_excel_files:
                self.logger.info(f"Skipped {len(all_excel_files)} temporary Excel files")
            return extracted_data
        
        self.logger.info(f"Found {len(excel_files)} Excel files to process")
        if len(all_excel_files) > len(excel_files):
            self.logger.info(f"Skipped {len(all_excel_files) - len(excel_files)} temporary Excel files")
        
        sheets_to_process = self.config.get_sheets_to_process()
        self.logger.info(f"Will process sheets: {sheets_to_process}")
        
        successful_extractions = 0
        failed_extractions = 0
        
        for file_path in excel_files:
            file_start_time = time.time()
            
            try:
                # Extract subject from filename (remove .xlsx extension)
                subject = file_path.stem
                self.logger.info(f"Processing file: {subject} ({file_path.name})")
                
                # Process each sheet in the Excel file
                with pd.ExcelFile(file_path) as xl_file:
                    available_sheets = xl_file.sheet_names
                    self.logger.info(f"Available sheets: {available_sheets}")
                    
                    file_successful = 0
                    
                    for sheet_name in available_sheets:
                        # Check if this sheet should be processed
                        if sheet_name in sheets_to_process:
                            sheet_start_time = time.time()
                            
                            try:
                                # Read the sheet
                                self.logger.info(f"Reading sheet: {sheet_name}")
                                df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
                                
                                # Check if DataFrame is empty or has no columns
                                if df.empty or len(df.columns) == 0:
                                    self.logger.warning(f"Sheet '{sheet_name}' in {file_path.name} is empty or has no columns - skipping")
                                    continue
                                
                                # Robust column name type enforcement
                                if len(df.columns) > 0:
                                    # Ensure columns are strings before applying string operations
                                    df.columns = [str(col).strip() if col is not None else 'Unnamed' for col in df.columns]
                                else:
                                    self.logger.warning(f"Sheet '{sheet_name}' in {file_path.name} has no columns to process - skipping")
                                    continue
                                
                                self.logger.info(f"Sheet '{sheet_name}' original columns: {list(df.columns)}")
                                
                                # Clean column names
                                df = clean_dataframe_columns(df)
                                
                                # Additional validation after cleaning
                                if df.empty:
                                    self.logger.warning(f"Sheet '{sheet_name}' in {file_path.name} became empty after cleaning - skipping")
                                    continue
                                
                                # Apply old system's name filtering logic
                                # This ensures consistent row counts throughout the pipeline
                                initial_count = len(df)
                                
                                # Check if 'Name' column exists before filtering
                                if 'Name' in df.columns:
                                    # Apply the exact same filtering logic as the old working system
                                    df = df[df['Name'].notna()]  # Remove NaN
                                    df = df[df['Name'].astype(str).str.strip() != '']  # Remove empty strings
                                    df = df[df['Name'].astype(str) != 'nan']  # Remove string 'nan'
                                    df = df[df['Name'].astype(str).str.lower() != 'none']  # Remove string 'none'
                                    
                                    final_count = len(df)
                                    if initial_count != final_count:
                                        self.logger.info(f"Filtered out {initial_count - final_count} rows with invalid names from '{sheet_name}'")
                                    
                                    # Ensure we still have data after filtering
                                    if final_count == 0:
                                        self.logger.warning(f"Sheet '{sheet_name}' in {file_path.name} has no valid data after filtering - skipping")
                                        continue
                                else:
                                    self.logger.warning(f"Sheet '{sheet_name}' in {file_path.name} has no 'Name' column")
                                    # Continue anyway, but log this issue
                                
                                # Generate table name
                                table_name = self.generate_table_name(subject, period_name, sheet_name)
                                
                                # Calculate extraction time for this sheet
                                sheet_duration = time.time() - sheet_start_time
                                
                                # Create metadata with ACTUAL final row count (after filtering)
                                metadata = DatasetMetadata(
                                    source_file=str(file_path),
                                    subject=subject,
                                    period=period_name,
                                    sheet_name=sheet_name,
                                    normalised_sheet_name=normalise_text(sheet_name),
                                    table_name=table_name,
                                    row_count=len(df),  # This is now the CORRECT count after filtering
                                    columns_mapped={},  # Will be populated during normalisation
                                    processing_timestamp=generate_timestamp(),
                                    extraction_duration_seconds=sheet_duration
                                )
                                
                                # Store DataFrame with metadata
                                extracted_data[table_name] = {
                                    'dataframe': df,
                                    'metadata': metadata
                                }
                                
                                self.logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns from '{sheet_name}'")
                                self.logger.info(f"Table name: {table_name}")
                                self.logger.info(f"Columns: {list(df.columns)}")
                                
                                # Log sample data for verification
                                if len(df) > 0:
                                    sample_data = df.iloc[0].to_dict()
                                    self.logger.debug(f"Sample row: {sample_data}")
                                
                                file_successful += 1
                                
                            except Exception as e:
                                self.logger.error(f"Failed to extract sheet '{sheet_name}' from {file_path.name}: {e}")
                                failed_extractions += 1
                                continue
                        else:
                            self.logger.debug(f"Skipping sheet '{sheet_name}' - not in sheets_to_process")
                    
                    if file_successful > 0:
                        file_duration = time.time() - file_start_time
                        self.logger.info(f"Successfully processed {file_successful} sheets from {file_path.name} in {file_duration:.2f}s")
                        successful_extractions += file_successful
                    else:
                        self.logger.warning(f"No sheets processed from {file_path.name}")
                            
            except Exception as e:
                self.logger.error(f"Failed to process file {file_path.name}: {e}")
                failed_extractions += 1
                continue
        
        total_duration = time.time() - start_time
        
        self.logger.info(f"Extraction complete: {successful_extractions} successful, {failed_extractions} failed")
        self.logger.info(f"Total extraction time: {total_duration:.2f}s")
        self.logger.info(f"Extracted {len(extracted_data)} datasets")
        
        if len(extracted_data) == 0:
            raise ExtractionError(f"No data extracted from {folder_path}", str(folder_path))
        
        return extracted_data