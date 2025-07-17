"""
Dataset Comparison for Dataset Comparison Pipeline
Handles comparison logic between two time periods
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import json
from pathlib import Path

from ..config.config_manager import ConfigManager
from ..utils.database_manager import DatabaseManager
from ..utils.exceptions import ComparisonError, DatabaseError
from ..utils.logging_config import get_logger, log_performance_metric
from ..utils.common import (
    format_number_with_commas, 
    calculate_percentage_change, 
    normalise_text,
    create_directory_if_not_exists
)

logger = get_logger('transform.compare')


@dataclass
class ResearcherChange:
    """Changes in researcher metrics between periods"""
    name: str
    esi_field: str
    highly_cited_papers_change: int
    cross_field_score_change: float
    hot_papers_change: int
    times_cited_change: int
    feb_values: Dict[str, Any]
    july_values: Dict[str, Any]
    percentage_changes: Dict[str, float]


@dataclass
class ComparisonReport:
    """Complete comparison report for matched datasets"""
    comparison_id: str
    dataset_1_info: Dict[str, Any]
    dataset_2_info: Dict[str, Any]
    comparison_date: str
    summary_statistics: Dict[str, Any]
    researchers_only_in_dataset_1: List[Dict[str, Any]]
    researchers_only_in_dataset_2: List[Dict[str, Any]]
    researcher_changes: List[ResearcherChange]
    researchers_unchanged: List[Dict[str, Any]]
    comparison_config: Dict[str, Any]


class DataComparator:
    """
    Compare datasets between two time periods
    Single responsibility: Comparison logic only
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialise with configuration manager
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.logger = get_logger('comparator')
        
        # Initialise database manager
        db_path = self.config.get_database_path()
        self.db_manager = DatabaseManager(db_path)
        
        # Get comparison configuration
        self.comparison_config = self.config.get_comparison_config()
        self.comparison_columns = self.comparison_config.get('comparison_columns', [])
        self.float_tolerance = self.comparison_config.get('float_tolerance', 0.001)
        self.include_unchanged = self.comparison_config.get('include_unchanged', True)
        
        # Get dataset period names
        self.dataset_1_period = self.config.get_data_source_config('dataset_1')['period_name']
        self.dataset_2_period = self.config.get_data_source_config('dataset_2')['period_name']
        
        self.logger.info(f"Initialised DataComparator for {self.dataset_1_period} vs {self.dataset_2_period}")
        
    def compare_all_matching_tables(self, clean_table_names: List[str]) -> Dict[str, Any]:
        """
        Compare all matching table pairs
        
        Args:
            clean_table_names: List of cleaned table names
            
        Returns:
            Dictionary with comparison results
        """
        start_time = time.time()
        self.logger.info(f"Starting comparison of {len(clean_table_names)} tables")
        
        # Group tables by subject and sheet type
        table_groups = self._group_tables_for_comparison(clean_table_names)
        
        comparison_results = {
            'comparison_reports': {},
            'summary': {
                'total_comparisons': 0,
                'successful_comparisons': 0,
                'failed_comparisons': 0,
                'comparison_date': datetime.now().isoformat(),
                'dataset_1_period': self.dataset_1_period,
                'dataset_2_period': self.dataset_2_period
            },
            'errors': []
        }
        
        successful_comparisons = 0
        failed_comparisons = 0
        
        for comparison_key, table_pair in table_groups.items():
            comparison_start_time = time.time()
            
            try:
                self.logger.info(f"Comparing tables for: {comparison_key}")
                
                # Perform comparison
                report = self._compare_table_pair(
                    table_pair['dataset_1'], 
                    table_pair['dataset_2'],
                    comparison_key
                )
                
                # Store report
                comparison_results['comparison_reports'][comparison_key] = asdict(report)
                
                # Performance metrics
                comparison_duration = time.time() - comparison_start_time
                log_performance_metric(
                    self.logger, 
                    f"compare_{comparison_key}", 
                    comparison_duration,
                    report.summary_statistics.get('total_researchers_compared', 0)
                )
                
                successful_comparisons += 1
                self.logger.info(f"Successfully completed comparison: {comparison_key}")
                
            except Exception as e:
                failed_comparisons += 1
                error_msg = f"Failed to compare {comparison_key}: {str(e)}"
                self.logger.error(error_msg)
                comparison_results['errors'].append(error_msg)
                continue
        
        # Update summary
        comparison_results['summary']['total_comparisons'] = len(table_groups)
        comparison_results['summary']['successful_comparisons'] = successful_comparisons
        comparison_results['summary']['failed_comparisons'] = failed_comparisons
        
        # Total duration
        total_duration = time.time() - start_time
        
        self.logger.info(f"Comparison complete:")
        self.logger.info(f"  Successful: {successful_comparisons}")
        self.logger.info(f"  Failed: {failed_comparisons}")
        self.logger.info(f"  Duration: {total_duration:.2f}s")
        
        if successful_comparisons == 0:
            raise ComparisonError("No successful comparisons completed")
        
        return comparison_results
    
    def _group_tables_for_comparison(self, table_names: List[str]) -> Dict[str, Dict[str, str]]:
        """Group tables by subject and sheet type for comparison"""
        self.logger.info("Grouping tables for comparison")
        
        # Parse table names to extract components
        table_info = {}
        for table_name in table_names:
            try:
                # Expected format: {subject}_{period}_{sheet_type}
                # Where subject can be multi-word like "space_science"
                # And sheet_type can be multi-word like "highly_cited_only"
                
                parts = table_name.split('_')
                if len(parts) < 3:
                    self.logger.warning(f"Table name has insufficient parts: {table_name}")
                    continue
                
                # Known periods (adjust these to match your actual periods)
                known_periods = [
                    normalise_text(self.dataset_1_period),  # 'feb'
                    normalise_text(self.dataset_2_period)   # 'july' 
                ]
                
                # Known sheet types
                known_sheet_types = ['highly_cited_only', 'incites_researchers']
                
                # Find period in the parts
                period = None
                period_index = None
                for i, part in enumerate(parts):
                    if part in known_periods:
                        period = part
                        period_index = i
                        break
                
                if period is None:
                    self.logger.warning(f"Could not identify period in table name: {table_name}")
                    continue
                
                # Find sheet type (everything after period)
                sheet_parts = parts[period_index + 1:]
                sheet_type = '_'.join(sheet_parts)
                
                if sheet_type not in known_sheet_types:
                    self.logger.warning(f"Unknown sheet type '{sheet_type}' in table: {table_name}")
                    continue
                
                # Subject is everything before period
                subject_parts = parts[:period_index]
                subject = '_'.join(subject_parts)
                
                # Create comparison key (subject + sheet type)
                comparison_key = f"{subject}_{sheet_type}"
                
                if comparison_key not in table_info:
                    table_info[comparison_key] = {}
                
                # Map periods to dataset slots
                if period == normalise_text(self.dataset_1_period):
                    table_info[comparison_key]['dataset_1'] = table_name
                elif period == normalise_text(self.dataset_2_period):
                    table_info[comparison_key]['dataset_2'] = table_name
                
                self.logger.debug(f"Parsed '{table_name}': subject='{subject}', period='{period}', sheet='{sheet_type}'")
                            
            except Exception as e:
                self.logger.warning(f"Could not parse table name {table_name}: {e}")
                continue
        
        # Filter to only complete pairs
        complete_pairs = {}
        for key, tables in table_info.items():
            if 'dataset_1' in tables and 'dataset_2' in tables:
                complete_pairs[key] = tables
                self.logger.info(f"Found matching pair for {key}: {tables['dataset_1']} vs {tables['dataset_2']}")
            else:
                missing_periods = []
                if 'dataset_1' not in tables:
                    missing_periods.append(self.dataset_1_period)
                if 'dataset_2' not in tables:
                    missing_periods.append(self.dataset_2_period)
                self.logger.warning(f"Incomplete pair for {key}: missing {missing_periods}")
        
        self.logger.info(f"Found {len(complete_pairs)} complete table pairs for comparison")
        return complete_pairs
    
    def _compare_table_pair(self, table_1_name: str, table_2_name: str, comparison_key: str) -> ComparisonReport:
        """
        Compare a pair of tables
        
        Args:
            table_1_name: First dataset table name
            table_2_name: Second dataset table name
            comparison_key: Unique identifier for this comparison
            
        Returns:
            ComparisonReport with results
        """
        self.logger.info(f"Comparing {table_1_name} vs {table_2_name}")
        
        try:
            # Get table data
            df_1 = self.db_manager.execute_query(f'SELECT * FROM "{table_1_name}"')
            df_2 = self.db_manager.execute_query(f'SELECT * FROM "{table_2_name}"')
            
            # Validate data
            if df_1.empty:
                raise ComparisonError(f"Dataset 1 table {table_1_name} is empty")
            if df_2.empty:
                raise ComparisonError(f"Dataset 2 table {table_2_name} is empty")
            
            self.logger.info(f"Dataset 1: {len(df_1)} rows, Dataset 2: {len(df_2)} rows")
            
            # Perform comparison analysis
            comparison_results = self._analyze_datasets(df_1, df_2)
            
            # Create comprehensive report
            report = ComparisonReport(
                comparison_id=comparison_key,
                dataset_1_info={
                    'table_name': table_1_name,
                    'period': self.dataset_1_period,
                    'row_count': len(df_1),
                    'columns': list(df_1.columns)
                },
                dataset_2_info={
                    'table_name': table_2_name,
                    'period': self.dataset_2_period,
                    'row_count': len(df_2),
                    'columns': list(df_2.columns)
                },
                comparison_date=datetime.now().isoformat(),
                summary_statistics=comparison_results['summary'],
                researchers_only_in_dataset_1=comparison_results['only_in_dataset_1'],
                researchers_only_in_dataset_2=comparison_results['only_in_dataset_2'],
                researcher_changes=comparison_results['changes'],
                researchers_unchanged=comparison_results['unchanged'] if self.include_unchanged else [],
                comparison_config=self.comparison_config
            )
            
            # Log summary
            self._log_comparison_summary(report)
            
            return report
            
        except Exception as e:
            raise ComparisonError(f"Comparison failed for {comparison_key}: {e}", table_1_name, table_2_name)
    
    def _analyze_datasets(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze two datasets and identify changes
        
        Args:
            df_1: First dataset DataFrame
            df_2: Second dataset DataFrame
            
        Returns:
            Dictionary with analysis results
        """
        # Prepare data for comparison
        df_1_clean = self._prepare_dataframe_for_comparison(df_1)
        df_2_clean = self._prepare_dataframe_for_comparison(df_2)
        
        # Find researchers in each dataset only
        only_in_1 = self._find_researchers_only_in_dataset(df_1_clean, df_2_clean, 1)
        only_in_2 = self._find_researchers_only_in_dataset(df_2_clean, df_1_clean, 2)
        
        # Find matching researchers and analyze changes
        matched_researchers = self._find_matching_researchers(df_1_clean, df_2_clean)
        changes, unchanged = self._analyze_researcher_changes(matched_researchers)
        
        # Calculate summary statistics
        summary = self._calculate_summary_statistics(df_1_clean, df_2_clean, only_in_1, only_in_2, changes, unchanged)
        
        return {
            'summary': summary,
            'only_in_dataset_1': only_in_1,
            'only_in_dataset_2': only_in_2,
            'changes': changes,
            'unchanged': unchanged
        }
    
    def _prepare_dataframe_for_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for comparison by cleaning and standardising
        
        Args:
            df: DataFrame to prepare
            
        Returns:
            Cleaned DataFrame
        """
        # Create copy
        clean_df = df.copy()
        
        # Ensure name column is clean for matching
        if 'name' in clean_df.columns:
            clean_df['name'] = clean_df['name'].astype(str).str.strip()
            # Remove rows with empty names
            clean_df = clean_df[clean_df['name'] != '']
            clean_df = clean_df[clean_df['name'].str.lower() != 'nan']
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['times_cited', 'highly_cited_papers', 'hot_papers', 'indicative_cross_field_score']
        for col in numeric_columns:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0)
        
        return clean_df
    
    def _find_researchers_only_in_dataset(self, df_source: pd.DataFrame, df_other: pd.DataFrame, dataset_num: int) -> List[Dict[str, Any]]:
        """
        Find researchers that exist only in one dataset
        
        Args:
            df_source: Source dataset
            df_other: Other dataset to compare against
            dataset_num: Dataset number (1 or 2) for logging
            
        Returns:
            List of researchers only in source dataset
        """
        # Get names from both datasets
        source_names = set(df_source['name'].str.lower().str.strip())
        other_names = set(df_other['name'].str.lower().str.strip())
        
        # Find names only in source
        only_in_source_names = source_names - other_names
        
        # Get full records for these researchers
        only_in_source = []
        for name in only_in_source_names:
            researcher_rows = df_source[df_source['name'].str.lower().str.strip() == name]
            if not researcher_rows.empty:
                researcher = researcher_rows.iloc[0]
                only_in_source.append({
                    'name': researcher['name'],
                    'esi_field': researcher.get('esi_field', ''),
                    'times_cited': int(researcher.get('times_cited', 0)),
                    'highly_cited_papers': int(researcher.get('highly_cited_papers', 0)),
                    'hot_papers': int(researcher.get('hot_papers', 0)),
                    'indicative_cross_field_score': float(researcher.get('indicative_cross_field_score', 0.0))
                })
        
        self.logger.info(f"Found {len(only_in_source)} researchers only in dataset {dataset_num}")
        return only_in_source
    
    def _find_matching_researchers(self, df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
        """
        Find researchers that exist in both datasets
        
        Args:
            df_1: First dataset
            df_2: Second dataset
            
        Returns:
            DataFrame with matched researchers from both periods
        """
        # Perform inner join on name (case-insensitive)
        df_1_match = df_1.copy()
        df_2_match = df_2.copy()
        
        # Create normalised name columns for matching
        df_1_match['name_lower'] = df_1_match['name'].str.lower().str.strip()
        df_2_match['name_lower'] = df_2_match['name'].str.lower().str.strip()
        
        # Merge on normalised names
        matched = pd.merge(
            df_1_match, 
            df_2_match, 
            on='name_lower', 
            suffixes=('_dataset1', '_dataset2'),
            how='inner'
        )
        
        self.logger.info(f"Found {len(matched)} researchers in both datasets")
        return matched
    
    def _analyze_researcher_changes(self, matched_df: pd.DataFrame) -> Tuple[List[ResearcherChange], List[Dict[str, Any]]]:
        """
        Analyze changes in researcher metrics
        
        Args:
            matched_df: DataFrame with matched researchers
            
        Returns:
            Tuple of (changes_list, unchanged_list)
        """
        changes = []
        unchanged = []
        
        for _, row in matched_df.iterrows():
            # Get values from both periods
            name = row['name_dataset2']  # Use dataset 2 name (more recent)
            esi_field = row.get('esi_field_dataset2', row.get('esi_field_dataset1', ''))
            
            # Calculate changes in comparison columns
            metric_changes = {}
            has_changes = False
            
            for column in self.comparison_columns:
                col_1 = f"{column}_dataset1"
                col_2 = f"{column}_dataset2"
                
                if col_1 in row and col_2 in row:
                    val_1 = row[col_1]
                    val_2 = row[col_2]
                    
                    # Handle float comparison with tolerance
                    if isinstance(val_1, float) or isinstance(val_2, float):
                        change = float(val_2) - float(val_1)
                        if abs(change) > self.float_tolerance:
                            has_changes = True
                            metric_changes[column] = change
                        else:
                            metric_changes[column] = 0.0
                    else:
                        change = int(val_2) - int(val_1)
                        if change != 0:
                            has_changes = True
                        metric_changes[column] = change
            
            # Prepare values dictionaries
            dataset_1_values = {}
            dataset_2_values = {}
            percentage_changes = {}
            
            for column in self.comparison_columns:
                col_1 = f"{column}_dataset1"
                col_2 = f"{column}_dataset2"
                
                if col_1 in row and col_2 in row:
                    val_1 = row[col_1]
                    val_2 = row[col_2]
                    
                    dataset_1_values[column] = val_1
                    dataset_2_values[column] = val_2
                    
                    # Calculate percentage change
                    if val_1 != 0:
                        pct_change = calculate_percentage_change(float(val_1), float(val_2))
                        percentage_changes[column] = pct_change
                    else:
                        percentage_changes[column] = float('inf') if val_2 > 0 else 0.0
            
            # Create record
            if has_changes:
                change_record = ResearcherChange(
                    name=name,
                    esi_field=esi_field,
                    highly_cited_papers_change=metric_changes.get('highly_cited_papers', 0),
                    cross_field_score_change=metric_changes.get('indicative_cross_field_score', 0.0),
                    hot_papers_change=metric_changes.get('hot_papers', 0),
                    times_cited_change=metric_changes.get('times_cited', 0),
                    feb_values=dataset_1_values,
                    july_values=dataset_2_values,
                    percentage_changes=percentage_changes
                )
                changes.append(change_record)
            else:
                unchanged_record = {
                    'name': name,
                    'esi_field': esi_field,
                    **dataset_2_values  # Use current values
                }
                unchanged.append(unchanged_record)
        
        self.logger.info(f"Found {len(changes)} researchers with changes, {len(unchanged)} unchanged")
        return changes, unchanged
    
    def _calculate_summary_statistics(self, df_1: pd.DataFrame, df_2: pd.DataFrame, 
                                    only_in_1: List, only_in_2: List, 
                                    changes: List, unchanged: List) -> Dict[str, Any]:
        """
        Calculate summary statistics for the comparison
        
        Args:
            df_1: First dataset
            df_2: Second dataset
            only_in_1: Researchers only in dataset 1
            only_in_2: Researchers only in dataset 2
            changes: Researchers with changes
            unchanged: Unchanged researchers
            
        Returns:
            Summary statistics dictionary
        """
        total_matched = len(changes) + len(unchanged)
        
        # Calculate aggregate changes
        aggregate_changes = {}
        for column in self.comparison_columns:
            total_change = sum(getattr(change, f"{column}_change", 0) for change in changes)
            aggregate_changes[f"total_{column}_change"] = total_change
            
            # Average change among those who changed
            changes_in_column = [getattr(change, f"{column}_change", 0) for change in changes 
                               if getattr(change, f"{column}_change", 0) != 0]
            if changes_in_column:
                aggregate_changes[f"avg_{column}_change"] = sum(changes_in_column) / len(changes_in_column)
            else:
                aggregate_changes[f"avg_{column}_change"] = 0
        
        return {
            'dataset_1_total': len(df_1),
            'dataset_2_total': len(df_2),
            'total_researchers_compared': total_matched,
            'researchers_only_in_dataset_1': len(only_in_1),
            'researchers_only_in_dataset_2': len(only_in_2),
            'researchers_with_changes': len(changes),
            'researchers_unchanged': len(unchanged),
            'match_rate': (total_matched / max(len(df_1), len(df_2))) if max(len(df_1), len(df_2)) > 0 else 0,
            'change_rate': (len(changes) / total_matched) if total_matched > 0 else 0,
            'aggregate_changes': aggregate_changes
        }
    
    def _log_comparison_summary(self, report: ComparisonReport) -> None:
        """
        Log summary of comparison results
        
        Args:
            report: Comparison report to summarize
        """
        stats = report.summary_statistics
        
        self.logger.info(f"Comparison Summary for {report.comparison_id}:")
        self.logger.info(f"  Dataset 1 ({self.dataset_1_period}): {format_number_with_commas(stats['dataset_1_total'])} researchers")
        self.logger.info(f"  Dataset 2 ({self.dataset_2_period}): {format_number_with_commas(stats['dataset_2_total'])} researchers")
        self.logger.info(f"  Matched: {format_number_with_commas(stats['total_researchers_compared'])} researchers")
        self.logger.info(f"  Only in {self.dataset_1_period}: {format_number_with_commas(stats['researchers_only_in_dataset_1'])}")
        self.logger.info(f"  Only in {self.dataset_2_period}: {format_number_with_commas(stats['researchers_only_in_dataset_2'])}")
        self.logger.info(f"  With changes: {format_number_with_commas(stats['researchers_with_changes'])}")
        self.logger.info(f"  Unchanged: {format_number_with_commas(stats['researchers_unchanged'])}")
        self.logger.info(f"  Match rate: {stats['match_rate']:.1%}")
        self.logger.info(f"  Change rate: {stats['change_rate']:.1%}")
    
    def save_comparison_reports(self, comparison_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save comparison reports to files
        
        Args:
            comparison_results: Results from compare_all_matching_tables
            
        Returns:
            Dictionary mapping comparison keys to file paths
        """
        output_config = self.config.get_output_config()
        reports_folder = output_config['reports_folder']
        
        # Create output directories
        create_directory_if_not_exists(reports_folder)
        
        saved_files = {}
        
        for comparison_key, report_data in comparison_results['comparison_reports'].items():
            try:
                # Determine output subdirectory based on sheet type
                if 'highly_cited' in comparison_key.lower():
                    folder_name = self.config.get_output_folder_name('highly_cited_only')
                    output_dir = Path(reports_folder) / folder_name
                elif 'incites' in comparison_key.lower():
                    folder_name = self.config.get_output_folder_name('incites_researchers')
                    output_dir = Path(reports_folder) / folder_name
                else:
                    output_dir = Path(reports_folder)
                
                # Create subdirectory
                create_directory_if_not_exists(str(output_dir))
                
                # Generate filename
                safe_filename = normalise_text(comparison_key)
                filename = f"{safe_filename}_comparison_report.json"
                file_path = output_dir / filename
                
                # Save report
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
                
                saved_files[comparison_key] = str(file_path)
                self.logger.info(f"Saved comparison report: {file_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to save report for {comparison_key}: {e}")
                continue
        
        # Save summary report
        try:
            summary_file = Path(reports_folder) / "comparison_summary.json"
            summary_data = {
                'generation_date': datetime.now().isoformat(),
                'summary': comparison_results['summary'],
                'errors': comparison_results['errors'],
                'reports_generated': list(saved_files.keys()),
                'config_used': self.comparison_config
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Saved comparison summary: {summary_file}")
            saved_files['_summary'] = str(summary_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save summary report: {e}")
        
        return saved_files