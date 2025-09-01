"""
SQL-Based Dataset Comparison Engine
Uses database queries for efficient comparison analysis between time periods
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
    duplicate_analysis: Dict[str, Any]
    outlier_analysis: Dict[str, Any]
    comparison_config: Dict[str, Any]


class SQLDataComparator:
    """
    SQL-based dataset comparison with config-driven field mapping
    Single responsibility: Database-driven comparison analysis
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialise with configuration manager
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.logger = get_logger('sql_comparator')
        
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
        
        # Get field tolerances from config
        self.field_tolerances = self._build_field_tolerances()
        
        self.logger.info(f"Initialised SQLDataComparator for {self.dataset_1_period} vs {self.dataset_2_period}")
    
    def _build_field_tolerances(self) -> Dict[str, float]:
        """Build field-specific tolerances from config"""
        tolerances = {}
        
        for field in self.comparison_columns:
            if field in ['times_cited', 'highly_cited_papers', 'hot_papers']:
                tolerances[field] = 0  # Integer fields - no tolerance
            elif field == 'indicative_cross_field_score':
                tolerances[field] = self.float_tolerance
            else:
                tolerances[field] = 0  # Default to no tolerance
                
        self.logger.debug(f"Field tolerances: {tolerances}")
        return tolerances
    
    def compare_all_matching_tables(self, table_names: List[str]) -> Dict[str, Any]:
        """
        Compare all matching table pairs using SQL
        
        Args:
            table_names: List of all available table names
            
        Returns:
            Dictionary with comparison results
        """
        start_time = time.time()
        self.logger.info(f"Starting SQL-based comparison of {len(table_names)} tables")
        
        # Group tables by subject and sheet type
        table_groups = self._group_tables_for_comparison(table_names)
        
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
                
                # Perform SQL-based comparison
                report = self._compare_table_pair_sql(
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
                self.logger.info(f"Successfully completed SQL comparison: {comparison_key}")
                
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
        
        self.logger.info(f"SQL comparison complete:")
        self.logger.info(f"  Successful: {successful_comparisons}")
        self.logger.info(f"  Failed: {failed_comparisons}")
        self.logger.info(f"  Duration: {total_duration:.2f}s")
        
        if successful_comparisons == 0:
            raise ComparisonError("No successful comparisons completed")
        
        return comparison_results
    
    def _compare_table_pair_sql(self, table_1_name: str, table_2_name: str, comparison_key: str) -> ComparisonReport:
        """
        Compare a pair of tables using SQL queries
        
        Args:
            table_1_name: First dataset table name
            table_2_name: Second dataset table name
            comparison_key: Unique identifier for this comparison
            
        Returns:
            ComparisonReport with results
        """
        self.logger.info(f"SQL comparison: {table_1_name} vs {table_2_name}")
        
        try:
            # Validate tables exist and get basic info
            table_1_info = self._get_table_info(table_1_name)
            table_2_info = self._get_table_info(table_2_name)
            
            # Perform SQL-based analysis
            only_in_1 = self._find_researchers_only_in_dataset_1(table_1_name, table_2_name)
            only_in_2 = self._find_researchers_only_in_dataset_2(table_1_name, table_2_name)
            changes = self._find_researchers_with_changes(table_1_name, table_2_name)
            unchanged = self._find_researchers_unchanged(table_1_name, table_2_name) if self.include_unchanged else []
            
            # Quality analysis
            duplicate_analysis = self._analyze_duplicate_changes(table_1_name, table_2_name)
            outlier_analysis = self._analyze_outlier_changes(table_1_name, table_2_name)
            
            # Calculate summary statistics using SQL
            summary_stats = self._calculate_summary_statistics_sql(table_1_name, table_2_name, len(changes), len(unchanged))
            
            # Create comprehensive report
            report = ComparisonReport(
                comparison_id=comparison_key,
                dataset_1_info=table_1_info,
                dataset_2_info=table_2_info,
                comparison_date=datetime.now().isoformat(),
                summary_statistics=summary_stats,
                researchers_only_in_dataset_1=only_in_1,
                researchers_only_in_dataset_2=only_in_2,
                researcher_changes=changes,
                researchers_unchanged=unchanged,
                duplicate_analysis=duplicate_analysis,
                outlier_analysis=outlier_analysis,
                comparison_config=self.comparison_config
            )
            
            # Log summary
            self._log_comparison_summary(report)
            
            return report
            
        except Exception as e:
            raise ComparisonError(f"SQL comparison failed for {comparison_key}: {e}", table_1_name, table_2_name)
    
    def _get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get basic table information using SQL"""
        try:
            count_query = f'SELECT COUNT(*) as row_count FROM "{table_name}"'
            count_result = self.db_manager.execute_query(count_query)
            row_count = count_result.iloc[0]['row_count']
            
            # Get column info
            schema = self.db_manager.get_table_schema(table_name)
            
            return {
                'table_name': table_name,
                'row_count': row_count,
                'columns': list(schema.keys()),
                'column_count': len(schema)
            }
        except Exception as e:
            raise ComparisonError(f"Failed to get table info for {table_name}: {e}")
    
    def _find_researchers_only_in_dataset_1(self, table_1: str, table_2: str) -> List[Dict[str, Any]]:
        """Find researchers only in dataset 1 using SQL EXCEPT"""
        query = f"""
        SELECT 
            name,
            esi_field,
            times_cited,
            highly_cited_papers,
            hot_papers,
            indicative_cross_field_score,
            is_duplicate,
            duplicate_profile,
            duplicate_group_id
        FROM "{table_1}"
        WHERE name NOT IN (SELECT name FROM "{table_2}")
        ORDER BY highly_cited_papers DESC
        """
        
        try:
            result_df = self.db_manager.execute_query(query)
            researchers = result_df.to_dict('records')
            
            self.logger.info(f"Found {len(researchers)} researchers only in {self.dataset_1_period}")
            return researchers
            
        except Exception as e:
            self.logger.error(f"Failed to find researchers only in dataset 1: {e}")
            return []
    
    def _find_researchers_only_in_dataset_2(self, table_1: str, table_2: str) -> List[Dict[str, Any]]:
        """Find researchers only in dataset 2 using SQL EXCEPT"""
        query = f"""
        SELECT 
            name,
            esi_field,
            times_cited,
            highly_cited_papers,
            hot_papers,
            indicative_cross_field_score,
            is_duplicate,
            duplicate_profile,
            duplicate_group_id
        FROM "{table_2}"
        WHERE name NOT IN (SELECT name FROM "{table_1}")
        ORDER BY highly_cited_papers DESC
        """
        
        try:
            result_df = self.db_manager.execute_query(query)
            researchers = result_df.to_dict('records')
            
            self.logger.info(f"Found {len(researchers)} researchers only in {self.dataset_2_period}")
            return researchers
            
        except Exception as e:
            self.logger.error(f"Failed to find researchers only in dataset 2: {e}")
            return []
    
    def _find_researchers_with_changes(self, table_1: str, table_2: str) -> List[ResearcherChange]:
        """Find researchers with metric changes using SQL"""
        
        # Build WHERE clause for change detection
        change_conditions = []
        for field, tolerance in self.field_tolerances.items():
            if tolerance > 0:
                change_conditions.append(f"ABS(t2.{field} - t1.{field}) > {tolerance}")
            else:
                change_conditions.append(f"t2.{field} != t1.{field}")
        
        where_clause = " OR ".join(change_conditions)
        
        # Build SELECT clause with all comparison fields
        select_fields = ["t1.name", "t2.esi_field"]
        for field in self.comparison_columns:
            select_fields.extend([
                f"t1.{field} as {field}_dataset1",
                f"t2.{field} as {field}_dataset2",
                f"(t2.{field} - t1.{field}) as {field}_change"
            ])
        
        query = f"""
        SELECT 
            {', '.join(select_fields)}
        FROM "{table_1}" t1
        INNER JOIN "{table_2}" t2 ON t1.name = t2.name
        WHERE {where_clause}
        ORDER BY (t2.highly_cited_papers - t1.highly_cited_papers) DESC
        """
        
        try:
            result_df = self.db_manager.execute_query(query)
            
            # Convert to ResearcherChange objects
            changes = []
            for _, row in result_df.iterrows():
                # Build values dictionaries
                feb_values = {}
                july_values = {}
                percentage_changes = {}
                
                for field in self.comparison_columns:
                    feb_val = row[f"{field}_dataset1"]
                    july_val = row[f"{field}_dataset2"]
                    
                    feb_values[field] = feb_val
                    july_values[field] = july_val
                    
                    # Calculate percentage change
                    if feb_val != 0:
                        pct_change = calculate_percentage_change(float(feb_val), float(july_val))
                        percentage_changes[field] = pct_change
                    else:
                        percentage_changes[field] = float('inf') if july_val > 0 else 0.0
                
                change_record = ResearcherChange(
                    name=row['name'],
                    esi_field=row['esi_field'],
                    highly_cited_papers_change=int(row.get('highly_cited_papers_change', 0)),
                    cross_field_score_change=float(row.get('indicative_cross_field_score_change', 0.0)),
                    hot_papers_change=int(row.get('hot_papers_change', 0)),
                    times_cited_change=int(row.get('times_cited_change', 0)),
                    feb_values=feb_values,
                    july_values=july_values,
                    percentage_changes=percentage_changes
                )
                changes.append(change_record)
            
            self.logger.info(f"Found {len(changes)} researchers with changes")
            return changes
            
        except Exception as e:
            self.logger.error(f"Failed to find researchers with changes: {e}")
            return []
    
    def _find_researchers_unchanged(self, table_1: str, table_2: str) -> List[Dict[str, Any]]:
        """Find researchers with no metric changes using SQL"""
        
        # Build WHERE clause for no changes
        no_change_conditions = []
        for field, tolerance in self.field_tolerances.items():
            if tolerance > 0:
                no_change_conditions.append(f"ABS(t2.{field} - t1.{field}) <= {tolerance}")
            else:
                no_change_conditions.append(f"t2.{field} = t1.{field}")
        
        where_clause = " AND ".join(no_change_conditions)
        
        query = f"""
        SELECT 
            t2.name,
            t2.esi_field,
            t2.times_cited,
            t2.highly_cited_papers,
            t2.hot_papers,
            t2.indicative_cross_field_score
        FROM "{table_1}" t1
        INNER JOIN "{table_2}" t2 ON t1.name = t2.name
        WHERE {where_clause}
        ORDER BY t2.highly_cited_papers DESC
        """
        
        try:
            result_df = self.db_manager.execute_query(query)
            unchanged = result_df.to_dict('records')
            
            self.logger.info(f"Found {len(unchanged)} researchers with no changes")
            return unchanged
            
        except Exception as e:
            self.logger.error(f"Failed to find unchanged researchers: {e}")
            return []
    
    def _calculate_summary_statistics_sql(self, table_1: str, table_2: str, changes_count: int, unchanged_count: int) -> Dict[str, Any]:
        """Calculate summary statistics using SQL aggregations"""
        
        try:
            # Basic counts
            count_query_1 = f'SELECT COUNT(*) as total FROM "{table_1}"'
            count_query_2 = f'SELECT COUNT(*) as total FROM "{table_2}"'
            
            total_1 = self.db_manager.execute_query(count_query_1).iloc[0]['total']
            total_2 = self.db_manager.execute_query(count_query_2).iloc[0]['total']
            
            # Matched researchers count
            matched_query = f"""
            SELECT COUNT(*) as matched
            FROM "{table_1}" t1
            INNER JOIN "{table_2}" t2 ON t1.name = t2.name
            """
            total_matched = self.db_manager.execute_query(matched_query).iloc[0]['matched']
            
            # Calculate aggregate changes using SQL
            aggregate_changes = self._calculate_aggregate_changes_sql(table_1, table_2)
            
            return {
                'dataset_1_total': total_1,
                'dataset_2_total': total_2,
                'total_researchers_compared': total_matched,
                'researchers_only_in_dataset_1': total_1 - total_matched,
                'researchers_only_in_dataset_2': total_2 - total_matched,
                'researchers_with_changes': changes_count,
                'researchers_unchanged': unchanged_count,
                'match_rate': (total_matched / max(total_1, total_2)) if max(total_1, total_2) > 0 else 0,
                'change_rate': (changes_count / total_matched) if total_matched > 0 else 0,
                'aggregate_changes': aggregate_changes
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate summary statistics: {e}")
            return {}
    
    def _calculate_aggregate_changes_sql(self, table_1: str, table_2: str) -> Dict[str, Any]:
        """Calculate aggregate changes across all metrics using SQL"""
        
        # Build aggregation query for all comparison fields
        aggregation_fields = []
        for field in self.comparison_columns:
            aggregation_fields.extend([
                f"SUM(t2.{field} - t1.{field}) as total_{field}_change",
                f"AVG(t2.{field} - t1.{field}) as avg_{field}_change"
            ])
        
        query = f"""
        SELECT 
            {', '.join(aggregation_fields)}
        FROM "{table_1}" t1
        INNER JOIN "{table_2}" t2 ON t1.name = t2.name
        """
        
        try:
            result = self.db_manager.execute_query(query)
            if not result.empty:
                return result.iloc[0].to_dict()
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to calculate aggregate changes: {e}")
            return {}
    
    def _analyze_duplicate_changes(self, table_1: str, table_2: str) -> Dict[str, Any]:
        """Analyze changes in duplicate status between periods"""
        
        duplicate_queries = {
            # Researchers who became duplicates
            'became_duplicates': f"""
                SELECT 
                    t2.name,
                    t2.esi_field,
                    t1.is_duplicate as feb_duplicate_status,
                    t2.is_duplicate as july_duplicate_status,
                    t2.duplicate_profile,
                    t2.duplicate_group_id
                FROM "{table_1}" t1
                INNER JOIN "{table_2}" t2 ON t1.name = t2.name
                WHERE t1.is_duplicate = FALSE AND t2.is_duplicate = TRUE
            """,
            
            # Researchers who were resolved from duplicates
            'resolved_duplicates': f"""
                SELECT 
                    t2.name,
                    t2.esi_field,
                    t1.is_duplicate as feb_duplicate_status,
                    t2.is_duplicate as july_duplicate_status,
                    t1.duplicate_profile as feb_duplicate_profile
                FROM "{table_1}" t1
                INNER JOIN "{table_2}" t2 ON t1.name = t2.name
                WHERE t1.is_duplicate = TRUE AND t2.is_duplicate = FALSE
            """,
            
            # Current duplicates in latest dataset
            'current_duplicates': f"""
                SELECT 
                    name,
                    esi_field,
                    duplicate_profile,
                    duplicate_group_id
                FROM "{table_2}"
                WHERE is_duplicate = TRUE
                ORDER BY duplicate_group_id
            """
        }
        
        analysis_results = {}
        
        for analysis_type, query in duplicate_queries.items():
            try:
                result_df = self.db_manager.execute_query(query)
                analysis_results[analysis_type] = result_df.to_dict('records')
                self.logger.info(f"Duplicate analysis - {analysis_type}: {len(result_df)} records")
                
            except Exception as e:
                self.logger.error(f"Failed duplicate analysis for {analysis_type}: {e}")
                analysis_results[analysis_type] = []
        
        # Summary statistics
        analysis_results['summary'] = {
            'total_became_duplicates': len(analysis_results.get('became_duplicates', [])),
            'total_resolved_duplicates': len(analysis_results.get('resolved_duplicates', [])),
            'current_duplicate_count': len(analysis_results.get('current_duplicates', []))
        }
        
        return analysis_results
    
    def _analyze_outlier_changes(self, table_1: str, table_2: str) -> Dict[str, Any]:
        """Analyze changes in outlier status between periods for statistical fields"""
        
        statistical_fields = ['times_cited', 'highly_cited_papers', 'hot_papers', 'indicative_cross_field_score']
        outlier_field_map = {
            'times_cited': 'is_outlier_times_cited',
            'highly_cited_papers': 'is_outlier_highly_cited_papers', 
            'hot_papers': 'is_outlier_hot_papers',
            'indicative_cross_field_score': 'is_outlier_cross_field_score'
        }
        
        analysis_results = {}
        
        for metric_field in statistical_fields:
            outlier_field = outlier_field_map.get(metric_field)
            
            outlier_queries = {
                f'{metric_field}_became_outliers': f"""
                    SELECT 
                        t2.name,
                        t2.esi_field,
                        t1.{metric_field} as feb_{metric_field},
                        t2.{metric_field} as july_{metric_field},
                        t2.outlier_score_{metric_field}
                    FROM "{table_1}" t1
                    INNER JOIN "{table_2}" t2 ON t1.name = t2.name
                    WHERE t1.{outlier_field} = FALSE AND t2.{outlier_field} = TRUE
                """,
                
                f'{metric_field}_resolved_outliers': f"""
                    SELECT 
                        t2.name,
                        t2.esi_field,
                        t1.{metric_field} as feb_{metric_field},
                        t2.{metric_field} as july_{metric_field},
                        t1.outlier_score_{metric_field} as feb_outlier_score
                    FROM "{table_1}" t1
                    INNER JOIN "{table_2}" t2 ON t1.name = t2.name
                    WHERE t1.{outlier_field} = TRUE AND t2.{outlier_field} = FALSE
                """,
                
                f'{metric_field}_current_outliers': f"""
                    SELECT 
                        name,
                        esi_field,
                        {metric_field},
                        outlier_score_{metric_field},
                        outlier_method_{metric_field}
                    FROM "{table_2}"
                    WHERE {outlier_field} = TRUE
                    ORDER BY outlier_score_{metric_field} DESC
                """
            }
            
            for analysis_type, query in outlier_queries.items():
                try:
                    result_df = self.db_manager.execute_query(query)
                    analysis_results[analysis_type] = result_df.to_dict('records')
                    self.logger.info(f"Outlier analysis - {analysis_type}: {len(result_df)} records")
                    
                except Exception as e:
                    self.logger.error(f"Failed outlier analysis for {analysis_type}: {e}")
                    analysis_results[analysis_type] = []
        
        # Calculate summary
        became_outliers_total = sum(len(v) for k, v in analysis_results.items() if 'became_outliers' in k)
        resolved_outliers_total = sum(len(v) for k, v in analysis_results.items() if 'resolved_outliers' in k)
        current_outliers_total = sum(len(v) for k, v in analysis_results.items() if 'current_outliers' in k)
        
        analysis_results['summary'] = {
            'total_became_outliers': became_outliers_total,
            'total_resolved_outliers': resolved_outliers_total,
            'current_outlier_count': current_outliers_total,
            'analyzed_metrics': list(statistical_fields)
        }
        
        return analysis_results
    
    def _group_tables_for_comparison(self, table_names: List[str]) -> Dict[str, Dict[str, str]]:
        """Group tables by subject and sheet type for comparison"""
        self.logger.info("Grouping tables for comparison")
        
        # Parse table names to extract components
        table_info = {}
        for table_name in table_names:
            try:
                # Expected format: {subject}_{period}_{sheet_type}
                parts = table_name.split('_')
                if len(parts) < 3:
                    self.logger.warning(f"Table name has insufficient parts: {table_name}")
                    continue
                
                # Known periods
                known_periods = [
                    normalise_text(self.dataset_1_period),  # 'feb'
                    normalise_text(self.dataset_2_period)   # 'july' 
                ]
                
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
                
                # Subject is everything before period
                subject_parts = parts[:period_index]
                subject = '_'.join(subject_parts)
                
                # Sheet type is everything after period
                sheet_parts = parts[period_index + 1:]
                sheet_type = '_'.join(sheet_parts)
                
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
    
    def _log_comparison_summary(self, report: ComparisonReport) -> None:
        """Log summary of comparison results"""
        stats = report.summary_statistics
        
        self.logger.info(f"Comparison Summary for {report.comparison_id}:")
        self.logger.info(f"  Dataset 1 ({self.dataset_1_period}): {format_number_with_commas(stats['dataset_1_total'])} researchers")
        self.logger.info(f"  Dataset 2 ({self.dataset_2_period}): {format_number_with_commas(stats['dataset_2_total'])} researchers")
        self.logger.info(f"  Matched: {format_number_with_commas(stats['total_researchers_compared'])} researchers")
        self.logger.info(f"  With changes: {format_number_with_commas(stats['researchers_with_changes'])}")
        self.logger.info(f"  Match rate: {stats['match_rate']:.1%}")
        self.logger.info(f"  Change rate: {stats['change_rate']:.1%}")
        
        # Log quality analysis summary
        if hasattr(report, 'duplicate_analysis') and report.duplicate_analysis.get('summary'):
            dup_summary = report.duplicate_analysis['summary']
            self.logger.info(f"  Current duplicates: {dup_summary.get('current_duplicate_count', 0)}")
        
        if hasattr(report, 'outlier_analysis') and report.outlier_analysis.get('summary'):
            outlier_summary = report.outlier_analysis['summary']
            self.logger.info(f"  Current outliers: {outlier_summary.get('current_outlier_count', 0)}")
    
    def save_comparison_reports(self, comparison_results: Dict[str, Any]) -> Dict[str, str]:
        """Save comparison reports to files"""
        output_config = self.config.get_output_config()
        reports_folder = output_config['reports_folder']
        
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
            
            saved_files['_summary'] = str(summary_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save summary report: {e}")
        
        return saved_files
    
    def extract_duplicate_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract duplicate analysis summary from comparison results"""
        total_current_duplicates = 0
        total_became_duplicates = 0
        total_resolved_duplicates = 0
        
        for report_data in comparison_results['comparison_reports'].values():
            if 'duplicate_analysis' in report_data:
                dup_analysis = report_data['duplicate_analysis']
                if 'summary' in dup_analysis:
                    summary = dup_analysis['summary']
                    total_current_duplicates += summary.get('current_duplicate_count', 0)
                    total_became_duplicates += summary.get('total_became_duplicates', 0)
                    total_resolved_duplicates += summary.get('total_resolved_duplicates', 0)
        
        return {
            'total_current_duplicates': total_current_duplicates,
            'total_became_duplicates': total_became_duplicates,
            'total_resolved_duplicates': total_resolved_duplicates
        }
    
    def extract_outlier_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract outlier analysis summary from comparison results"""
        total_current_outliers = 0
        total_became_outliers = 0
        total_resolved_outliers = 0
        
        for report_data in comparison_results['comparison_reports'].values():
            if 'outlier_analysis' in report_data:
                outlier_analysis = report_data['outlier_analysis']
                if 'summary' in outlier_analysis:
                    summary = outlier_analysis['summary']
                    total_current_outliers += summary.get('current_outlier_count', 0)
                    total_became_outliers += summary.get('total_became_outliers', 0)
                    total_resolved_outliers += summary.get('total_resolved_outliers', 0)
        
        return {
            'total_current_outliers': total_current_outliers,
            'total_became_outliers': total_became_outliers,
            'total_resolved_outliers': total_resolved_outliers
        }