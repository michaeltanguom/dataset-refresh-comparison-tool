"""
Extract-specific test fixtures and factories
Provides sample data, mock file structures, and utilities for extraction testing
"""
import pytest
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock
from datetime import datetime

from src.extract.base_extract import DatasetMetadata


class ExtractDataFactory:
    """Factory for creating test data for extraction testing"""
    
    @staticmethod
    def create_sample_excel_dataframe() -> pd.DataFrame:
        """Create sample DataFrame matching expected Excel structure"""
        return pd.DataFrame({
            'Name': [
                'Dr. John Smith',
                'Prof. Jane Doe', 
                'Dr. Bob Wilson',
                'Prof. Alice Johnson',
                'Dr. Charlie Brown'
            ],
            'Times Cited': [1250, 890, 2100, 750, 1500],
            'Highly Cited Papers': [15, 8, 25, 12, 18],
            'ESI Field': [
                'Engineering',
                'computer science', 
                'Computer Science',
                'engineering',
                'Mathematics'
            ],
            '% Docs Cited': [85.5, 72.3, 91.2, 88.7, 76.4],
            'Indicative Cross-Field Score': [1.2, 0.8, 1.8, 1.0, 1.4],
            'Hot Papers': [2, 1, 4, 3, 2],
            'Affiliation': [
                'University A',
                'University B', 
                'University C',
                'University D',
                'University E'
            ]
        })
    
    @staticmethod
    def create_excel_dataframe_with_nulls() -> pd.DataFrame:
        """Create DataFrame with null values for testing data cleaning"""
        return pd.DataFrame({
            'Name': ['Dr. Valid Name', None, '', 'nan', 'Dr. Another Valid'],
            'Times Cited': [1000, 500, 750, 1200, 900],
            'Highly Cited Papers': [10, 5, 8, 12, 7],
            'ESI Field': ['Engineering', 'Mathematics', 'Computer Science', 'Physics', 'Chemistry']
        })
    
    @staticmethod
    def create_comparison_report_json() -> Dict[str, Any]:
        """Create sample JSON comparison report"""
        return {
            "comparison_id": "test_comparison_2024_01_15",
            "metadata": {
                "dataset_1": {
                    "period": "feb",
                    "total_researchers": 100,
                    "processed_timestamp": "2024-01-15T10:30:00"
                },
                "dataset_2": {
                    "period": "july", 
                    "total_researchers": 105,
                    "processed_timestamp": "2024-01-15T10:35:00"
                },
                "comparison_timestamp": "2024-01-15T10:40:00"
            },
            "summary_statistics": {
                "total_researchers_compared": 95,
                "researchers_with_changes": 25,
                "researchers_unchanged": 70,
                "researchers_only_in_dataset_1": 5,
                "researchers_only_in_dataset_2": 10,
                "percentage_with_changes": 26.3
            },
            "researcher_changes": [
                {
                    "name": "Dr. John Smith",
                    "changes": {
                        "times_cited": {"old": 1000, "new": 1200, "change": 200, "percentage_change": 20.0},
                        "highly_cited_papers": {"old": 10, "new": 12, "change": 2, "percentage_change": 20.0}
                    }
                }
            ],
            "researchers_unchanged": [
                {"name": "Dr. Bob Wilson", "times_cited": 2100, "highly_cited_papers": 25}
            ],
            "researchers_only_in_dataset_1": [
                {"name": "Dr. Old Researcher", "times_cited": 500, "highly_cited_papers": 5}
            ],
            "researchers_only_in_dataset_2": [
                {"name": "Dr. New Researcher", "times_cited": 600, "highly_cited_papers": 6}
            ]
        }


class MockFileSystemFactory:
    """Factory for creating mock file system structures"""
    
    @staticmethod
    def create_excel_file_structure(base_dir: Path) -> Dict[str, Path]:
        """Create sample Excel file structure for testing"""
        feb_dir = base_dir / "feb"
        july_dir = base_dir / "july" 
        feb_dir.mkdir(parents=True, exist_ok=True)
        july_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        files['feb_cs'] = feb_dir / "computer_science.xlsx"
        files['feb_eng'] = feb_dir / "engineering.xlsx"
        files['july_cs'] = july_dir / "computer_science.xlsx"
        files['july_eng'] = july_dir / "engineering.xlsx"
        files['temp_file'] = feb_dir / "~$computer_science.xlsx"
        
        return files
    
    @staticmethod
    def create_json_report_structure(base_dir: Path) -> Dict[str, Path]:
        """Create sample JSON report structure for testing"""
        highly_cited_dir = base_dir / "comparison_reports" / "highly_cited_only"
        incites_dir = base_dir / "comparison_reports" / "incites_researchers"
        highly_cited_dir.mkdir(parents=True, exist_ok=True)
        incites_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        files['hc_cs_report'] = highly_cited_dir / "computer_science_highly_cited_only_comparison_report.json"
        files['hc_eng_report'] = highly_cited_dir / "engineering_highly_cited_only_comparison_report.json"
        files['inc_cs_report'] = incites_dir / "computer_science_incites_researchers_comparison_report.json"
        files['inc_eng_report'] = incites_dir / "engineering_incites_researchers_comparison_report.json"
        
        return files


@pytest.fixture
def extract_data_factory():
    """Provide extract data factory for tests"""
    return ExtractDataFactory

@pytest.fixture
def mock_filesystem_factory():
    """Provide mock filesystem factory for tests"""
    return MockFileSystemFactory

@pytest.fixture
def sample_excel_dataframe(extract_data_factory):
    """Provide sample Excel DataFrame"""
    return extract_data_factory.create_sample_excel_dataframe()

@pytest.fixture
def excel_dataframe_with_nulls(extract_data_factory):
    """Provide Excel DataFrame with null values"""
    return extract_data_factory.create_excel_dataframe_with_nulls()

@pytest.fixture
def sample_comparison_report(extract_data_factory):
    """Provide sample JSON comparison report"""
    return extract_data_factory.create_comparison_report_json()

@pytest.fixture
def temp_extract_workspace(test_temp_dir):
    """Create temporary workspace for extraction testing"""
    extract_workspace = test_temp_dir / "extract_tests"
    extract_workspace.mkdir(exist_ok=True)
    return extract_workspace

@pytest.fixture
def excel_test_files(temp_extract_workspace, mock_filesystem_factory, sample_excel_dataframe):
    """Create actual Excel test files for integration testing"""
    file_structure = mock_filesystem_factory.create_excel_file_structure(temp_extract_workspace)
    
    # Create actual Excel files with test data
    for file_key, file_path in file_structure.items():
        if file_key.startswith('temp_'):
            file_path.touch()
        elif file_path.suffix == '.xlsx':
            with pd.ExcelWriter(str(file_path), engine='openpyxl') as writer:
                highly_cited_df = sample_excel_dataframe.copy()
                highly_cited_df.to_excel(writer, sheet_name='Highly Cited only', index=False)
                
                incites_df = sample_excel_dataframe.copy()
                incites_df['Additional Column'] = 'Extra data'
                incites_df.to_excel(writer, sheet_name='Incites Researchers', index=False)
    
    return file_structure

@pytest.fixture
def json_test_files(temp_extract_workspace, mock_filesystem_factory, sample_comparison_report):
    """Create actual JSON test files for integration testing"""
    file_structure = mock_filesystem_factory.create_json_report_structure(temp_extract_workspace)
    
    for file_key, file_path in file_structure.items():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = sample_comparison_report.copy()
        if 'cs' in file_key:
            report_data['comparison_id'] = f"computer_science_{file_key}"
        elif 'eng' in file_key:
            report_data['comparison_id'] = f"engineering_{file_key}"
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
    
    return file_structure

@pytest.fixture
def sample_dataset_metadata():
    """Provide sample dataset metadata for testing"""
    return DatasetMetadata(
        source_file="/test/path/computer_science.xlsx",
        subject="computer_science",
        period="feb",
        sheet_name="Highly Cited only",
        normalised_sheet_name="highly_cited_only",
        table_name="computer_science_feb_highly_cited_only",
        row_count=100,
        columns_mapped={
            'name': 'Name',
            'times_cited': 'Times Cited',
            'highly_cited_papers': 'Highly Cited Papers'
        },
        processing_timestamp="2024-01-15T10:30:45.123456",
        extraction_duration_seconds=2.5,
        normalisation_duration_seconds=1.2
    )

@pytest.fixture  
def esi_field_test_cases():
    """Provide ESI field normalisation test cases"""
    return {
        'valid_mappings': [
            ('agricultural sciences', 'Agricultural Sciences'),
            ('biology_biochemistry', 'Biology Biochemistry'),
            ('chemistry', 'Chemistry'),
            ('clinical medicine', 'Clinical Medicine'),
            ('computer science', 'Computer Science'),
            ('engineering', 'Engineering'),
            ('physics', 'Physics')
        ],
        'case_insensitive': [
            ('CHEMISTRY', 'Chemistry'),
            ('Computer Science', 'Computer Science'),
            ('ENGINEERING', 'Engineering')
        ],
        'whitespace_handling': [
            ('  chemistry  ', 'Chemistry'),
            ('\tcomputer science\n', 'Computer Science'),
            ('engineering ', 'Engineering')
        ],
        'unknown_fields': [
            ('Unknown Field', 'Unknown Field'),
            ('Invalid ESI', 'Invalid ESI'),
            ('', ''),
            ('   ', '')
        ]
    }

@pytest.fixture
def column_mapping_test_scenarios():
    """Provide various column mapping scenarios for testing"""
    return {
        'perfect_match': {
            'input_columns': ['Name', 'Times Cited', 'Highly Cited Papers', 'ESI Field'],
            'expected_mapped': ['name', 'times_cited', 'highly_cited_papers', 'esi_field'],
            'expected_unmapped': []
        },
        'variant_match': {
            'input_columns': ['Full Name', 'Citation Count', 'HC Papers', 'Research Field'],  
            'expected_mapped': ['name', 'times_cited', 'highly_cited_papers', 'esi_field'],
            'expected_unmapped': []
        },
        'partial_match': {
            'input_columns': ['Name', 'Times Cited', 'Unknown Column', 'ESI Field'],
            'expected_mapped': ['name', 'times_cited', 'esi_field'],
            'expected_unmapped': ['Unknown Column']
        },
        'normalised_match': {
            'input_columns': ['  Name  ', 'Times-Cited', 'Highly_Cited_Papers', 'ESI.Field'],
            'expected_mapped': ['name', 'times_cited', 'highly_cited_papers', 'esi_field'],
            'expected_unmapped': []
        }
    }