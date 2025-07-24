"""
Unit tests for custom exception classes
Tests exception hierarchy and custom attributes
"""
import pytest

from src.utils.exceptions import (
    DatasetComparisonError,
    ConfigurationError,
    ValidationError,
    ExtractionError,
    NormalisationError,
    DatabaseError,
    DataQualityError,
    ComparisonError,
    PipelineError,
    HtmlGenerationError
)


class TestDatasetComparisonError:
    """Test suite for base exception class"""
    
    def test_dataset_comparison_error_is_base_exception(self):
        """Test that DatasetComparisonError is the base exception"""
        # Act
        error = DatasetComparisonError("Test message")
        
        # Assert
        assert isinstance(error, Exception)
        assert str(error) == "Test message"
    
    def test_dataset_comparison_error_inheritance(self):
        """Test that all custom exceptions inherit from DatasetComparisonError"""
        # Arrange
        exception_classes = [
            ConfigurationError,
            ValidationError,
            ExtractionError,
            NormalisationError,
            DatabaseError,
            DataQualityError,
            ComparisonError,
            PipelineError,
            HtmlGenerationError
        ]
        
        # Act & Assert
        for exception_class in exception_classes:
            error = exception_class("Test message")
            assert isinstance(error, DatasetComparisonError)
            assert isinstance(error, Exception)


class TestConfigurationError:
    """Test suite for ConfigurationError"""
    
    def test_configuration_error_basic_usage(self):
        """Test basic ConfigurationError usage"""
        # Act
        error = ConfigurationError("Configuration is invalid")
        
        # Assert
        assert isinstance(error, DatasetComparisonError)
        assert str(error) == "Configuration is invalid"
    
    def test_configuration_error_in_exception_handling(self):
        """Test ConfigurationError in exception handling context"""
        # Act & Assert
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Missing required section")
        
        assert "Missing required section" in str(exc_info.value)


class TestValidationError:
    """Test suite for ValidationError with additional attributes"""
    
    def test_validation_error_with_basic_message(self):
        """Test ValidationError with just a message"""
        # Act
        error = ValidationError("Validation failed")
        
        # Assert
        assert str(error) == "Validation failed"
        assert error.column is None
        assert error.sample_values is None
    
    def test_validation_error_with_column_info(self):
        """Test ValidationError with column information"""
        # Act
        error = ValidationError("Invalid values in column", column="test_column")
        
        # Assert
        assert str(error) == "Invalid values in column"
        assert error.column == "test_column"
        assert error.sample_values is None
    
    def test_validation_error_with_sample_values(self):
        """Test ValidationError with sample values"""
        # Arrange
        sample_values = ["invalid1", "invalid2", "invalid3"]
        
        # Act
        error = ValidationError(
            "Invalid values found", 
            column="test_column", 
            sample_values=sample_values
        )
        
        # Assert
        assert error.column == "test_column"
        assert error.sample_values == sample_values
        assert len(error.sample_values) == 3
    
    def test_validation_error_with_empty_sample_values(self):
        """Test ValidationError with empty sample values list"""
        # Act
        error = ValidationError("No samples", sample_values=[])
        
        # Assert
        assert error.sample_values == []


class TestExtractionError:
    """Test suite for ExtractionError with file context"""
    
    def test_extraction_error_with_basic_message(self):
        """Test ExtractionError with just a message"""
        # Act
        error = ExtractionError("Failed to extract data")
        
        # Assert
        assert str(error) == "Failed to extract data"
        assert error.file_path is None
        assert error.sheet_name is None
    
    def test_extraction_error_with_file_path(self):
        """Test ExtractionError with file path information"""
        # Act
        error = ExtractionError("File not found", file_path="/path/to/file.xlsx")
        
        # Assert
        assert error.file_path == "/path/to/file.xlsx"
        assert error.sheet_name is None
    
    def test_extraction_error_with_file_and_sheet_info(self):
        """Test ExtractionError with both file and sheet information"""
        # Act
        error = ExtractionError(
            "Sheet not found", 
            file_path="/path/to/file.xlsx", 
            sheet_name="Missing Sheet"
        )
        
        # Assert
        assert error.file_path == "/path/to/file.xlsx"
        assert error.sheet_name == "Missing Sheet"


class TestNormalisationError:
    """Test suite for NormalisationError with table context"""
    
    def test_normalisation_error_with_basic_message(self):
        """Test NormalisationError with just a message"""
        # Act
        error = NormalisationError("Normalisation failed")
        
        # Assert
        assert str(error) == "Normalisation failed"
        assert error.table_name is None
        assert error.missing_columns == []
    
    def test_normalisation_error_with_table_name(self):
        """Test NormalisationError with table name"""
        # Act
        error = NormalisationError("Table invalid", table_name="test_table")
        
        # Assert
        assert error.table_name == "test_table"
        assert error.missing_columns == []
    
    def test_normalisation_error_with_missing_columns(self):
        """Test NormalisationError with missing columns information"""
        # Arrange
        missing_cols = ["column1", "column2", "column3"]
        
        # Act
        error = NormalisationError(
            "Missing columns", 
            table_name="test_table", 
            missing_columns=missing_cols
        )
        
        # Assert
        assert error.table_name == "test_table"
        assert error.missing_columns == missing_cols
        assert len(error.missing_columns) == 3
    
    def test_normalisation_error_with_none_missing_columns_defaults_to_empty_list(self):
        """Test that None missing_columns defaults to empty list"""
        # Act
        error = NormalisationError("Test", missing_columns=None)
        
        # Assert
        assert error.missing_columns == []


class TestDatabaseError:
    """Test suite for DatabaseError with database context"""
    
    def test_database_error_with_basic_message(self):
        """Test DatabaseError with just a message"""
        # Act
        error = DatabaseError("Database operation failed")
        
        # Assert
        assert str(error) == "Database operation failed"
        assert error.table_name is None
        assert error.query is None
    
    def test_database_error_with_table_name(self):
        """Test DatabaseError with table name"""
        # Act
        error = DatabaseError("Table error", table_name="problematic_table")
        
        # Assert
        assert error.table_name == "problematic_table"
        assert error.query is None
    
    def test_database_error_with_query_context(self):
        """Test DatabaseError with query information"""
        # Arrange
        failed_query = "SELECT * FROM non_existent_table"
        
        # Act
        error = DatabaseError("Query failed", query=failed_query)
        
        # Assert
        assert error.query == failed_query
        assert error.table_name is None
    
    def test_database_error_with_full_context(self):
        """Test DatabaseError with all context information"""
        # Act
        error = DatabaseError(
            "Complete failure", 
            table_name="test_table", 
            query="SELECT * FROM test_table"
        )
        
        # Assert
        assert error.table_name == "test_table"
        assert error.query == "SELECT * FROM test_table"


class TestDataQualityError:
    """Test suite for DataQualityError"""
    
    def test_data_quality_error_with_basic_message(self):
        """Test DataQualityError with just a message"""
        # Act
        error = DataQualityError("Data quality issues found")
        
        # Assert
        assert str(error) == "Data quality issues found"
        assert error.table_name is None
        assert error.affected_rows is None
    
    def test_data_quality_error_with_affected_rows(self):
        """Test DataQualityError with affected rows count"""
        # Act
        error = DataQualityError(
            "Invalid data found", 
            table_name="data_table", 
            affected_rows=150
        )
        
        # Assert
        assert error.table_name == "data_table"
        assert error.affected_rows == 150


class TestComparisonError:
    """Test suite for ComparisonError"""
    
    def test_comparison_error_with_basic_message(self):
        """Test ComparisonError with just a message"""
        # Act
        error = ComparisonError("Comparison failed")
        
        # Assert
        assert str(error) == "Comparison failed"
        assert error.dataset_1 is None
        assert error.dataset_2 is None
    
    def test_comparison_error_with_dataset_info(self):
        """Test ComparisonError with dataset information"""
        # Act
        error = ComparisonError(
            "Datasets incompatible", 
            dataset_1="february_data", 
            dataset_2="july_data"
        )
        
        # Assert
        assert error.dataset_1 == "february_data"
        assert error.dataset_2 == "july_data"


class TestPipelineError:
    """Test suite for PipelineError with step context"""
    
    def test_pipeline_error_with_basic_message(self):
        """Test PipelineError with just a message"""
        # Act
        error = PipelineError("Pipeline execution failed")
        
        # Assert
        assert str(error) == "Pipeline execution failed"
        assert error.step is None
        assert error.original_error is None
    
    def test_pipeline_error_with_step_info(self):
        """Test PipelineError with step information"""
        # Act
        error = PipelineError("Step failed", step="data_extraction")
        
        # Assert
        assert error.step == "data_extraction"
        assert error.original_error is None
    
    def test_pipeline_error_with_original_error(self):
        """Test PipelineError with original error information"""
        # Arrange
        original_error = ValueError("Original problem")
        
        # Act
        error = PipelineError(
            "Pipeline step failed", 
            step="transformation", 
            original_error=original_error
        )
        
        # Assert
        assert error.step == "transformation"
        assert error.original_error == original_error
        assert isinstance(error.original_error, ValueError)


class TestHtmlGenerationError:
    """Test suite for HtmlGenerationError"""
    
    def test_html_generation_error_with_basic_message(self):
        """Test HtmlGenerationError with just a message"""
        # Act
        error = HtmlGenerationError("HTML generation failed")
        
        # Assert
        assert str(error) == "HTML generation failed"
        assert error.template_name is None
        assert error.dataset_key is None
    
    def test_html_generation_error_with_template_info(self):
        """Test HtmlGenerationError with template information"""
        # Act
        error = HtmlGenerationError(
            "Template not found", 
            template_name="comparison_report.html"
        )
        
        # Assert
        assert error.template_name == "comparison_report.html"
        assert error.dataset_key is None
    
    def test_html_generation_error_with_dataset_context(self):
        """Test HtmlGenerationError with dataset context"""
        # Act
        error = HtmlGenerationError(
            "Data rendering failed", 
            template_name="report.html", 
            dataset_key="highly_cited_papers"
        )
        
        # Assert
        assert error.template_name == "report.html"
        assert error.dataset_key == "highly_cited_papers"


# ========================================
# Exception Chaining and Handling Tests
# ========================================

class TestExceptionChaining:
    """Test suite for exception chaining and handling patterns"""
    
    def test_exception_chaining_preserves_original_error(self):
        """Test that exception chaining preserves original error information"""
        # Arrange
        original_error = ValueError("Original error")
        
        # Act & Assert
        with pytest.raises(PipelineError) as exc_info:
            try:
                raise original_error
            except ValueError as e:
                raise PipelineError("Pipeline failed", original_error=e)
        
        pipeline_error = exc_info.value
        assert pipeline_error.original_error == original_error
        assert isinstance(pipeline_error.original_error, ValueError)
    
    def test_multiple_exception_types_in_context(self):
        """Test handling multiple exception types in the same context"""
        # Arrange
        exception_scenarios = [
            (ConfigurationError, "Config error"),
            (ValidationError, "Validation error"),
            (DatabaseError, "Database error"),
            (ExtractionError, "Extraction error")
        ]
        
        # Act & Assert
        for exception_class, message in exception_scenarios:
            with pytest.raises(DatasetComparisonError):
                raise exception_class(message)
    
    def test_exception_attributes_are_accessible_after_raising(self):
        """Test that custom attributes are accessible after exception is raised"""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError(
                "Validation failed", 
                column="test_col", 
                sample_values=["val1", "val2"]
            )
        
        error = exc_info.value
        assert error.column == "test_col"
        assert error.sample_values == ["val1", "val2"]