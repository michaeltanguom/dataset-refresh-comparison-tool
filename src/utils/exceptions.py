"""
Custom exceptions for the dataset comparison pipeline
Centralised exception handling for better error management
"""


class DatasetComparisonError(Exception):
    """Base exception for all dataset comparison pipeline errors"""
    pass


class ConfigurationError(DatasetComparisonError):
    """Raised when configuration is invalid or missing"""
    pass


class ValidationError(DatasetComparisonError):
    """Raised when data validation fails"""
    def __init__(self, message: str, column: str = None, sample_values: list = None):
        super().__init__(message)
        self.column = column
        self.sample_values = sample_values


class ExtractionError(DatasetComparisonError):
    """Raised when data extraction fails"""
    def __init__(self, message: str, file_path: str = None, sheet_name: str = None):
        super().__init__(message)
        self.file_path = file_path
        self.sheet_name = sheet_name


class NormalisationError(DatasetComparisonError):
    """Raised when data normalisation fails"""
    def __init__(self, message: str, table_name: str = None, missing_columns: list = None):
        super().__init__(message)
        self.table_name = table_name
        self.missing_columns = missing_columns or []


class DatabaseError(DatasetComparisonError):
    """Raised when database operations fail"""
    def __init__(self, message: str, table_name: str = None, query: str = None):
        super().__init__(message)
        self.table_name = table_name
        self.query = query


class DataQualityError(DatasetComparisonError):
    """Raised when data quality issues are found"""
    def __init__(self, message: str, table_name: str = None, affected_rows: int = None):
        super().__init__(message)
        self.table_name = table_name
        self.affected_rows = affected_rows


class ComparisonError(DatasetComparisonError):
    """Raised when comparison logic fails"""
    def __init__(self, message: str, dataset_1: str = None, dataset_2: str = None):
        super().__init__(message)
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2


class PipelineError(DatasetComparisonError):
    """Raised when pipeline orchestration fails"""
    def __init__(self, message: str, step: str = None, original_error: Exception = None):
        super().__init__(message)
        self.step = step
        self.original_error = original_error
