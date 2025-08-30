
Testing framework:
Test Naming Convention
pythondef test_[method_name]_[scenario]_[expected_outcome]():
    """
    Examples:
    - test_extract_files_with_valid_excel_returns_dataframes()
    - test_normalise_esi_fields_with_invalid_input_raises_error()
    - test_compare_datasets_with_empty_data_returns_empty_results()
    """

Test Isolation
Each test should be independent
Use fixtures for setup/teardown
Mock external dependencies
Clean database state between tests

Arrange, Act, Assert (AAA) Pattern
def test_function():
    # Arrange - Set up test data and mocks
    config = create_test_config()
    
    # Act - Execute the code under test
    result = function_under_test(config)
    
    # Assert - Verify the results
    assert result.is_valid
    assert len(result.data) == expected_count

• Configuration Management (test_config/) - Foundation for everything else
• Common Utilities (test_utils/) - Used throughout the pipeline
• Data Extraction (test_extract/) - Entry point of your pipeline
• Data Transformation (test_transform/) - Your core business logic
• Database Operations (test_load/) - Critical for data integrity
• HTML Generation (test_html_generator/) - User-facing output

With attention to test
• Database state management
• DataFrame testing patterns
• Configuration validation
• Error condition testing
• Performance considerations

Folder structure:
tests/
├── conftest.py                    # Global fixtures and configuration
├── fixtures/                     # Test data and mock objects
│   ├── __init__.py
│   ├── sample_data.py            # Sample DataFrames and datasets
│   ├── config_fixtures.py        # Configuration test data
│   └── mock_objects.py           # Mock external dependencies
├── unit/                         # Unit tests (80% coverage target)
│   ├── __init__.py
│   ├── test_config/              # Configuration module tests
│   │   ├── test_config_manager.py
│   │   └── test_config_validator.py
│   ├── test_extract/             # Extraction module tests
│   │   ├── test_base_extract.py
│   │   ├── test_excel_extractor.py
│   │   ├── test_json_extractor.py
│   │   └── test_light_transform.py
│   ├── test_load/                # Loading module tests
│   │   └── test_load_duckdb.py
│   ├── test_transform/           # Transformation module tests -- missing...
│   │   ├── test_clean_duckdb_tables.py
│   │   └── test_compare_datasets.py
│   ├── test_html_generator/      # HTML generation tests -- missing...
│   │   ├── test_templates/
│   │   └── test_renderers/
│   └── test_utils/               # Utility module tests
│       ├── test_common.py
│       ├── test_database_manager.py
│       └── test_exceptions.py
├── component/                    # Component integration tests based on Prefect
│   ├── test_etl_integration.py
│   └── test_html_pipeline.py
└── integration/                  # End-to-end tests
    └── test_full_pipeline.py

Unit tests left to create:
test_html_generator
component
integration