# Researcher Dataset Refresh Comparison Tool

A robust Python-based EtLT (Extract, light transform, Load, Transform) pipeline for comparing research datasets across different time periods, with automated report generation and interactive HTML dashboards.

## Overview

This tool is designed to analyse changes in research performance data between different time periods (e.g., February 2025 vs July 2025). It processes Excel files containing researcher data, performs comprehensive comparisons, and generates detailed reports with interactive visualisations.

### Key Features

- **Multi-format Data Processing**: Extract from Excel files with multiple sheets
- **Automated Data Comparison**: Compare researcher metrics across time periods
- **Interactive HTML Dashboards**: Generate responsive web-based reports
- **Modular Architecture**: Composition-based design for maximum flexibility
- **Prefect Orchestration**: Proper task and workflow management
- **ESI Field Normalisation**: Standardisation of research field names
- **DuckDB Integration**: High-performance analytical database powering transformation logic
- **Comprehensive Metrics**: Track changes in citations, papers, and cross-field scores

## Architecture

The system follows modern software engineering principles with a clear separation of concerns:

```
src/
├── config/           # Configuration management and validation
├── extract/          # Data extraction and light transformation
├── load/            # Database loading operations
├── transform/       # Data cleaning and comparison logic
├── html_generator/  # Interactive dashboard generation
└── utils/           # Shared utilities and database management
```

### Design Patterns Used

- **Composition over Inheritance**: Flexible component assembly
- **Factory Pattern**: Multiple data source support
- **Strategy Pattern**: Configurable processing algorithms
- **Observer Pattern**: Pipeline monitoring and alerting
- **Template Method**: Consistent ETL workflows

## Quick Start

### Prerequisites

- Python 3.9+
- Required packages: `pip install -r requirements.txt`

### Virtual Environment set up

Create and activate a virtual environment
1. **Create virtual environment**
    ```bash
    python -m venv venv
    ```

2a. **Activate virtual environment**
    **On macOS/Linux:**
    ```
    source venv/bin/activate
    ```

2b. **On Windows:**
    ```
    venv\Scripts\activate
    ```

3. **Install dependencies**
    **Upgrade pip**
    ```
    python -m pip install --upgrade pip
    ```

4. **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```

### Basic Usage

1. **Configure your pipeline**:
   ```yaml
   # config/comparison_config.yaml
   data_sources:
     dataset_1:
       folder: "Feb2025"
       period_name: "feb"
     dataset_2:
       folder: "July2025"
       period_name: "july"
   ```

2. **Start the Prefect server in the terminal**
   ```bash
   prefect server start
   ```

3. **Run the complete pipeline**:
   ```bash
   python prefect_orchestration.py --config config/comparison_config.yaml --run
   ```

4. **Generate HTML dashboards**:
   ```bash
   python prefect_orchestration.py --config config/comparison_config.yaml --run --generate-html
   ```

## Detailed Usage

### Pipeline Orchestration
#### The tool is orchestrated with Prefect:
```bash
# Complete pipeline with monitoring
python prefect_orchestration.py --config config/comparison_config.yaml --run

# With HTML dashboard generation
python prefect_orchestration.py --config config/comparison_config.yaml --run --generate-html

# Include legacy JSON extraction
python prefect_orchestration.py --config config/comparison_config.yaml --run --include-json --generate-html
```

#### Development Mode
```bash
# Start Prefect server for development
python prefect_orchestration.py --serve
```

### Configuration

The system uses YAML configuration files for maximum flexibility:

#### Main Configuration (`config/comparison_config.yaml`)

```yaml
data_sources:
  dataset_1:
    folder: "Feb2025"
    period_name: "feb"
  dataset_2:
    folder: "July2025"
    period_name: "july"

sheets_to_process:
  - "Highly Cited only"
  - "Incites Researchers"

column_mapping:
  name: "Name"
  times_cited: "Times Cited"
  highly_cited_papers: "Highly Cited Papers"
  esi_field: "ESI Field"
  # ... additional mappings

validation_rules:
  times_cited:
    min: 50
    max: 100000
    required: true
  # ... additional rules

output:
  reports_folder: "comparison_reports"

html_generation:
  enabled: true
  config_path: "config/html_generator_config.yaml"
```

#### HTML Generator Configuration (`config/html_generator_config.yaml`)

```yaml
html_generation:
  input_source: "comparison_reports"
  output_directory: "html_reports"
  template_mapping:
    highly_cited_only: "research_dashboard"
    incites_researchers: "research_dashboard"

templates:
  research_dashboard:
    class: "ResearchDashboardTemplate"
    config:
      title_format: "{dataset_type} - Research Performance Reports"
      colour_scheme: "blue_gradient"
```

## 🔧 Pipeline Components

### 1. Extract Phase
- **Excel Data Extractor**: Processes multiple sheets from Excel files
- **Data Normalizer**: Standardises column names and ESI field values
- **Validation**: Ensures data quality and completeness

### 2. Load Phase
- **DuckDB Integration**: High-performance analytical database powering transformations and loading
- **Schema Management**: Configurable table schemas
- **Data Validation**: Post-load integrity checks

### 3. Transform Phase
- **Data Cleaning**: Handle nulls, duplicates and identifies outliers using IQR
- **Comparison Engine**: Identify changes between time periods
- **Report Generation**: JSON-based comparison reports

### 4. Dashboard Generation
- **Template System**: Flexible HTML template engine
- **Interactive Features**: Sorting, filtering, and search

## Output Reports

The tool generates several types of outputs:

### 1. JSON Comparison Reports
```
comparison_reports/
├── highly_cited_only/
│   ├── agricultural_sciences_comparison_report.json
│   ├── computer_science_comparison_report.json
│   └── ...
└── incites_researchers/
    ├── biology_biochemistry_comparison_report.json
    └── ...
```

### 2. Interactive HTML Dashboards
```
html_reports/
├── highly_cited_only_25_subjects_research_dashboard_dashboard_20250121_143022.html
├── incites_researchers_25_subjects_research_dashboard_dashboard_20250121_143045.html
└── ...
```

### 3. Summary Statistics
- Total researchers compared
- Retention rates between periods
- Change rates and significant variations
- Cross-field researcher analysis

## Advanced Features

### ESI Field Normalization

The system automatically standardises research field names to canonical formats:

```python
# Automatic normalization
"agricultural sciences" → "Agricultural Sciences"
"computer science" → "Computer Science"
"biology_biochemistry" → "Biology Biochemistry"
```

### Cross-Field Analysis

Identifies researchers appearing in multiple ESI fields with sophisticated scoring:

- **Cross-Field Researchers**: Total score ≥ 1.0
- **Near Cross-Field**: Score 0.85-0.99
- **Approaching Cross-Field**: Score 0.70-0.84
- **Multidisciplinary**: Score < 0.70

### Performance Monitoring

Built-in performance tracking and metrics:

```python
# Automatic performance logging
⏱️  Extraction: 2.3s (1,234 records)
⏱️  Normalisation: 1.1s (ESI fields: 89 normalised)
⏱️  Loading: 0.8s (2 tables created)
⏱️  Cleaning: 1.5s (validation passed)
⏱️  Comparison: 3.2s (5 comparisons completed)
```

## Database Schema

The tool uses DuckDB with configurable schemas defined in SQL files:

```sql
-- schema/hcr_default.sql
CREATE TABLE IF NOT EXISTS "{table_name}" (
    name VARCHAR NOT NULL,
    times_cited INTEGER,
    highly_cited_papers INTEGER,
    hot_papers INTEGER,
    esi_field VARCHAR NOT NULL,
    indicative_cross_field_score DOUBLE,
    -- ... additional columns
);
```

## Testing and Validation

### Configuration Validation
```bash
# Validate configuration before running
python prefect_orchestration.py --config config/comparison_config.yaml --validate-only
```

### Built-in Data Quality Checks
- **Null Value Handling**: Configurable strategies (fail/skip/default)
- **Duplicate Detection**: Based on name and ESI field
- **Outlier Detection**: IQR and Z-score methods
- **Cross-field Consistency**: Logical validation rules

### Sample Data Verification
- Automatic logging of sample records
- Data type validation
- Range checking for numeric fields

## Folder Structure

```
dataset-refresh-comparison-tool/
├── config/
│   ├── comparison_config.yaml
│   └── html_generator_config.yaml
├── schema/
│   └── hcr_default.sql
├── src/
│   ├── config/
│   │   ├── config_manager.py
│   │   └── config_validator.py
│   ├── extract/
│   │   ├── excel_extractor.py
│   │   ├── json_extractor.py
│   │   └── light_transform.py
│   ├── load/
│   │   └── load_duckdb.py
│   ├── transform/
│   │   ├── clean_duckdb_tables.py
│   │   └── compare_datasets.py
│   ├── html_generator/
│   │   ├── templates/
│   │   └── renderers/
│   └── utils/
│       ├── database_manager.py
│       ├── exceptions.py
│       └── logging_config.py
├── prefect_orchestration.py
├── requirements.txt
└── README.md
```

## Configuration Options

### Data Sources
- **Multiple Formats**: Excel, JSON (extensible to CSV, databases)
- **Flexible Mapping**: Configurable column name mappings
- **Period Management**: Support for any number of time periods

### Processing Options
- **Sheet Selection**: Choose which Excel sheets to process
- **Cleaning Strategies**: Handle nulls, duplicates, outliers
- **Validation Rules**: Customisable data quality checks

### Output Customization
- **Report Formats**: JSON, HTML dashboards
- **Template Selection**: Multiple dashboard templates
- **Styling Options**: Configurable themes and layouts

## Error Handling

The system includes comprehensive error handling:

- **Configuration Errors**: Invalid YAML, missing files
- **Data Errors**: Schema mismatches, validation failures
- **Processing Errors**: Extraction, transformation failures
- **Output Errors**: Report generation, file system issues

All errors are logged with detailed context and suggested resolutions.

## 📈 Design Considerations

### Scalability
- **Modular Design**: Easy to extend with new data sources
- **Configuration-Driven**: No code changes for new analyses utilising the default schema
- **Resource Monitoring**: Built-in performance metrics

## Extension

This tool is built with extensibility in mind:

### Adding New Data Sources
1. Implement the `DataExtractor` interface
2. Add factory method for new source type
3. Update configuration schema

### Custom Transformations
1. Create new strategy classes
2. Register with the transformation engine
3. Add configuration options

### New Dashboard Templates
1. Extend `BaseTemplate` class
2. Implement template-specific logic
3. Register with template factory
