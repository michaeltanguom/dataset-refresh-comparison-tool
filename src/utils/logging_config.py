"""
Centralised logging configuration for the dataset comparison pipeline
Single source of truth for all logging setup
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Set up logging configuration based on config settings
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Configured logger instance
    """
    # Get logging configuration with defaults
    log_level = config.get('level', 'INFO').upper()
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_to_file = config.get('log_to_file', True)
    log_file_path = config.get('log_file_path', 'comparison_pipeline.log')
    max_file_size_mb = config.get('max_file_size_mb', 10)
    backup_count = config.get('backup_count', 5)
    
    # Create logger
    logger = logging.getLogger('dataset_comparison')
    logger.setLevel(getattr(logging, log_level))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if log_to_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Rotating file handler
            max_bytes = max_file_size_mb * 1024 * 1024  # Convert MB to bytes
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setLevel(getattr(logging, log_level))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file_path}")
            
        except Exception as e:
            logger.warning(f"Failed to set up file logging: {e}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get logger instance for a specific module
    
    Args:
        name: Module name (optional)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f'dataset_comparison.{name}')
    else:
        return logging.getLogger('dataset_comparison')


def log_function_entry(logger: logging.Logger, function_name: str, **kwargs) -> None:
    """
    Log function entry with parameters
    
    Args:
        logger: Logger instance
        function_name: Name of function being entered
        **kwargs: Function parameters to log
    """
    if kwargs:
        params = ', '.join([f"{k}={v}" for k, v in kwargs.items() if not str(k).startswith('_')])
        logger.debug(f"Entering {function_name}({params})")
    else:
        logger.debug(f"Entering {function_name}()")


def log_function_exit(logger: logging.Logger, function_name: str, result=None) -> None:
    """
    Log function exit with optional result
    
    Args:
        logger: Logger instance  
        function_name: Name of function being exited
        result: Function result to log (optional)
    """
    if result is not None:
        logger.debug(f"Exiting {function_name} with result: {type(result).__name__}")
    else:
        logger.debug(f"Exiting {function_name}")


def log_dataframe_summary(logger: logging.Logger, df, name: str) -> None:
    """
    Log summary information about a DataFrame
    
    Args:
        logger: Logger instance
        df: pandas DataFrame
        name: Name for the DataFrame in logs
    """
    import pandas as pd
    
    if df is None:
        logger.warning(f"DataFrame '{name}' is None")
        return
    
    if df.empty:
        logger.warning(f"DataFrame '{name}' is empty")
        return
    
    logger.info(f"DataFrame '{name}' summary:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Log null counts for important columns
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.info(f"  Null values: {null_counts[null_counts > 0].to_dict()}")


def log_processing_step(logger: logging.Logger, step_name: str, details: str = None) -> None:
    """
    Log processing step with consistent formatting
    
    Args:
        logger: Logger instance
        step_name: Name of processing step
        details: Additional details (optional)
    """
    separator = "=" * 50
    logger.info(separator)
    logger.info(f"PROCESSING STEP: {step_name.upper()}")
    if details:
        logger.info(f"Details: {details}")
    logger.info(separator)


def log_validation_result(logger: logging.Logger, validation_name: str, 
                         passed: bool, details: str = None) -> None:
    """
    Log validation result with consistent formatting
    
    Args:
        logger: Logger instance
        validation_name: Name of validation
        passed: Whether validation passed
        details: Additional details (optional)
    """
    status = "PASSED" if passed else "FAILED"
    level = logging.INFO if passed else logging.ERROR
    
    message = f"Validation '{validation_name}': {status}"
    if details:
        message += f" - {details}"
    
    logger.log(level, message)


def log_performance_metric(logger: logging.Logger, operation: str, 
                          duration_seconds: float, records_processed: int = None) -> None:
    """
    Log performance metrics for operations
    
    Args:
        logger: Logger instance
        operation: Name of operation
        duration_seconds: Duration in seconds
        records_processed: Number of records processed (optional)
    """
    message = f"Performance - {operation}: {duration_seconds:.2f}s"
    
    if records_processed is not None:
        rate = records_processed / duration_seconds if duration_seconds > 0 else 0
        message += f" ({records_processed:,} records, {rate:.0f} records/sec)"
    
    logger.info(message)
