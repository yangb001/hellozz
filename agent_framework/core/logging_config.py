"""Logging Configuration - Centralized logging setup for the agent framework.

This module provides a unified logging configuration system that supports:
- Console and file dual output
- Module-specific log files
- Log rotation (configurable size and backup count)
- Separate error log file
- Configuration via config.json

参考：详细设计.md
"""
import os
import logging
import logging.handlers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


# Module-specific logger names and their log file names
MODULE_LOGGERS = {
    "gateway": "gateway.log",
    "core": "session.log",
    "core.session_manager": "session.log",
    "core.event_bus": "session.log",
    "core.config": "session.log",
    "memory": "memory.log",
    "memory.buffer_memory": "memory.log",
    "memory.vector_memory": "memory.log",
    "memory.extractor": "memory.log",
    "memory.manager": "memory.log",
    "planners": "planner.log",
    "planners.react_planner": "planner.log",
    "infrastructure": "llm.log",
    "infrastructure.llm_gateway": "llm.log",
    "infrastructure.openai_llm": "llm.log",
    "infrastructure.ollama_llm": "llm.log",
    "infrastructure.storage": "llm.log",
    "infrastructure.llm_debug": "llm_debug.log",
    "runtime": "app.log",
    "runtime.agent_runtime": "app.log",
}

# Default logging configuration
DEFAULT_CONFIG = {
    "level": "INFO",
    "log_dir": "logs",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
    "console_output": True,
    "file_output": True,
}


@dataclass
class LoggingConfig:
    """Logging configuration data class.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files.
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of backup files to keep.
        console_output: Whether to output logs to console.
        file_output: Whether to output logs to files.
    """
    level: str = "INFO"
    log_dir: str = "logs"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    file_output: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "LoggingConfig":
        """Create LoggingConfig from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            LoggingConfig instance.
        """
        if not data:
            return cls()

        return cls(
            level=data.get("level", DEFAULT_CONFIG["level"]),
            log_dir=data.get("log_dir", DEFAULT_CONFIG["log_dir"]),
            max_bytes=data.get("max_bytes", DEFAULT_CONFIG["max_bytes"]),
            backup_count=data.get("backup_count", DEFAULT_CONFIG["backup_count"]),
            console_output=data.get("console_output", DEFAULT_CONFIG["console_output"]),
            file_output=data.get("file_output", DEFAULT_CONFIG["file_output"]),
        )


def _get_log_level(level_name: str) -> int:
    """Convert log level name to logging level constant.

    Args:
        level_name: Log level name (e.g., "INFO", "DEBUG").

    Returns:
        Logging level constant.
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level_name.upper(), logging.INFO)


def _create_formatter() -> logging.Formatter:
    """Create a standard log formatter.

    Returns:
        logging.Formatter instance.
    """
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def _create_console_handler(config: LoggingConfig) -> Optional[logging.StreamHandler]:
    """Create a console handler.

    Args:
        config: Logging configuration.

    Returns:
        Console handler or None if console output is disabled.
    """
    if not config.console_output:
        return None

    handler = logging.StreamHandler()
    handler.setLevel(_get_log_level(config.level))
    handler.setFormatter(_create_formatter())
    return handler


def _create_file_handler(
    log_dir: str,
    filename: str,
    config: LoggingConfig,
    is_error_handler: bool = False
) -> Optional[logging.handlers.RotatingFileHandler]:
    """Create a rotating file handler.

    Args:
        log_dir: Directory for log files.
        filename: Log file name.
        config: Logging configuration.
        is_error_handler: If True, only log ERROR and above.

    Returns:
        RotatingFileHandler or None if file output is disabled.
    """
    if not config.file_output:
        return None

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, filename)

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8"
    )

    if is_error_handler:
        handler.setLevel(logging.ERROR)
    else:
        handler.setLevel(_get_log_level(config.level))

    handler.setFormatter(_create_formatter())
    return handler


def _setup_module_logger(
    module_name: str,
    log_filename: str,
    config: LoggingConfig,
    log_dir: str
) -> logging.Logger:
    """Set up a logger for a specific module.

    Args:
        module_name: Full module logger name.
        log_filename: Log file name for this module.
        config: Logging configuration.
        log_dir: Directory for log files.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(f"agent_framework.{module_name}")
    logger.setLevel(_get_log_level(config.level))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Add console handler
    console_handler = _create_console_handler(config)
    if console_handler:
        logger.addHandler(console_handler)

    # Add file handler for this module
    file_handler = _create_file_handler(log_dir, log_filename, config)
    if file_handler:
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Set up the logging system.

    This function configures the root logger and module-specific loggers
    based on the provided configuration.

    Args:
        config: Logging configuration. If None, uses default configuration.
    """
    if config is None:
        config = LoggingConfig()

    log_dir = config.log_dir

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger("agent_framework")
    root_logger.setLevel(_get_log_level(config.level))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add console handler to root logger
    console_handler = _create_console_handler(config)
    if console_handler:
        root_logger.addHandler(console_handler)

    # Add main app file handler
    app_handler = _create_file_handler(log_dir, "app.log", config)
    if app_handler:
        root_logger.addHandler(app_handler)

    # Add error file handler to root logger
    error_handler = _create_file_handler(log_dir, "error.log", config, is_error_handler=True)
    if error_handler:
        root_logger.addHandler(error_handler)

    # Set up module-specific loggers
    for module_name, log_filename in MODULE_LOGGERS.items():
        _setup_module_logger(module_name, log_filename, config, log_dir)

    # Log startup message
    root_logger.info(f"Logging initialized: level={config.level}, log_dir={log_dir}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name.

    Args:
        name: Logger name.

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)


def get_module_logger(module_name: str) -> logging.Logger:
    """Get a logger for a specific module.

    This is a convenience function that adds the agent_framework prefix.

    Args:
        module_name: Module name (e.g., "gateway", "planners.react").

    Returns:
        logging.Logger instance with agent_framework prefix.
    """
    return logging.getLogger(f"agent_framework.{module_name}")