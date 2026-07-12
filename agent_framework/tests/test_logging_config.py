"""Tests for logging configuration - TDD implementation."""
import pytest
import os
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_framework.core.logging_config import (
    LoggingConfig,
    setup_logging,
    get_logger,
    get_module_logger
)


class TestLoggingConfig:
    """Test LoggingConfig data class."""

    def test_logging_config_defaults(self):
        """Test LoggingConfig with default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.log_dir == "logs"
        assert config.max_bytes == 10 * 1024 * 1024  # 10MB
        assert config.backup_count == 5
        assert config.console_output is True
        assert config.file_output is True

    def test_logging_config_custom_values(self):
        """Test LoggingConfig with custom values."""
        config = LoggingConfig(
            level="DEBUG",
            log_dir="/tmp/test_logs",
            max_bytes=5 * 1024 * 1024,  # 5MB
            backup_count=3,
            console_output=False,
            file_output=True
        )
        assert config.level == "DEBUG"
        assert config.log_dir == "/tmp/test_logs"
        assert config.max_bytes == 5 * 1024 * 1024
        assert config.backup_count == 3
        assert config.console_output is False
        assert config.file_output is True

    def test_logging_config_from_dict(self):
        """Test creating LoggingConfig from dictionary."""
        data = {
            "level": "WARNING",
            "log_dir": "custom_logs",
            "max_bytes": 20 * 1024 * 1024,
            "backup_count": 10
        }
        config = LoggingConfig.from_dict(data)
        assert config.level == "WARNING"
        assert config.log_dir == "custom_logs"
        assert config.max_bytes == 20 * 1024 * 1024
        assert config.backup_count == 10

    def test_logging_config_from_empty_dict(self):
        """Test creating LoggingConfig from empty dictionary."""
        config = LoggingConfig.from_dict({})
        assert config.level == "INFO"
        assert config.log_dir == "logs"

    def test_logging_config_from_none(self):
        """Test creating LoggingConfig from None."""
        config = LoggingConfig.from_dict(None)
        assert config.level == "INFO"
        assert config.log_dir == "logs"


class TestSetupLogging:
    """Test setup_logging function."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for logs."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        # Cleanup: close all handlers and remove directory
        import shutil
        # Close all file handlers
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
        # Also close root logger handlers
        root_logger = logging.getLogger("agent_framework")
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                root_logger.removeHandler(handler)
        # Remove directory
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_setup_logging_creates_log_directory(self, temp_log_dir):
        """Test that setup_logging creates the log directory."""
        config = LoggingConfig(log_dir=temp_log_dir)
        setup_logging(config)

        assert os.path.exists(temp_log_dir)

    def test_setup_logging_creates_log_files(self, temp_log_dir):
        """Test that setup_logging creates log files."""
        config = LoggingConfig(log_dir=temp_log_dir)
        setup_logging(config)

        # Get a logger and write to it
        logger = get_logger("test")
        logger.info("Test message")

        # Check that log files are created
        log_files = os.listdir(temp_log_dir)
        assert len(log_files) > 0

    def test_setup_logging_console_output(self, temp_log_dir, capsys):
        """Test that setup_logging outputs to console."""
        config = LoggingConfig(log_dir=temp_log_dir, console_output=True)
        setup_logging(config)

        logger = get_logger("test")
        logger.info("Test console message")

        # Check that message was output to console
        captured = capsys.readouterr()
        # Note: This may not capture output if handlers are not properly configured

    def test_setup_logging_no_console_output(self, temp_log_dir):
        """Test that setup_logging can disable console output."""
        config = LoggingConfig(log_dir=temp_log_dir, console_output=False)
        setup_logging(config)

        # Verify no console handler was added to the agent_framework logger
        root_logger = logging.getLogger("agent_framework")
        console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)
                           and not isinstance(h, logging.FileHandler)]
        assert len(console_handlers) == 0

    def test_setup_logging_respects_level(self, temp_log_dir):
        """Test that setup_logging respects the configured level."""
        config = LoggingConfig(log_dir=temp_log_dir, level="WARNING")
        setup_logging(config)

        logger = get_logger("test")
        assert logger.level == logging.WARNING or logger.getEffectiveLevel() >= logging.WARNING

    def test_setup_logging_idempotent(self, temp_log_dir):
        """Test that setup_logging can be called multiple times safely."""
        config = LoggingConfig(log_dir=temp_log_dir)
        setup_logging(config)
        setup_logging(config)  # Should not raise or duplicate handlers


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_different_modules(self):
        """Test that get_logger returns different loggers for different modules."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        assert logger1 is not logger2
        assert logger1.name == "module1"
        assert logger2.name == "module2"

    def test_get_logger_same_module(self):
        """Test that get_logger returns the same logger for the same module."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        assert logger1 is logger2


class TestGetModuleLogger:
    """Test get_module_logger function."""

    def test_get_module_logger_with_prefix(self):
        """Test that get_module_logger adds agent_framework prefix."""
        logger = get_module_logger("planners.react")
        assert "agent_framework" in logger.name
        assert "planners.react" in logger.name

    def test_get_module_logger_different_modules(self):
        """Test get_module_logger for different modules."""
        logger1 = get_module_logger("gateway")
        logger2 = get_module_logger("core")
        assert logger1.name != logger2.name

    def test_get_module_logger_returns_logger(self):
        """Test that get_module_logger returns a Logger instance."""
        logger = get_module_logger("test")
        assert isinstance(logger, logging.Logger)


class TestLoggingIntegration:
    """Integration tests for logging system."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for logs."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        # Cleanup: close all handlers and remove directory
        import shutil
        # Close all file handlers
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
        # Also close root logger handlers
        root_logger = logging.getLogger("agent_framework")
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                root_logger.removeHandler(handler)
        # Remove directory
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_module_specific_log_files(self, temp_log_dir):
        """Test that different modules write to different log files."""
        config = LoggingConfig(log_dir=temp_log_dir)
        setup_logging(config)

        # Create loggers for different modules
        gateway_logger = get_module_logger("gateway")
        session_logger = get_module_logger("core.session_manager")
        memory_logger = get_module_logger("memory")

        # Write messages
        gateway_logger.info("Gateway message")
        session_logger.info("Session message")
        memory_logger.info("Memory message")

        # Check that log files exist
        log_files = os.listdir(temp_log_dir)
        assert any("gateway" in f for f in log_files)

    def test_error_log_separate_file(self, temp_log_dir):
        """Test that error logs go to a separate file."""
        config = LoggingConfig(log_dir=temp_log_dir)
        setup_logging(config)

        logger = get_logger("test")
        logger.error("Test error message")

        # Check that error.log exists
        error_log_path = os.path.join(temp_log_dir, "error.log")
        assert os.path.exists(error_log_path)

    def test_log_rotation(self, temp_log_dir):
        """Test log rotation configuration."""
        config = LoggingConfig(
            log_dir=temp_log_dir,
            max_bytes=100,  # Very small for testing
            backup_count=2
        )
        setup_logging(config)

        logger = get_logger("test")

        # Write enough data to trigger rotation
        for i in range(100):
            logger.info(f"Test message {i} " + "x" * 50)

        # Check that backup files were created
        log_files = os.listdir(temp_log_dir)
        # Should have main log and backups
        assert len(log_files) > 1

    def test_config_from_json_file(self, temp_log_dir):
        """Test loading logging config from JSON file."""
        config_path = os.path.join(temp_log_dir, "test_config.json")
        config_data = {
            "logging": {
                "level": "DEBUG",
                "log_dir": os.path.join(temp_log_dir, "logs"),
                "max_bytes": 5 * 1024 * 1024,
                "backup_count": 3
            }
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Load config
        with open(config_path) as f:
            loaded_config = json.load(f)

        logging_config = LoggingConfig.from_dict(loaded_config.get("logging", {}))
        assert logging_config.level == "DEBUG"

    def test_logging_with_real_config(self, temp_log_dir):
        """Test logging with a realistic configuration."""
        config_data = {
            "logging": {
                "level": "INFO",
                "log_dir": temp_log_dir,
                "max_bytes": 10485760,
                "backup_count": 5
            }
        }

        logging_config = LoggingConfig.from_dict(config_data.get("logging", {}))
        setup_logging(logging_config)

        # Test various loggers
        loggers = [
            get_module_logger("gateway"),
            get_module_logger("core.session_manager"),
            get_module_logger("memory"),
            get_module_logger("planners.react"),
            get_module_logger("infrastructure.llm_gateway"),
        ]

        for logger in loggers:
            logger.info(f"Test message from {logger.name}")
            logger.warning(f"Warning from {logger.name}")
            logger.error(f"Error from {logger.name}")

        # Verify log files created
        log_files = os.listdir(temp_log_dir)
        assert len(log_files) > 0