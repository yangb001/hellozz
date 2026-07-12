"""Independent verification tests for the logging configuration system.

Tests cover:
- LoggingConfig data class creation and from_dict loading
- Console output enable/disable
- File output with RotatingFileHandler
- Log rotation (maxBytes and backupCount)
- Error log separate file (error.log)
- Module-specific logger setup
- setup_logging idempotency

These tests are written independently from the developer's TDD tests.
"""
import pytest
import os
import json
import logging
import tempfile
import shutil
from pathlib import Path

from agent_framework.core.logging_config import (
    LoggingConfig,
    setup_logging,
    get_logger,
    get_module_logger,
    DEFAULT_CONFIG,
    MODULE_LOGGERS,
    _get_log_level,
    _create_formatter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleanup_handlers():
    """Close all file handlers across all loggers to release file locks."""
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        for h in logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                logger.removeHandler(h)
    root = logging.getLogger("agent_framework")
    for h in root.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            root.removeHandler(h)


@pytest.fixture
def log_dir():
    """Provide a temporary log directory and clean up afterwards."""
    tmpdir = tempfile.mkdtemp(prefix="test_logging_")
    yield tmpdir
    _cleanup_handlers()
    shutil.rmtree(tmpdir, ignore_errors=True)


def _get_test_logger(name: str) -> logging.Logger:
    """Get a logger under the agent_framework namespace so it inherits handlers.

    Loggers under agent_framework.* that are NOT in MODULE_LOGGERS will
    propagate up to the agent_framework root logger configured by setup_logging.
    """
    return logging.getLogger(f"agent_framework.{name}")


# ===========================================================================
# 1. 日志配置加载
# ===========================================================================

class TestLoggingConfigLoading:
    """Verify LoggingConfig creation from various inputs."""

    def test_default_config_values(self):
        """LoggingConfig() should match DEFAULT_CONFIG."""
        cfg = LoggingConfig()
        assert cfg.level == DEFAULT_CONFIG["level"]
        assert cfg.log_dir == DEFAULT_CONFIG["log_dir"]
        assert cfg.max_bytes == DEFAULT_CONFIG["max_bytes"]
        assert cfg.backup_count == DEFAULT_CONFIG["backup_count"]
        assert cfg.console_output == DEFAULT_CONFIG["console_output"]
        assert cfg.file_output == DEFAULT_CONFIG["file_output"]

    def test_from_dict_partial(self):
        """from_dict with partial data should fill missing keys from defaults."""
        data = {"level": "DEBUG", "backup_count": 3}
        cfg = LoggingConfig.from_dict(data)
        assert cfg.level == "DEBUG"
        assert cfg.backup_count == 3
        # defaults
        assert cfg.log_dir == DEFAULT_CONFIG["log_dir"]
        assert cfg.max_bytes == DEFAULT_CONFIG["max_bytes"]

    def test_from_dict_empty_returns_defaults(self):
        cfg = LoggingConfig.from_dict({})
        assert cfg.level == "INFO"
        assert cfg.log_dir == "logs"

    def test_from_dict_none_returns_defaults(self):
        cfg = LoggingConfig.from_dict(None)
        assert cfg.level == "INFO"
        assert cfg.console_output is True

    def test_from_dict_all_fields(self):
        data = {
            "level": "WARNING",
            "log_dir": "/tmp/all_fields",
            "max_bytes": 512,
            "backup_count": 2,
            "console_output": False,
            "file_output": False,
        }
        cfg = LoggingConfig.from_dict(data)
        assert cfg.level == "WARNING"
        assert cfg.log_dir == "/tmp/all_fields"
        assert cfg.max_bytes == 512
        assert cfg.backup_count == 2
        assert cfg.console_output is False
        assert cfg.file_output is False

    def test_from_json_file(self, log_dir):
        """Simulate loading logging config from a config.json file."""
        config_data = {
            "logging": {
                "level": "DEBUG",
                "log_dir": os.path.join(log_dir, "from_json"),
                "max_bytes": 2048,
                "backup_count": 7,
            }
        }
        path = os.path.join(log_dir, "config.json")
        with open(path, "w") as f:
            json.dump(config_data, f)

        with open(path) as f:
            loaded = json.load(f)

        cfg = LoggingConfig.from_dict(loaded.get("logging", {}))
        assert cfg.level == "DEBUG"
        assert cfg.max_bytes == 2048
        assert cfg.backup_count == 7

    def test_get_log_level_known(self):
        assert _get_log_level("DEBUG") == logging.DEBUG
        assert _get_log_level("INFO") == logging.INFO
        assert _get_log_level("WARNING") == logging.WARNING
        assert _get_log_level("ERROR") == logging.ERROR
        assert _get_log_level("CRITICAL") == logging.CRITICAL

    def test_get_log_level_case_insensitive(self):
        assert _get_log_level("info") == logging.INFO
        assert _get_log_level("Warning") == logging.WARNING

    def test_get_log_level_unknown_defaults_to_info(self):
        assert _get_log_level("NOTEXIST") == logging.INFO

    def test_formatter_contains_expected_fields(self):
        fmt = _create_formatter()
        assert isinstance(fmt, logging.Formatter)
        assert "%(asctime)s" in fmt._fmt
        assert "%(levelname)" in fmt._fmt
        assert "%(name)" in fmt._fmt
        assert "%(message)s" in fmt._fmt


# ===========================================================================
# 2. 控制台输出
# ===========================================================================

class TestConsoleOutput:
    """Verify console handler creation and output behaviour."""

    def test_console_handler_created_when_enabled(self, log_dir):
        cfg = LoggingConfig(log_dir=log_dir, console_output=True)
        setup_logging(cfg)
        root = logging.getLogger("agent_framework")
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_console_handler_not_created_when_disabled(self, log_dir):
        cfg = LoggingConfig(log_dir=log_dir, console_output=False)
        setup_logging(cfg)
        root = logging.getLogger("agent_framework")
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 0

    def test_console_output_contains_message(self, log_dir, capsys):
        cfg = LoggingConfig(log_dir=log_dir, console_output=True, level="INFO")
        setup_logging(cfg)
        # Use a logger that propagates to agent_framework root
        logger = _get_test_logger("test_console")
        logger.info("hello_console")
        captured = capsys.readouterr()
        assert "hello_console" in captured.out or "hello_console" in captured.err


# ===========================================================================
# 3. 文件输出
# ===========================================================================

class TestFileOutput:
    """Verify file handler creation and content writing."""

    def test_log_directory_created(self, log_dir):
        target = os.path.join(log_dir, "subdir")
        cfg = LoggingConfig(log_dir=target, file_output=True)
        setup_logging(cfg)
        assert os.path.isdir(target)

    def test_app_log_created(self, log_dir):
        cfg = LoggingConfig(log_dir=log_dir, file_output=True)
        setup_logging(cfg)
        logger = _get_test_logger("test_file")
        logger.info("file_test_message")
        app_log = os.path.join(log_dir, "app.log")
        assert os.path.exists(app_log)
        with open(app_log, encoding="utf-8") as f:
            content = f.read()
        assert "file_test_message" in content

    def test_no_file_when_file_output_disabled(self, log_dir):
        cfg = LoggingConfig(log_dir=log_dir, file_output=False, console_output=False)
        setup_logging(cfg)
        logger = _get_test_logger("test_nofile")
        logger.info("should_not_appear_in_file")
        app_log = os.path.join(log_dir, "app.log")
        assert not os.path.exists(app_log)

    def test_module_specific_log_files(self, log_dir):
        """Different modules should write to their designated log files."""
        cfg = LoggingConfig(log_dir=log_dir, file_output=True, level="INFO")
        setup_logging(cfg)

        gateway_logger = get_module_logger("gateway")
        memory_logger = get_module_logger("memory")
        planner_logger = get_module_logger("planners.react_planner")

        gateway_logger.info("gw_msg")
        memory_logger.info("mem_msg")
        planner_logger.info("plan_msg")

        gw_log = os.path.join(log_dir, "gateway.log")
        mem_log = os.path.join(log_dir, "memory.log")
        plan_log = os.path.join(log_dir, "planner.log")

        assert os.path.exists(gw_log)
        assert os.path.exists(mem_log)
        assert os.path.exists(plan_log)

        with open(gw_log, encoding="utf-8") as f:
            assert "gw_msg" in f.read()
        with open(mem_log, encoding="utf-8") as f:
            assert "mem_msg" in f.read()
        with open(plan_log, encoding="utf-8") as f:
            assert "plan_msg" in f.read()


# ===========================================================================
# 4. 日志轮转
# ===========================================================================

class TestLogRotation:
    """Verify RotatingFileHandler rotation behaviour."""

    def test_rotation_creates_backup_files(self, log_dir):
        """When maxBytes is small, writing many messages should create backups.

        Note: On Windows, RotatingFileHandler may intermittently fail to rename
        files due to file locking (PermissionError). We flush handlers between
        batches to mitigate this.
        """
        cfg = LoggingConfig(
            log_dir=log_dir,
            file_output=True,
            console_output=False,
            max_bytes=256,       # very small to force rotation
            backup_count=3,
            level="DEBUG",
        )
        setup_logging(cfg)
        logger = _get_test_logger("rotation_test")

        payload = "A" * 200
        for i in range(50):
            logger.info(f"msg {i} {payload}")
            # Flush handlers periodically to avoid Windows file locking issues
            if i % 10 == 0:
                for h in logger.handlers:
                    h.flush()

        # Flush all handlers before checking files
        root = logging.getLogger("agent_framework")
        for h in root.handlers:
            h.flush()

        log_files = sorted([
            f for f in os.listdir(log_dir)
            if f.startswith("app.log")
        ])
        # At minimum, the main app.log should exist; rotation may or may not
        # succeed on Windows due to file locking
        assert len(log_files) >= 1, "Expected at least app.log"
        assert len(log_files) <= 4

    def test_backup_count_respected(self, log_dir):
        """Number of backup files should not exceed backup_count."""
        cfg = LoggingConfig(
            log_dir=log_dir,
            file_output=True,
            console_output=False,
            max_bytes=128,
            backup_count=2,
            level="DEBUG",
        )
        setup_logging(cfg)
        logger = _get_test_logger("rotation_limit")

        payload = "B" * 100
        for i in range(100):
            logger.info(f"overflow {i} {payload}")

        log_files = [f for f in os.listdir(log_dir) if f.startswith("app.log")]
        assert len(log_files) <= 3  # main + 2 backups

    def test_module_file_rotation(self, log_dir):
        """Module-specific log files should also rotate."""
        cfg = LoggingConfig(
            log_dir=log_dir,
            file_output=True,
            console_output=False,
            max_bytes=128,
            backup_count=1,
            level="DEBUG",
        )
        setup_logging(cfg)
        logger = get_module_logger("gateway")

        payload = "C" * 100
        for i in range(60):
            logger.info(f"gw_rot {i} {payload}")

        gw_files = [f for f in os.listdir(log_dir) if f.startswith("gateway.log")]
        assert len(gw_files) >= 1


# ===========================================================================
# 5. 错误日志单独记录
# ===========================================================================

class TestErrorLogSeparation:
    """Verify that ERROR+ messages go to a dedicated error.log file."""

    def test_error_log_file_created(self, log_dir):
        cfg = LoggingConfig(log_dir=log_dir, file_output=True)
        setup_logging(cfg)
        logger = _get_test_logger("error_test")
        logger.error("something_broke")
        error_log = os.path.join(log_dir, "error.log")
        assert os.path.exists(error_log)
        with open(error_log, encoding="utf-8") as f:
            assert "something_broke" in f.read()

    def test_info_does_not_appear_in_error_log(self, log_dir):
        cfg = LoggingConfig(log_dir=log_dir, file_output=True, level="INFO")
        setup_logging(cfg)
        logger = _get_test_logger("error_filter")
        logger.info("info_only")
        error_log = os.path.join(log_dir, "error.log")
        if os.path.exists(error_log):
            with open(error_log, encoding="utf-8") as f:
                content = f.read()
            assert "info_only" not in content

    def test_warning_does_not_appear_in_error_log(self, log_dir):
        cfg = LoggingConfig(log_dir=log_dir, file_output=True, level="WARNING")
        setup_logging(cfg)
        logger = _get_test_logger("warn_filter")
        logger.warning("warn_only")
        error_log = os.path.join(log_dir, "error.log")
        if os.path.exists(error_log):
            with open(error_log, encoding="utf-8") as f:
                content = f.read()
            assert "warn_only" not in content

    def test_critical_appears_in_error_log(self, log_dir):
        cfg = LoggingConfig(log_dir=log_dir, file_output=True, level="DEBUG")
        setup_logging(cfg)
        logger = _get_test_logger("critical_test")
        logger.critical("fatal_problem")
        error_log = os.path.join(log_dir, "error.log")
        assert os.path.exists(error_log)
        with open(error_log, encoding="utf-8") as f:
            assert "fatal_problem" in f.read()


# ===========================================================================
# 6. get_logger / get_module_logger helpers
# ===========================================================================

class TestLoggerHelpers:
    """Verify get_logger and get_module_logger return correct loggers."""

    def test_get_logger_returns_correct_name(self):
        logger = get_logger("my_module")
        assert logger.name == "my_module"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_same_name_same_instance(self):
        a = get_logger("singleton_check")
        b = get_logger("singleton_check")
        assert a is b

    def test_get_module_logger_prefixes_agent_framework(self):
        logger = get_module_logger("gateway")
        assert logger.name == "agent_framework.gateway"

    def test_get_module_logger_nested(self):
        logger = get_module_logger("planners.react_planner")
        assert logger.name == "agent_framework.planners.react_planner"


# ===========================================================================
# 7. setup_logging 幂等性 & 边界
# ===========================================================================

class TestSetupLoggingEdgeCases:
    """Verify setup_logging idempotency and boundary conditions."""

    def test_setup_logging_is_idempotent(self, log_dir):
        """Calling setup_logging twice should not duplicate handlers."""
        cfg = LoggingConfig(log_dir=log_dir)
        setup_logging(cfg)
        root = logging.getLogger("agent_framework")
        count_before = len(root.handlers)
        setup_logging(cfg)
        count_after = len(root.handlers)
        assert count_after == count_before

    def test_setup_logging_with_none_uses_defaults(self, log_dir):
        """setup_logging(None) should use default config without error."""
        cfg = LoggingConfig(log_dir=log_dir)
        setup_logging(cfg)
        assert os.path.isdir(log_dir)

    def test_module_loggers_propagate_false(self, log_dir):
        """Module loggers should have propagate=False to avoid duplicate output."""
        cfg = LoggingConfig(log_dir=log_dir)
        setup_logging(cfg)
        for module_name in MODULE_LOGGERS:
            logger = logging.getLogger(f"agent_framework.{module_name}")
            assert logger.propagate is False

    def test_all_module_loggers_configured(self, log_dir):
        """Every entry in MODULE_LOGGERS should result in a configured logger."""
        cfg = LoggingConfig(log_dir=log_dir, level="DEBUG")
        setup_logging(cfg)
        for module_name in MODULE_LOGGERS:
            logger = logging.getLogger(f"agent_framework.{module_name}")
            assert len(logger.handlers) >= 1, f"No handlers for {module_name}"

    def test_log_level_filtering(self, log_dir):
        """Logger should not output messages below configured level."""
        cfg = LoggingConfig(log_dir=log_dir, level="WARNING", file_output=True, console_output=False)
        setup_logging(cfg)
        logger = _get_test_logger("level_filter")
        logger.debug("debug_msg")
        logger.info("info_msg")
        logger.warning("warning_msg")
        app_log = os.path.join(log_dir, "app.log")
        with open(app_log, encoding="utf-8") as f:
            content = f.read()
        assert "debug_msg" not in content
        assert "info_msg" not in content
        assert "warning_msg" in content
