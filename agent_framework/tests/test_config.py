"""Tests for core/config.py configuration management."""
import pytest
import tempfile
import os
import shutil
import logging
from pathlib import Path

from agent_framework.core.config import (
    Config,
    MemoryConfig,
    LLMConfig,
    LLMProviderConfig,
    SQLiteConfig,
    PlannerConfig,
    LoggingConfig,
    load_config,
    save_config,
)
from agent_framework.core.logging_config import setup_logging


class TestMemoryConfig:
    """Tests for MemoryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MemoryConfig()
        assert config.short_term_size == 20
        assert config.vector_db == "lancedb"
        assert config.vector_path == "./data/vectors"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.trigger == "smart"
        assert config.extraction_model == "ollama/llama3"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MemoryConfig(
            short_term_size=50,
            vector_db="chroma",
            vector_path="/custom/path",
            trigger="every_n:10",
        )
        assert config.short_term_size == 50
        assert config.vector_db == "chroma"
        assert config.trigger == "every_n:10"


class TestLLMProviderConfig:
    """Tests for LLMProviderConfig dataclass."""

    def test_default_values(self):
        """Test default LLM provider values."""
        config = LLMProviderConfig()
        assert config.type == "ollama"
        assert config.model == "llama3"
        assert config.base_url == "http://localhost:11434"
        assert config.api_key is None

    def test_custom_values(self):
        """Test custom LLM provider values."""
        config = LLMProviderConfig(
            type="openai",
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="secret-key",
        )
        assert config.type == "openai"
        assert config.model == "gpt-4"
        assert config.api_key == "secret-key"


class TestSQLiteConfig:
    """Tests for SQLiteConfig dataclass."""

    def test_default_values(self):
        """Test default SQLite config values."""
        config = SQLiteConfig()
        assert config.path == "./data/sessions.db"

    def test_custom_values(self):
        """Test custom SQLite config values."""
        config = SQLiteConfig(path="/var/data/sessions.db")
        assert config.path == "/var/data/sessions.db"


class TestPlannerConfig:
    """Tests for PlannerConfig dataclass."""

    def test_default_values(self):
        """Test default Planner config values."""
        config = PlannerConfig()
        assert config.type == "planners.react_planner.ReActPlanner"

    def test_custom_values(self):
        """Test custom Planner config values."""
        config = PlannerConfig(type="planners.graph_planner.GraphPlanner")
        assert config.type == "planners.graph_planner.GraphPlanner"


class TestConfigToDict:
    """Tests for Config.to_dict() method."""

    def test_to_dict_structure(self):
        """Test that to_dict produces expected structure."""
        config = Config()
        data = config.to_dict()

        assert "sqlite" in data
        assert "memory" in data
        assert "llm" in data
        assert "planner" in data

    def test_to_dict_memory_values(self):
        """Test memory section in to_dict."""
        config = Config()
        data = config.to_dict()

        assert data["memory"]["short_term_size"] == 20
        assert data["memory"]["vector_db"] == "lancedb"

    def test_to_dict_llm_values(self):
        """Test LLM section in to_dict."""
        config = Config()
        data = config.to_dict()

        assert data["llm"]["default"] == "ollama"
        assert data["llm"]["providers"] == {}

    def test_to_dict_sqlite_values(self):
        """Test sqlite section in to_dict."""
        config = Config(sqlite=SQLiteConfig(path="/custom/db.db"))
        data = config.to_dict()

        assert data["sqlite"]["path"] == "/custom/db.db"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_file(self):
        """Test loading config from JSON file."""
        json_content = """{
    "sqlite": {
        "path": "/custom/path/sessions.db"
    },
    "memory": {
        "short_term_size": 30,
        "vector_db": "chroma",
        "embedding_model": "custom-model",
        "extraction": {
            "trigger": "every_n:5",
            "model": "custom/model"
        }
    },
    "llm": {
        "default": "openai",
        "providers": {
            "openai": {
                "type": "openai",
                "model": "gpt-4",
                "base_url": "https://api.openai.com/v1"
            },
            "local": {
                "type": "ollama",
                "model": "llama3",
                "base_url": "http://localhost:11434"
            }
        }
    },
    "planner": "planners.graph_planner.GraphPlanner"
}"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            assert config.sqlite.path == "/custom/path/sessions.db"
            assert config.memory.short_term_size == 30
            assert config.memory.vector_db == "chroma"
            assert config.memory.embedding_model == "custom-model"
            assert config.memory.trigger == "every_n:5"
            assert config.llm.default == "openai"
            assert "openai" in config.llm.providers
            assert config.llm.providers["openai"].model == "gpt-4"
            assert "local" in config.llm.providers
            assert config.planner.type == "planners.graph_planner.GraphPlanner"
        finally:
            os.unlink(temp_path)

    def test_load_missing_file_raises(self):
        """Test that loading missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.json")

    def test_load_partial_config_uses_defaults(self):
        """Test that missing keys use default values."""
        json_content = """{
    "sqlite": {
        "path": "/custom/path.db"
    }
}"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            assert config.sqlite.path == "/custom/path.db"
            assert config.memory.short_term_size == 20
            assert config.memory.vector_db == "lancedb"
            assert config.llm.default == "ollama"
        finally:
            os.unlink(temp_path)

    def test_load_empty_file(self):
        """Test loading empty file uses defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            config = load_config(temp_path)

            assert config.sqlite.path == "./data/sessions.db"
            assert config.memory.short_term_size == 20
        finally:
            os.unlink(temp_path)

    def test_load_empty_json_object(self):
        """Test that empty JSON object uses defaults."""
        json_content = "{}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            assert config.sqlite.path == "./data/sessions.db"
            assert config.memory.short_term_size == 20
        finally:
            os.unlink(temp_path)

    def test_load_invalid_json_raises(self):
        """Test that invalid JSON raises an error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json content")
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_and_reload(self):
        """Test that config can be saved and reloaded as JSON."""
        config = Config(
            sqlite=SQLiteConfig(path="/test/path.db"),
            memory=MemoryConfig(short_term_size=100),
            llm=LLMConfig(
                default="openai",
                providers={
                    "openai": LLMProviderConfig(
                        type="openai",
                        model="gpt-4",
                        base_url="https://api.openai.com/v1",
                    )
                },
            ),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_config(config, temp_path)

            reloaded = load_config(temp_path)
            assert reloaded.sqlite.path == "/test/path.db"
            assert reloaded.memory.short_term_size == 100
            assert reloaded.llm.default == "openai"
            assert "openai" in reloaded.llm.providers
        finally:
            os.unlink(temp_path)

    def test_save_creates_directory(self):
        """Test that save_config creates parent directories."""
        config = Config()
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "subdir", "config.json")

        try:
            save_config(config, temp_path)
            assert os.path.exists(temp_path)
        finally:
            shutil.rmtree(temp_dir)

    def test_save_produces_valid_json(self):
        """Test that saved file is valid JSON."""
        config = Config()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_config(config, temp_path)
            import json
            with open(temp_path, "r") as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert "sqlite" in data
            assert "memory" in data
        finally:
            os.unlink(temp_path)


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass in config module."""

    def test_default_values(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.log_dir == "logs"
        assert config.max_bytes == 10 * 1024 * 1024
        assert config.backup_count == 5
        assert config.console_output is True
        assert config.file_output is True

    def test_custom_values(self):
        """Test LoggingConfig with custom values."""
        config = LoggingConfig(
            level="DEBUG",
            log_dir="/tmp/my_logs",
            max_bytes=5 * 1024 * 1024,
            backup_count=3,
            console_output=False,
            file_output=True,
        )
        assert config.level == "DEBUG"
        assert config.log_dir == "/tmp/my_logs"
        assert config.max_bytes == 5 * 1024 * 1024
        assert config.backup_count == 3
        assert config.console_output is False


class TestConfigWithLogging:
    """Tests for Config class with logging section."""

    def test_config_has_logging_field(self):
        """Test that Config has a logging attribute."""
        config = Config()
        assert hasattr(config, "logging")
        assert isinstance(config.logging, LoggingConfig)

    def test_config_logging_defaults(self):
        """Test Config logging defaults."""
        config = Config()
        assert config.logging.level == "INFO"
        assert config.logging.log_dir == "logs"

    def test_config_logging_custom(self):
        """Test Config with custom logging."""
        logging_cfg = LoggingConfig(level="DEBUG", log_dir="/custom/logs")
        config = Config(logging=logging_cfg)
        assert config.logging.level == "DEBUG"
        assert config.logging.log_dir == "/custom/logs"

    def test_config_to_dict_includes_logging(self):
        """Test that to_dict includes logging section."""
        config = Config()
        data = config.to_dict()
        assert "logging" in data
        assert "level" in data["logging"]
        assert "log_dir" in data["logging"]
        assert data["logging"]["level"] == "INFO"
        assert data["logging"]["log_dir"] == "logs"

    def test_config_to_dict_logging_custom(self):
        """Test to_dict with custom logging config."""
        logging_cfg = LoggingConfig(level="WARNING", log_dir="/var/logs")
        config = Config(logging=logging_cfg)
        data = config.to_dict()
        assert data["logging"]["level"] == "WARNING"
        assert data["logging"]["log_dir"] == "/var/logs"


class TestLoadConfigWithLogging:
    """Tests for load_config with logging section."""

    def test_load_config_with_logging_section(self):
        """Test loading config with logging section from JSON."""
        json_content = """{
    "logging": {
        "level": "DEBUG",
        "log_dir": "custom_logs",
        "max_bytes": 5242880,
        "backup_count": 3
    }
}"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.logging.level == "DEBUG"
            assert config.logging.log_dir == "custom_logs"
            assert config.logging.max_bytes == 5242880
            assert config.logging.backup_count == 3
        finally:
            os.unlink(temp_path)

    def test_load_config_without_logging_uses_defaults(self):
        """Test that missing logging section uses defaults."""
        json_content = """{
    "sqlite": {
        "path": "/test.db"
    }
}"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.logging.level == "INFO"
            assert config.logging.log_dir == "logs"
            assert config.logging.max_bytes == 10 * 1024 * 1024
            assert config.logging.backup_count == 5
        finally:
            os.unlink(temp_path)

    def test_load_config_partial_logging_section(self):
        """Test loading config with partial logging section."""
        json_content = """{
    "logging": {
        "level": "WARNING"
    }
}"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.logging.level == "WARNING"
            assert config.logging.log_dir == "logs"
            assert config.logging.max_bytes == 10 * 1024 * 1024
        finally:
            os.unlink(temp_path)

    def test_load_config_logging_console_output(self):
        """Test loading config with console_output setting."""
        json_content = """{
    "logging": {
        "level": "INFO",
        "console_output": false
    }
}"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.logging.console_output is False
        finally:
            os.unlink(temp_path)


class TestSaveConfigWithLogging:
    """Tests for save_config with logging section."""

    def test_save_and_reload_preserves_logging(self):
        """Test that save/reload round-trip preserves logging config."""
        logging_cfg = LoggingConfig(
            level="DEBUG",
            log_dir="/test/logs",
            max_bytes=5242880,
            backup_count=3,
        )
        config = Config(logging=logging_cfg)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_config(config, temp_path)
            reloaded = load_config(temp_path)
            assert reloaded.logging.level == "DEBUG"
            assert reloaded.logging.log_dir == "/test/logs"
            assert reloaded.logging.max_bytes == 5242880
            assert reloaded.logging.backup_count == 3
        finally:
            os.unlink(temp_path)

    def test_save_config_with_default_logging(self):
        """Test saving config with default logging values."""
        config = Config()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_config(config, temp_path)
            import json
            with open(temp_path, "r") as f:
                data = json.load(f)
            assert "logging" in data
            assert data["logging"]["level"] == "INFO"
            assert data["logging"]["log_dir"] == "logs"
        finally:
            os.unlink(temp_path)


class TestSetupLoggingFromConfig:
    """Tests for initializing logging from Config object."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for logs."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
        root_logger = logging.getLogger("agent_framework")
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                root_logger.removeHandler(handler)
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_setup_logging_from_config_object(self, temp_log_dir):
        """Test that logging can be initialized from Config object."""
        from agent_framework.core.logging_config import LoggingConfig as CoreLoggingConfig
        logging_cfg = LoggingConfig(level="INFO", log_dir=temp_log_dir)
        core_cfg = CoreLoggingConfig(
            level=logging_cfg.level,
            log_dir=logging_cfg.log_dir,
            max_bytes=logging_cfg.max_bytes,
            backup_count=logging_cfg.backup_count,
            console_output=logging_cfg.console_output,
            file_output=logging_cfg.file_output,
        )
        setup_logging(core_cfg)
        assert os.path.exists(temp_log_dir)

    def test_setup_logging_creates_expected_files(self, temp_log_dir):
        """Test that setup_logging creates the expected log files."""
        from agent_framework.core.logging_config import LoggingConfig as CoreLoggingConfig, get_module_logger
        logging_cfg = LoggingConfig(level="DEBUG", log_dir=temp_log_dir)
        core_cfg = CoreLoggingConfig(
            level=logging_cfg.level,
            log_dir=logging_cfg.log_dir,
            max_bytes=logging_cfg.max_bytes,
            backup_count=logging_cfg.backup_count,
            console_output=logging_cfg.console_output,
            file_output=logging_cfg.file_output,
        )
        setup_logging(core_cfg)

        # Trigger logging from various modules
        get_module_logger("gateway").info("Gateway test")
        get_module_logger("core.session_manager").info("Session test")
        get_module_logger("memory").info("Memory test")
        get_module_logger("planners.react_planner").info("Planner test")
        get_module_logger("infrastructure.llm_gateway").info("LLM test")

        log_files = os.listdir(temp_log_dir)
        assert any("gateway" in f for f in log_files)
        assert any("session" in f for f in log_files)
        assert any("memory" in f for f in log_files)
        assert any("planner" in f for f in log_files)
        assert any("llm" in f for f in log_files)

    def test_error_log_separate_from_config(self, temp_log_dir):
        """Test that error log is separate when initialized from config."""
        from agent_framework.core.logging_config import LoggingConfig as CoreLoggingConfig
        import logging as _logging
        logging_cfg = LoggingConfig(level="DEBUG", log_dir=temp_log_dir)
        core_cfg = CoreLoggingConfig(
            level=logging_cfg.level,
            log_dir=logging_cfg.log_dir,
            max_bytes=logging_cfg.max_bytes,
            backup_count=logging_cfg.backup_count,
            console_output=logging_cfg.console_output,
            file_output=logging_cfg.file_output,
        )
        setup_logging(core_cfg)

        # Use the root agent_framework logger which has the error handler attached
        root_logger = _logging.getLogger("agent_framework")
        root_logger.error("Test error message")

        error_log = os.path.join(temp_log_dir, "error.log")
        assert os.path.exists(error_log)
        with open(error_log, "r") as f:
            content = f.read()
        assert "Test error message" in content