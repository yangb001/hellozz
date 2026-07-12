"""Independent test cases for configuration management.

This module contains independent verification tests for Config data classes
and load/save functions defined in core/config.py.

Test categories:
1. MemoryConfig data class
2. LLMProviderConfig data class
3. LLMConfig data class
4. SQLiteConfig data class
5. PlannerConfig data class
6. Config root container
7. load_config function
8. save_config function
9. Boundary conditions
"""
import pytest
import os
import tempfile
from pathlib import Path

from agent_framework.core.config import (
    MemoryConfig, LLMProviderConfig, LLMConfig,
    SQLiteConfig, PlannerConfig, Config,
    load_config, save_config,
)


def _write_temp_json(content: str) -> str:
    """Write content to a temp JSON file and return its path.

    Properly closes the file before returning so it can be read on Windows.
    """
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        os.unlink(path)
        raise
    return path


# ============================================================================
# 1. MemoryConfig
# ============================================================================


class TestMemoryConfig:
    """Test MemoryConfig data class."""

    def test_default_short_term_size(self):
        """Default short_term_size should be 20."""
        cfg = MemoryConfig()
        assert cfg.short_term_size == 20

    def test_default_vector_db(self):
        """Default vector_db should be 'lancedb'."""
        cfg = MemoryConfig()
        assert cfg.vector_db == "lancedb"

    def test_default_vector_path(self):
        """Default vector_path should be './data/vectors'."""
        cfg = MemoryConfig()
        assert cfg.vector_path == "./data/vectors"

    def test_default_embedding_model(self):
        """Default embedding_model should be 'all-MiniLM-L6-v2'."""
        cfg = MemoryConfig()
        assert cfg.embedding_model == "all-MiniLM-L6-v2"

    def test_default_extraction_trigger(self):
        """Default extraction_trigger should be 'smart'."""
        cfg = MemoryConfig()
        assert cfg.extraction_trigger == "smart"

    def test_default_extraction_model(self):
        """Default extraction_model should be 'ollama/llama3'."""
        cfg = MemoryConfig()
        assert cfg.extraction_model == "ollama/llama3"

    def test_custom_values(self):
        """MemoryConfig accepts custom values."""
        cfg = MemoryConfig(
            short_term_size=50,
            vector_db="chroma",
            vector_path="/tmp/vectors",
            embedding_model="custom-model",
            extraction_trigger="every_n_turns:5",
            extraction_model="openai/gpt-4",
        )
        assert cfg.short_term_size == 50
        assert cfg.vector_db == "chroma"
        assert cfg.vector_path == "/tmp/vectors"
        assert cfg.embedding_model == "custom-model"
        assert cfg.extraction_trigger == "every_n_turns:5"
        assert cfg.extraction_model == "openai/gpt-4"


# ============================================================================
# 2. LLMProviderConfig
# ============================================================================


class TestLLMProviderConfig:
    """Test LLMProviderConfig data class."""

    def test_default_type(self):
        """Default type should be 'ollama'."""
        cfg = LLMProviderConfig()
        assert cfg.type == "ollama"

    def test_default_model(self):
        """Default model should be 'llama3'."""
        cfg = LLMProviderConfig()
        assert cfg.model == "llama3"

    def test_default_base_url(self):
        """Default base_url should be 'http://localhost:11434'."""
        cfg = LLMProviderConfig()
        assert cfg.base_url == "http://localhost:11434"

    def test_default_api_key_none(self):
        """Default api_key should be None."""
        cfg = LLMProviderConfig()
        assert cfg.api_key is None

    def test_custom_values(self):
        """LLMProviderConfig accepts custom values."""
        cfg = LLMProviderConfig(
            type="openai",
            model="gpt-4",
            base_url="https://api.openai.com",
            api_key="sk-test",
        )
        assert cfg.type == "openai"
        assert cfg.model == "gpt-4"
        assert cfg.base_url == "https://api.openai.com"
        assert cfg.api_key == "sk-test"


# ============================================================================
# 3. LLMConfig
# ============================================================================


class TestLLMConfig:
    """Test LLMConfig data class."""

    def test_default_provider(self):
        """Default provider should be 'ollama'."""
        cfg = LLMConfig()
        assert cfg.default == "ollama"

    def test_default_providers_empty(self):
        """Default providers should be empty dict."""
        cfg = LLMConfig()
        assert cfg.providers == {}

    def test_custom_providers(self):
        """LLMConfig accepts custom providers."""
        providers = {
            "ollama": LLMProviderConfig(type="ollama", model="llama3"),
            "openai": LLMProviderConfig(type="openai", model="gpt-4"),
        }
        cfg = LLMConfig(default="openai", providers=providers)
        assert cfg.default == "openai"
        assert len(cfg.providers) == 2
        assert cfg.providers["ollama"].model == "llama3"


# ============================================================================
# 4. SQLiteConfig
# ============================================================================


class TestSQLiteConfig:
    """Test SQLiteConfig data class."""

    def test_default_path(self):
        """Default path should be './data/sessions.db'."""
        cfg = SQLiteConfig()
        assert cfg.path == "./data/sessions.db"

    def test_custom_path(self):
        """SQLiteConfig accepts custom path."""
        cfg = SQLiteConfig(path="/tmp/test.db")
        assert cfg.path == "/tmp/test.db"


# ============================================================================
# 5. PlannerConfig
# ============================================================================


class TestPlannerConfig:
    """Test PlannerConfig data class."""

    def test_default_type(self):
        """Default type should be the ReAct planner path."""
        cfg = PlannerConfig()
        assert cfg.type == "planners.react_planner.ReActPlanner"

    def test_custom_type(self):
        """PlannerConfig accepts custom type."""
        cfg = PlannerConfig(type="planners.plan_execute.PlanExecutePlanner")
        assert cfg.type == "planners.plan_execute.PlanExecutePlanner"


# ============================================================================
# 6. Config Root Container
# ============================================================================


class TestConfig:
    """Test Config root container."""

    def test_default_config(self):
        """Config can be created with all defaults."""
        cfg = Config()
        assert isinstance(cfg.sqlite, SQLiteConfig)
        assert isinstance(cfg.memory, MemoryConfig)
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.planner, PlannerConfig)

    def test_custom_sub_configs(self):
        """Config accepts custom sub-configurations."""
        cfg = Config(
            sqlite=SQLiteConfig(path="/tmp/test.db"),
            memory=MemoryConfig(short_term_size=50),
        )
        assert cfg.sqlite.path == "/tmp/test.db"
        assert cfg.memory.short_term_size == 50

    def test_to_dict(self):
        """Config.to_dict returns a dictionary."""
        cfg = Config()
        data = cfg.to_dict()
        assert isinstance(data, dict)

    def test_to_dict_has_sqlite(self):
        """to_dict includes sqlite section."""
        cfg = Config()
        data = cfg.to_dict()
        assert "sqlite" in data
        assert "path" in data["sqlite"]

    def test_to_dict_has_memory(self):
        """to_dict includes memory section."""
        cfg = Config()
        data = cfg.to_dict()
        assert "memory" in data
        assert "short_term_size" in data["memory"]
        assert "vector_db" in data["memory"]

    def test_to_dict_has_llm(self):
        """to_dict includes llm section."""
        cfg = Config()
        data = cfg.to_dict()
        assert "llm" in data
        assert "default" in data["llm"]
        assert "providers" in data["llm"]

    def test_to_dict_has_planner(self):
        """to_dict includes planner section."""
        cfg = Config()
        data = cfg.to_dict()
        assert "planner" in data

    def test_to_dict_preserves_values(self):
        """to_dict preserves configured values."""
        cfg = Config(sqlite=SQLiteConfig(path="/custom/path.db"))
        data = cfg.to_dict()
        assert data["sqlite"]["path"] == "/custom/path.db"

    def test_to_dict_with_providers(self):
        """to_dict serializes LLM providers correctly."""
        providers = {
            "ollama": LLMProviderConfig(
                type="ollama", model="llama3",
                base_url="http://localhost:11434"
            ),
        }
        cfg = Config(llm=LLMConfig(default="ollama", providers=providers))
        data = cfg.to_dict()
        assert "ollama" in data["llm"]["providers"]
        assert data["llm"]["providers"]["ollama"]["type"] == "ollama"


# ============================================================================
# 7. load_config Function
# ============================================================================


class TestLoadConfig:
    """Test load_config function."""

    def test_load_nonexistent_file_raises(self):
        """load_config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.json")

    def test_load_empty_json(self):
        """load_config handles empty JSON file."""
        path = _write_temp_json("")
        try:
            cfg = load_config(path)
            assert isinstance(cfg, Config)
        finally:
            os.unlink(path)

    def test_load_partial_json(self):
        """load_config handles partial JSON with defaults."""
        path = _write_temp_json('{"sqlite": {"path": "/tmp/test.db"}}')
        try:
            cfg = load_config(path)
            assert cfg.sqlite.path == "/tmp/test.db"
            assert cfg.memory.short_term_size == 20
        finally:
            os.unlink(path)

    def test_load_full_json(self):
        """load_config loads complete JSON configuration."""
        import json
        data = {
            "sqlite": {"path": "/tmp/sessions.db"},
            "memory": {
                "short_term_size": 50,
                "vector_db": "chroma",
                "vector_path": "/tmp/vectors",
                "embedding_model": "custom-model",
                "extraction": {
                    "trigger": "smart",
                    "model": "ollama/llama3",
                },
            },
            "llm": {
                "default": "openai",
                "providers": {
                    "ollama": {
                        "type": "ollama",
                        "model": "llama3",
                        "base_url": "http://localhost:11434",
                    },
                    "openai": {
                        "type": "openai",
                        "model": "gpt-4",
                        "base_url": "https://api.openai.com",
                        "api_key": "sk-test",
                    },
                },
            },
            "planner": "planners.react_planner.ReActPlanner",
        }
        path = _write_temp_json(json.dumps(data))
        try:
            cfg = load_config(path)
            assert cfg.sqlite.path == "/tmp/sessions.db"
            assert cfg.memory.short_term_size == 50
            assert cfg.memory.vector_db == "chroma"
            assert cfg.llm.default == "openai"
            assert len(cfg.llm.providers) == 2
            assert cfg.llm.providers["openai"].api_key == "sk-test"
            assert cfg.planner.type == "planners.react_planner.ReActPlanner"
        finally:
            os.unlink(path)

    def test_load_invalid_json_raises(self):
        """load_config raises on invalid JSON."""
        path = _write_temp_json("{{invalid json}}")
        try:
            with pytest.raises(Exception):
                load_config(path)
        finally:
            os.unlink(path)

    def test_load_returns_config_type(self):
        """load_config always returns a Config instance."""
        path = _write_temp_json('{"sqlite": {"path": "/tmp/test.db"}}')
        try:
            cfg = load_config(path)
            assert isinstance(cfg, Config)
        finally:
            os.unlink(path)


# ============================================================================
# 8. save_config Function
# ============================================================================


class TestSaveConfig:
    """Test save_config function."""

    def test_save_creates_file(self):
        """save_config creates the config file."""
        cfg = Config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            save_config(cfg, path)
            assert os.path.exists(path)

    def test_save_creates_directory(self):
        """save_config creates parent directories if needed."""
        cfg = Config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "config.json")
            save_config(cfg, path)
            assert os.path.exists(path)

    def test_save_load_roundtrip(self):
        """Config survives save then load roundtrip."""
        cfg = Config(
            sqlite=SQLiteConfig(path="/tmp/test.db"),
            memory=MemoryConfig(short_term_size=50),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            save_config(cfg, path)
            loaded = load_config(path)
            assert loaded.sqlite.path == "/tmp/test.db"
            assert loaded.memory.short_term_size == 50

    def test_save_produces_valid_json(self):
        """save_config produces valid JSON."""
        import json
        cfg = Config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            save_config(cfg, path)
            with open(path, "r") as f:
                data = json.load(f)
            assert isinstance(data, dict)


# ============================================================================
# 9. Boundary Conditions
# ============================================================================


class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_memory_config_zero_short_term(self):
        """MemoryConfig accepts zero short_term_size."""
        cfg = MemoryConfig(short_term_size=0)
        assert cfg.short_term_size == 0

    def test_memory_config_large_short_term(self):
        """MemoryConfig accepts large short_term_size."""
        cfg = MemoryConfig(short_term_size=10000)
        assert cfg.short_term_size == 10000

    def test_llm_provider_empty_api_key(self):
        """LLMProviderConfig accepts empty string api_key."""
        cfg = LLMProviderConfig(api_key="")
        assert cfg.api_key == ""

    def test_config_to_dict_empty_providers(self):
        """to_dict handles empty providers dict."""
        cfg = Config()
        data = cfg.to_dict()
        assert data["llm"]["providers"] == {}

    def test_config_independent_instances(self):
        """Multiple Config instances are independent."""
        cfg1 = Config()
        cfg2 = Config()
        cfg1.memory.short_term_size = 999
        assert cfg2.memory.short_term_size == 20
