"""Tests for SSL configuration in OpenAI LLM and config system."""
import pytest
from unittest.mock import patch, MagicMock

from agent_framework.infrastructure.openai_llm import OpenAIConfig, OpenAILLM
from agent_framework.core.config import LLMProviderConfig, Config, load_config, save_config
import tempfile
import os


class TestOpenAIConfigSSL:
    """Test SSL configuration in OpenAIConfig."""

    def test_default_verify_ssl_is_true(self):
        """Default verify_ssl should be True (secure by default)."""
        config = OpenAIConfig(model="test-model")
        assert config.verify_ssl is True

    def test_verify_ssl_true(self):
        """Can set verify_ssl to True."""
        config = OpenAIConfig(model="test-model", verify_ssl=True)
        assert config.verify_ssl is True

    def test_verify_ssl_false_explicit(self):
        """Can explicitly set verify_ssl to False for local/dev environments."""
        config = OpenAIConfig(model="test-model", verify_ssl=False)
        assert config.verify_ssl is False


class TestOpenAILLMSSLClient:
    """Test that SSL config is passed to httpx client."""

    def test_client_created_with_verify_false(self):
        """httpx client should accept verify=False config."""
        config = OpenAIConfig(model="test-model", verify_ssl=False)
        llm = OpenAILLM(config)
        client = llm._get_client()
        assert client is not None
        assert llm.config.verify_ssl is False

    def test_client_created_with_verify_true(self):
        """httpx client should accept verify=True config."""
        config = OpenAIConfig(model="test-model", verify_ssl=True)
        llm = OpenAILLM(config)
        client = llm._get_client()
        assert client is not None
        assert llm.config.verify_ssl is True


class TestLLMProviderConfigSSL:
    """Test SSL field in LLMProviderConfig."""

    def test_default_verify_ssl(self):
        """Default verify_ssl should be True (secure by default)."""
        config = LLMProviderConfig()
        assert config.verify_ssl is True

    def test_custom_verify_ssl_false(self):
        """Can set verify_ssl to False for local/dev environments."""
        config = LLMProviderConfig(verify_ssl=False)
        assert config.verify_ssl is False


class TestConfigSSLIntegration:
    """Test SSL config flows through from config file to OpenAIConfig."""

    def test_config_loads_verify_ssl(self):
        """Config file with verify_ssl should be loaded correctly."""
        json_content = """{
    "llm": {
        "default": "test",
        "providers": {
            "test": {
                "type": "openai",
                "model": "gpt-4",
                "base_url": "https://api.test.com/v1",
                "api_key": "test-key",
                "verify_ssl": true
            }
        }
    }
}"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            provider = config.llm.providers["test"]
            assert provider.verify_ssl is True
        finally:
            os.unlink(temp_path)

    def test_config_default_verify_ssl_true(self):
        """Config without verify_ssl should default to True (secure)."""
        json_content = """{
    "llm": {
        "default": "test",
        "providers": {
            "test": {
                "type": "openai",
                "model": "gpt-4",
                "base_url": "https://api.test.com/v1"
            }
        }
    }
}"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            provider = config.llm.providers["test"]
            assert provider.verify_ssl is True
        finally:
            os.unlink(temp_path)

    def test_save_and_reload_verify_ssl(self):
        """verify_ssl should survive save/reload cycle."""
        config = Config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_config(config, temp_path)
            reloaded = load_config(temp_path)
            # Default provider should have verify_ssl=True (secure default)
            for name, prov in reloaded.llm.providers.items():
                assert prov.verify_ssl is True
        finally:
            os.unlink(temp_path)
