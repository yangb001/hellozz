"""Configuration management - Load and access JSON configuration."""
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    """Memory subsystem configuration."""
    short_term_size: int = 20
    vector_db: str = "lancedb"
    vector_path: str = "./data/vectors"
    embedding_model: str = "all-MiniLM-L6-v2"
    extraction_trigger: str = "smart"
    extraction_model: str = "ollama/llama3"


@dataclass
class LLMProviderConfig:
    """LLM provider configuration."""
    type: str = "ollama"
    model: str = "llama3"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    verify_ssl: bool = True


@dataclass
class LLMConfig:
    """LLM subsystem configuration."""
    default: str = "ollama"
    providers: Dict[str, LLMProviderConfig] = field(default_factory=dict)


@dataclass
class SQLiteConfig:
    """SQLite configuration."""
    path: str = "./data/sessions.db"


@dataclass
class PlannerConfig:
    """Planner configuration."""
    type: str = "planners.react_planner.ReActPlanner"


@dataclass
class LoggingConfig:
    """Logging subsystem configuration.

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


@dataclass
class Config:
    """
    Root configuration container.

    Attributes:
        sqlite: SQLite database configuration
        memory: Memory subsystem configuration
        llm: LLM gateway configuration
        planner: Planner configuration
        logging: Logging subsystem configuration
    """
    sqlite: SQLiteConfig = field(default_factory=SQLiteConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "sqlite": {
                "path": self.sqlite.path,
            },
            "memory": {
                "short_term_size": self.memory.short_term_size,
                "vector_db": self.memory.vector_db,
                "vector_path": self.memory.vector_path,
                "embedding_model": self.memory.embedding_model,
                "extraction": {
                    "trigger": self.memory.extraction_trigger,
                    "model": self.memory.extraction_model,
                },
            },
            "llm": {
                "default": self.llm.default,
                "providers": {
                    name: {
                        "type": prov.type,
                        "model": prov.model,
                        "base_url": prov.base_url,
                        "api_key": prov.api_key,
                        "verify_ssl": prov.verify_ssl,
                    }
                    for name, prov in self.llm.providers.items()
                },
            },
            "planner": self.planner.type,
            "logging": {
                "level": self.logging.level,
                "log_dir": self.logging.log_dir,
                "max_bytes": self.logging.max_bytes,
                "backup_count": self.logging.backup_count,
                "console_output": self.logging.console_output,
                "file_output": self.logging.file_output,
            },
        }


def _dict_to_provider_config(data: Dict[str, Any]) -> LLMProviderConfig:
    """Convert dictionary to LLMProviderConfig."""
    return LLMProviderConfig(
        type=data.get("type", "ollama"),
        model=data.get("model", "llama3"),
        base_url=data.get("base_url", "http://localhost:11434"),
        api_key=data.get("api_key"),
        verify_ssl=data.get("verify_ssl", True),
    )


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the JSON config file. If None, looks for
                     config.json in the current directory and parent directories.

    Returns:
        Config object with loaded values (defaults for missing keys)

    Raises:
        FileNotFoundError: If config_path is specified but not found
        json.JSONDecodeError: If the file contains invalid JSON
    """
    if config_path is None:
        search_paths = [
            Path.cwd() / "config.json",
            Path.cwd().parent / "config.json",
            Path(__file__).parent.parent.parent / "config.json",
        ]
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path and not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            data = json.loads(content) if content else {}
    else:
        data = {}

    sqlite_data = data.get("sqlite", {})
    memory_data = data.get("memory", {})
    llm_data = data.get("llm", {})
    planner_data = data.get("planner", "")

    extraction_data = memory_data.get("extraction", {})

    sqlite_config = SQLiteConfig(
        path=sqlite_data.get("path", "./data/sessions.db")
    )

    memory_config = MemoryConfig(
        short_term_size=memory_data.get("short_term_size", 20),
        vector_db=memory_data.get("vector_db", "lancedb"),
        vector_path=memory_data.get("vector_path", "./data/vectors"),
        embedding_model=memory_data.get("embedding_model", "all-MiniLM-L6-v2"),
        extraction_trigger=extraction_data.get("trigger", "smart"),
        extraction_model=extraction_data.get("model", "ollama/llama3"),
    )

    llm_providers = {}
    for name, prov_data in llm_data.get("providers", {}).items():
        llm_providers[name] = _dict_to_provider_config(prov_data)

    llm_config = LLMConfig(
        default=llm_data.get("default", "ollama"),
        providers=llm_providers,
    )

    planner_config = PlannerConfig(
        type=planner_data or "planners.react_planner.ReActPlanner"
    )

    logging_data = data.get("logging", {})
    logging_config = LoggingConfig(
        level=logging_data.get("level", "INFO"),
        log_dir=logging_data.get("log_dir", "logs"),
        max_bytes=logging_data.get("max_bytes", 10 * 1024 * 1024),
        backup_count=logging_data.get("backup_count", 5),
        console_output=logging_data.get("console_output", True),
        file_output=logging_data.get("file_output", True),
    )

    config = Config(
        sqlite=sqlite_config,
        memory=memory_config,
        llm=llm_config,
        planner=planner_config,
        logging=logging_config,
    )

    return config


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to a JSON file.

    Args:
        config: Config object to save.
        config_path: Path to save the JSON file.
    """
    data = config.to_dict()
    dir_path = os.path.dirname(config_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)