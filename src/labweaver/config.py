"""Explicit model configuration; credentials never enter run reports."""

from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import dotenv_values


class ConfigurationError(ValueError):
    """A missing or invalid local model setting."""


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    base_url: str
    api_key: str = field(repr=False)
    timeout: float = 30.0


def load_config(
    env_file: str | Path | None = None, *, use_default_env: bool = True
) -> ModelConfig:
    path = Path(env_file) if env_file is not None else (Path(".env") if use_default_env else None)
    if env_file is not None and not path.is_file():
        raise ConfigurationError("The selected configuration file does not exist.")
    values = dotenv_values(path) if path is not None and path.is_file() else {}

    def setting(name: str) -> str:
        return str(os.environ.get(name, values.get(name) or "")).strip()

    required = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_ID")
    missing = [name for name in required if not setting(name)]
    if missing:
        raise ConfigurationError("Missing model settings: " + ", ".join(missing))
    try:
        timeout = float(setting("LLM_TIMEOUT") or "30")
        if not 0 < timeout <= 120:
            raise ValueError
    except ValueError:
        raise ConfigurationError("LLM_TIMEOUT must be a finite number between 0 and 120.") from None
    return ModelConfig(
        model_id=setting("LLM_MODEL_ID"),
        base_url=setting("LLM_BASE_URL"),
        api_key=setting("LLM_API_KEY"),
        timeout=timeout,
    )


def create_model(config: ModelConfig):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.model_id,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=0,
        timeout=config.timeout,
        max_retries=0,
        max_tokens=1200,
    )

