"""Configuration loading for hearth.

Config lives at ~/.config/hearth/config.toml and is created with defaults on
first run. Every value can also be overridden by a HEARTH_* environment
variable, which is what makes remote/SSH use convenient.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import tomli_w

CONFIG_DIR = Path(os.environ.get("HEARTH_CONFIG_DIR", Path.home() / ".config" / "hearth"))
CONFIG_PATH = CONFIG_DIR / "config.toml"
DATA_DIR = Path(os.environ.get("HEARTH_DATA_DIR", Path.home() / ".local" / "share" / "hearth"))


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass
class TextModelConfig:
    repo: str = "mlx-community/Qwen3.6-35B-A3B-8bit"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    # Trim history sent to the model so long threads don't blow the context window.
    max_history_messages: int = 40
    system_prompt: str = "You are a helpful assistant running locally on the user's own machine."
    # Qwen3.6 is a hybrid reasoning model. Thinking is off by default so chat
    # feels snappy; flip it per-request from the CLI (--think) or the web UI.
    enable_thinking: bool = False
    thinking_budget: int = 2048


@dataclass
class ImageModelConfig:
    repo: str = "mlx-community/Qwen-Image-2512-4bit"
    steps: int = 20
    width: int = 1024
    height: int = 1024
    guidance: float = 4.0


@dataclass
class MemoryConfig:
    # Seconds a model may sit unused before it is evicted to free unified memory.
    # 0 disables eviction (models stay resident forever).
    idle_evict_seconds: int = 900
    # When true, only one model is resident at a time: loading the image model
    # evicts the text model and vice versa. Safer on machines under ~64GB.
    exclusive: bool = False


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    text: TextModelConfig = field(default_factory=TextModelConfig)
    image: ImageModelConfig = field(default_factory=ImageModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @property
    def db_path(self) -> Path:
        return DATA_DIR / "hearth.db"

    @property
    def image_dir(self) -> Path:
        return DATA_DIR / "images"

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.server.host in ("0.0.0.0", "::") else self.server.host
        return f"http://{host}:{self.server.port}"


def _merge(section: Any, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if hasattr(section, key):
            setattr(section, key, value)


def _apply_env(cfg: Config) -> None:
    """HEARTH_* env vars win over the file. Handy over SSH and in scripts."""
    env_map = {
        "HEARTH_HOST": (cfg.server, "host", str),
        "HEARTH_PORT": (cfg.server, "port", int),
        "HEARTH_TEXT_MODEL": (cfg.text, "repo", str),
        "HEARTH_IMAGE_MODEL": (cfg.image, "repo", str),
        "HEARTH_MAX_TOKENS": (cfg.text, "max_tokens", int),
        "HEARTH_TEMPERATURE": (cfg.text, "temperature", float),
        "HEARTH_THINK": (cfg.text, "enable_thinking", lambda v: v.lower() in ("1", "true", "yes")),
        "HEARTH_IMAGE_STEPS": (cfg.image, "steps", int),
        "HEARTH_IDLE_EVICT": (cfg.memory, "idle_evict_seconds", int),
        "HEARTH_EXCLUSIVE": (cfg.memory, "exclusive", lambda v: v.lower() in ("1", "true", "yes")),
    }
    for var, (section, attr, cast) in env_map.items():
        raw = os.environ.get(var)
        if raw:
            setattr(section, attr, cast(raw))


def load() -> Config:
    cfg = Config()
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("rb") as fh:
            data = tomllib.load(fh)
        _merge(cfg.server, data.get("server", {}))
        _merge(cfg.text, data.get("models", {}).get("text", {}))
        _merge(cfg.image, data.get("models", {}).get("image", {}))
        _merge(cfg.memory, data.get("memory", {}))
    _apply_env(cfg)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.image_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def write_default(force: bool = False) -> Path:
    """Materialize a commented default config file."""
    if CONFIG_PATH.exists() and not force:
        return CONFIG_PATH
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    payload = {
        "server": asdict(cfg.server),
        "models": {"text": asdict(cfg.text), "image": asdict(cfg.image)},
        "memory": asdict(cfg.memory),
    }
    with CONFIG_PATH.open("wb") as fh:
        tomli_w.dump(payload, fh)
    return CONFIG_PATH
