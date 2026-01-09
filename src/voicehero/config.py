"""Configuration management for VoiceHero."""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class VoiceHeroConfig(BaseModel):
    """VoiceHero configuration settings."""

    hotkey: list[str] = ["ctrl", "cmd"]
    model: str = "base"
    auto_paste: bool = True


def get_config_dir() -> Path:
    """Get the VoiceHero configuration directory."""
    config_dir = Path.home() / ".voicehero"
    config_dir.mkdir(exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    return get_config_dir() / "config.json"


def load_config() -> Optional[VoiceHeroConfig]:
    """Load configuration from disk."""
    config_path = get_config_path()
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            data = json.load(f)
            return VoiceHeroConfig(**data)
    except Exception:
        return None


def save_config(config: VoiceHeroConfig) -> None:
    """Save configuration to disk."""
    config_path = get_config_path()
    with open(config_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)


def get_recordings_dir() -> Path:
    """Get the directory for debug recordings."""
    recordings_dir = get_config_dir() / "recordings"
    recordings_dir.mkdir(exist_ok=True)
    return recordings_dir
