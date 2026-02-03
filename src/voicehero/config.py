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


class VoiceHeroStats(BaseModel):
    """VoiceHero cumulative statistics."""

    total_words: int = 0
    total_transcriptions: int = 0


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


def get_stats_path() -> Path:
    """Get the path to the statistics file."""
    return get_config_dir() / "stats.json"


def load_stats() -> VoiceHeroStats:
    """Load statistics from disk, or return default stats if not found."""
    stats_path = get_stats_path()
    if not stats_path.exists():
        return VoiceHeroStats()

    try:
        with open(stats_path) as f:
            data = json.load(f)
            return VoiceHeroStats(**data)
    except Exception:
        return VoiceHeroStats()


def save_stats(stats: VoiceHeroStats) -> None:
    """Save statistics to disk."""
    stats_path = get_stats_path()
    with open(stats_path, "w") as f:
        json.dump(stats.model_dump(), f, indent=2)
