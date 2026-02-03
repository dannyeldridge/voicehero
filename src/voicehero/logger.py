"""Logging utilities for VoiceHero."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class VoiceHeroLogger:
    """Logger for VoiceHero with optional file output."""

    def __init__(self, debug: bool = False, log_dir: Optional[Path] = None):
        """Initialize the logger.

        Args:
            debug: If True, log to file with detailed debug info
            log_dir: Directory to save log files (required if debug=True)
        """
        self.is_debug = debug
        self.log_dir = log_dir
        self.logger = logging.getLogger("voicehero")
        self.logger.setLevel(logging.DEBUG if self.is_debug else logging.INFO)

        # Remove any existing handlers
        self.logger.handlers.clear()

        # Always add console handler for errors
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(console_handler)

        # Add file handler if in debug mode
        if debug and log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_file = log_dir / f"voicehero-{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )
            self.logger.addHandler(file_handler)

            self.logger.info(f"VoiceHero debug logging started - Log file: {log_file}")
            self.logger.info(f"Python version: {sys.version}")
            self.logger.info(f"Platform: {sys.platform}")

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

    def exception(self, message: str) -> None:
        """Log an exception with traceback."""
        self.logger.exception(message)


# Global logger instance
_logger: Optional[VoiceHeroLogger] = None


def init_logger(debug: bool = False, log_dir: Optional[Path] = None) -> VoiceHeroLogger:
    """Initialize the global logger.

    Args:
        debug: If True, enable debug logging to file
        log_dir: Directory to save log files

    Returns:
        The initialized logger instance
    """
    global _logger
    _logger = VoiceHeroLogger(debug=debug, log_dir=log_dir)
    return _logger


def get_logger() -> VoiceHeroLogger:
    """Get the global logger instance.

    Returns:
        The logger instance

    Raises:
        RuntimeError: If logger has not been initialized
    """
    if _logger is None:
        raise RuntimeError("Logger not initialized. Call init_logger() first.")
    return _logger
