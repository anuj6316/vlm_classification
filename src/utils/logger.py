# src/utils/logger.py
# ============================================================
# Structured Logging Setup
# ============================================================
# Provides a pre-configured logger with Rich console output for
# human-readable local development logs. Colors, timestamps, and
# module names are included automatically.
#
# Usage:
#   from src.utils.logger import get_logger
#   logger = get_logger(__name__)
#   logger.info("Processing page 1 of 10")
# ============================================================

import logging
import sys

from rich.logging import RichHandler

from config.settings import settings


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a pre-configured logger with Rich formatting.

    Args:
        name: Logger name, typically __name__ from the calling module.
              This appears in log output to identify the source.

    Returns:
        A logging.Logger instance with Rich console handler attached.

    Example:
        >>> logger = get_logger("src.ocr.engine")
        >>> logger.info("Model loaded in 2.3s")
        [10:30:45] INFO     src.ocr.engine — Model loaded in 2.3s
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        # Parse log level from settings (e.g., "INFO" → logging.INFO)
        level = getattr(logging, settings.log_level.upper(), logging.INFO)
        logger.setLevel(level)

        # Rich handler: colorized, timestamped console output
        rich_handler = RichHandler(
            level=level,
            rich_tracebacks=True,       # Pretty-print exception tracebacks
            tracebacks_show_locals=True, # Show local variables in tracebacks
            show_time=True,
            show_path=False,            # Module name is enough, skip file paths
            markup=True,                # Allow Rich markup in log messages
        )

        # Format: just the message — Rich handles time/level/module decoration
        formatter = logging.Formatter("%(name)s — %(message)s")
        rich_handler.setFormatter(formatter)

        logger.addHandler(rich_handler)

        # Prevent log propagation to root logger (avoids duplicate messages)
        logger.propagate = False

    return logger
