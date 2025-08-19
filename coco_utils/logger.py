"""Logging configuration for coco_utils package.

This module provides a centralized logging configuration that can be used
throughout the coco_utils package for consistent logging behavior.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: The name of the logger (typically __name__ of the calling module).
        level: Optional logging level. If not provided, defaults to INFO.

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure if the logger doesn't have handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Set level
    if level is not None:
        logger.setLevel(level)
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    return logger


def configure_file_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> None:
    """Configure file-based logging for the entire package.

    Args:
        log_file: Path to the log file. If None, uses 'coco_utils.log' in current directory.
        level: Logging level to use.
    """
    if log_file is None:
        log_file = Path("coco_utils.log")

    # Get the root logger for the package
    logger = logging.getLogger("coco_utils")

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)
    logger.setLevel(level)
