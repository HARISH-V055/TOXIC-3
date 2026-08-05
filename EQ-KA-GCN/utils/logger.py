"""
Logging Utility for EQ-KA-GCN

Sets up a clean, reusable logging pipeline that directs output to both stdout
and a persistent log file in the logs/ directory.
"""

import logging
import sys
from pathlib import Path


def setup_logger(log_dir: Path, log_filename: str = "training.log", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up and configures the logger.

    Args:
        log_dir (Path): Path to the directory where log files should be written.
        log_filename (str): Name of the file where log entries are stored.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: The configured Logger instance.
    """
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / log_filename

    # Get root logger
    logger = logging.getLogger("EQ-KA-GCN")
    logger.setLevel(level)

    # Avoid adding duplicate handlers if the logger is already initialized
    if not logger.handlers:
        # Formatter for log messages
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File Handler
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger
