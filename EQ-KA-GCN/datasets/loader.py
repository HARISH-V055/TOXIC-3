"""
Dataset Loader Module for EQ-KA-GCN

Handles raw CSV dataset loading into a pandas DataFrame.
"""

import logging
from pathlib import Path
from typing import Union
import pandas as pd

# Use child logger under the main project logger
logger = logging.getLogger("EQ-KA-GCN.datasets.loader")


def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads dataset from a CSV file.

    Args:
        path (Union[str, Path]): Path to the CSV dataset file.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file or is empty.
    """
    file_path = Path(path)
    logger.info(f"Attempting to load raw dataset from: {file_path}")

    # Validate file existence
    if not file_path.exists():
        err_msg = f"Dataset file not found at path: {file_path.absolute()}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    if not file_path.is_file():
        err_msg = f"Target path is not a file: {file_path.absolute()}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded raw dataset. Shape: {df.shape}")
        return df
    except Exception as e:
        err_msg = f"Failed to read CSV dataset file: {str(e)}"
        logger.error(err_msg)
        raise ValueError(err_msg) from e
