"""
Dataset Validator Module for EQ-KA-GCN

Provides verification tools to check the columns of the raw dataset DataFrame.
"""

import logging
import pandas as pd

# Use child logger under the main project logger
logger = logging.getLogger("EQ-KA-GCN.datasets.validator")


def validate_dataset(df: pd.DataFrame, smiles_column: str, target_column: str) -> None:
    """
    Validates that the required columns exist in the DataFrame.

    Args:
        df (pd.DataFrame): The loaded dataset DataFrame.
        smiles_column (str): Name of the SMILES column to validate.
        target_column (str): Name of the target toxicity endpoint column to validate.

    Raises:
        ValueError: If required columns are missing, or if the DataFrame is empty.
    """
    logger.info("Starting dataset schema validation...")

    if df is None:
        err_msg = "DataFrame validation failed: input DataFrame is None."
        logger.error(err_msg)
        raise ValueError(err_msg)

    if df.empty:
        err_msg = "DataFrame validation failed: input DataFrame is empty."
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Validate column names
    missing_columns = []
    if smiles_column not in df.columns:
        missing_columns.append(smiles_column)
    if target_column not in df.columns:
        missing_columns.append(target_column)

    if missing_columns:
        err_msg = (
            f"Dataset validation failed. Missing required columns: {missing_columns}. "
            f"Available columns in dataset: {list(df.columns)}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info(
        f"Dataset validation successful. Found target column: '{target_column}' "
        f"and SMILES column: '{smiles_column}'."
    )
