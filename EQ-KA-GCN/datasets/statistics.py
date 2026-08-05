"""
Dataset Statistics Module for EQ-KA-GCN

Calculates, logs, and returns key metrics about the Tox21 dataset and target
toxicological endpoints, including positive/negative distribution and data cleanup counts.
"""

import logging
from typing import Dict, Union
import pandas as pd

# Use child logger under the main project logger
logger = logging.getLogger("EQ-KA-GCN.datasets.statistics")


def dataset_statistics(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    smiles_column: str,
) -> Dict[str, Union[int, float, str]]:
    """
    Generates statistics for raw vs cleaned datasets.

    Args:
        raw_df (pd.DataFrame): The raw input dataset.
        clean_df (pd.DataFrame): The cleaned dataset.
        dataset_name (str): Friendly name of the dataset.
        target_column (str): Configuration key for the target column.
        smiles_column (str): Configuration key for the SMILES column.

    Returns:
        Dict[str, Union[int, float, str]]: Dictionary of all computed statistics.
    """
    logger.info("Generating dataset statistics...")

    total_samples = len(clean_df)

    # 1. Counts of positive / negative samples in cleaned df
    pos_samples = int((clean_df[target_column] == 1).sum())
    neg_samples = int((clean_df[target_column] == 0).sum())

    # 2. Percentages
    pos_pct = round((pos_samples / total_samples) * 100, 2) if total_samples > 0 else 0.0
    neg_pct = round((neg_samples / total_samples) * 100, 2) if total_samples > 0 else 0.0

    # 3. Calculate missing values removed
    # A row is considered missing if either SMILES or target is missing (empty, null, or whitespace)
    is_missing_smiles = raw_df[smiles_column].isna()
    if raw_df[smiles_column].dtype == object:
        is_missing_smiles = is_missing_smiles | (raw_df[smiles_column].astype(str).str.strip() == "")
    
    is_missing_target = raw_df[target_column].isna()
    
    any_missing_mask = is_missing_smiles | is_missing_target
    missing_removed = int(any_missing_mask.sum())

    # 4. Calculate duplicate molecules removed
    # Duplicates are assessed on the valid (non-missing) SMILES remaining
    df_valid_smiles = raw_df[~any_missing_mask]
    duplicates_removed = int(df_valid_smiles[smiles_column].duplicated().sum())

    stats = {
        "Dataset Name": dataset_name,
        "Target Column": target_column,
        "Total Samples": total_samples,
        "Positive Samples": pos_samples,
        "Negative Samples": neg_samples,
        "Positive Percentage": pos_pct,
        "Negative Percentage": neg_pct,
        "Number of Duplicate Molecules Removed": duplicates_removed,
        "Number of Missing Values Removed": missing_removed,
    }

    # Log all statistics
    logger.info("==================================================================")
    logger.info(f"DATASET STATISTICS SUMMARY - {dataset_name}")
    logger.info("==================================================================")
    logger.info(f"Target Column:                         {target_column}")
    logger.info(f"Total Samples (Cleaned):               {total_samples}")
    logger.info(f"Positive (Toxic) Samples:              {pos_samples} ({pos_pct}%)")
    logger.info(f"Negative (Non-Toxic) Samples:          {neg_samples} ({neg_pct}%)")
    logger.info(f"Missing Values Removed:                {missing_removed}")
    logger.info(f"Duplicate Molecules Removed:           {duplicates_removed}")
    logger.info("==================================================================")

    return stats
