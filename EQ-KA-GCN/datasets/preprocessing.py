"""
Dataset Preprocessing Module for EQ-KA-GCN

Responsible for cleaning raw toxicity datasets by selecting relevant columns,
removing missing values and duplicate SMILES, and filtering out invalid target values.
"""

import logging
import pandas as pd

# Use child logger under the main project logger
logger = logging.getLogger("EQ-KA-GCN.datasets.preprocessing")


def clean_dataset(
    df: pd.DataFrame, smiles_column: str, target_column: str
) -> pd.DataFrame:
    """
    Cleans the input DataFrame by:
      - Keeping only the SMILES and the configured target columns.
      - Removing missing values (NaNs, None, or empty/whitespace SMILES).
      - Removing invalid labels (keeps only 0 and 1).
      - Removing duplicate molecules based on the SMILES column.
      - Resetting the index.

    Args:
        df (pd.DataFrame): The loaded dataset DataFrame.
        smiles_column (str): The column name containing SMILES notations.
        target_column (str): The column name containing target toxicity endpoints.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    initial_count = len(df)
    logger.info(f"Starting dataset cleaning of {initial_count} rows...")

    # 1. Keep only SMILES and target_column
    df_clean = df[[smiles_column, target_column]].copy()

    # 2. Remove missing SMILES (NaN or empty/whitespace strings)
    # Convert empty strings or whitespace strings to NaN for uniform dropping
    if df_clean[smiles_column].dtype == object:
        df_clean[smiles_column] = df_clean[smiles_column].astype(str).str.strip()
        df_clean[smiles_column] = df_clean[smiles_column].replace("", pd.NA)

    # 3. Remove missing target values
    df_clean = df_clean.dropna(subset=[smiles_column, target_column])  # type: ignore
    count_after_nan = len(df_clean)
    missing_removed = initial_count - count_after_nan
    logger.info(f"Removed {missing_removed} rows with missing values.")

    # 4. Remove invalid target values (accept only 0 and 1)
    # Ensure target column is numeric
    df_clean[target_column] = pd.to_numeric(df_clean[target_column], errors="coerce")
    df_clean = df_clean.dropna(subset=[target_column])  # type: ignore
    
    # Filter target values to retain only 0 and 1 (or float representations 0.0 and 1.0)
    df_clean = df_clean[df_clean[target_column].isin([0.0, 1.0, 0, 1])]
    
    # Cast target column to integer type
    df_clean[target_column] = df_clean[target_column].astype(int)
    
    count_after_labels = len(df_clean)
    invalid_labels_removed = count_after_nan - count_after_labels
    logger.info(f"Removed {invalid_labels_removed} rows with invalid labels.")

    # 5. Remove duplicate molecules based on SMILES
    df_clean = df_clean.drop_duplicates(subset=[smiles_column], keep="first")
    final_count = len(df_clean)
    duplicates_removed = count_after_labels - final_count
    logger.info(f"Removed {duplicates_removed} duplicate molecules based on SMILES.")

    # 6. Reset index
    df_clean = df_clean.reset_index(drop=True)

    logger.info(
        f"Data cleaning finished. Final count: {final_count} rows. "
        f"Total rows removed: {initial_count - final_count}."
    )

    return df_clean
