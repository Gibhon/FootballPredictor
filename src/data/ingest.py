"""Kaggle dataset ingestion and validation module.

Handles downloading raw game data via the Kaggle API, caching it locally,
and enforcing strict data integrity checks before downstream processing.
"""

from pathlib import Path
import os
import typing
import pandas as pd
import kaggle

# Dataset target configurations
DATA_SRC: str = "davidcariboo/player-scores"
FILENAME: str = "games.csv"

# Resolve target raw directory relative to project root
RAW_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
FILE_PATH: Path = RAW_DIR / FILENAME


def load_df(filepath: Path | str) -> pd.DataFrame:
    """Reads a CSV file into a pandas DataFrame and displays column names.

    Args:
        filepath (Path | str): Absolute or relative path to the target CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(filepath)
    print("The columns in the dataframe are:")
    print(list(df.columns))

    return df


def load_raw_data() -> typing.Optional[pd.DataFrame]:
    """Retrieves raw dataset from local cache or downloads it via the Kaggle API.

    Ensures target local directories exist before downloading. If local files are 
    present, API authentication and download steps are bypassed.

    Returns:
        typing.Optional[pd.DataFrame]: Loaded DataFrame if successful, else None.
    """
    print("Fetching Data..........")

    # Serve cached file if already downloaded locally
    if os.path.exists(FILE_PATH):
        print(f"Data already available locally at '{FILE_PATH}'. Skipping download.")
        return load_df(FILE_PATH)

    # Ensure project data directories exist
    output_folders = ["./data", "./data/raw", "./data/processed"]
    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)

    try:
        # Authenticate using local Kaggle credentials (~/.kaggle/kaggle.json)
        print("Attempting to authenticate...")
        kaggle.api.authenticate()

        print("Fetching Kaggle Data...")
        kaggle.api.dataset_download_file(DATA_SRC, file_name=FILENAME, path=RAW_DIR)
        print(f"Data download to: {RAW_DIR}")

        return load_df(filepath=FILE_PATH)

    except OSError as e:
        # Catch credential file location/access errors
        print(f"\nAuthentication File Error: {e}")
        print(
            "Tip: If you're providing variables above, ensure they are exact strings without trailing spaces."
        )
        return None

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return None


def validate_raw(df: pd.DataFrame) -> None:
    """Validate dataset schema integrity and data quality.

    Checks for non-empty input, presence of essential columns, null records in 
    critical features, and uniqueness of the primary key field.

    Args:
        df (pd.DataFrame): The raw input DataFrame to validate.

    Raises:
        ValueError: If the DataFrame is empty, contains null values in critical 
            columns, or has duplicate primary key records.
        KeyError: If required schema columns are missing.
    """
    print("\n \nValidating data...")

    if df.empty:
        raise ValueError("Fetched data is empty!!!")

    required_columns = ["game_id", "date", "home_club_goals", "away_club_goals"]
    missing_columns = [column for column in required_columns if column not in df.columns]

    # Verify structural schema compliance
    if missing_columns:
        raise KeyError(f"Missing critical columns from dataset: {missing_columns}")

    # Ensure critical fields have no null/missing entries
    for col in required_columns:
        if df[col].isnull().any():
            total_nulls = df[col].isnull().sum()
            raise ValueError(f"Column '{col}' contains {total_nulls} null/missing values!")

    # Verify primary key uniqueness constraint
    if not df["game_id"].is_unique:
        raise ValueError("Duplicate game_id rows detected!")

    print("No Issues Detected. Data is safe to use.")


if __name__ == "__main__":
    df = load_raw_data()
    if df is not None:
        validate_raw(df)