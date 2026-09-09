"""Kaggle dataset ingestion and validation module.

Handles downloading raw game data via the Kaggle API, caching it locally,
and enforcing strict data integrity checks before downstream processing.
"""

import pandas as pd
from pathlib import Path
import os
import kaggle

DATA_SRC = "davidcariboo/player-scores"
FILENAME = "games.csv"


def load_df(filepath):
    df = pd.read_csv(filepath)
    print(f"The columns in the dataframe are: {list(df.columns)}")
    print(df.head(10))

    return df


RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
FILENAME = "games.csv"
FILE_PATH = RAW_DIR / FILENAME


def load_raw_data() -> pd.DataFrame:
    """Fetch raw game data from Kaggle or load it from local cache if present.

    Attempts Kaggle API authentication to download the file if not already 
    cached at FILE_PATH. Handles authentication and runtime exceptions gracefully.
    """
    if os.path.exists(FILE_PATH):
        print(f"📦 Data already available locally at '{FILE_PATH}'. Skipping download.")
        df = load_df(FILE_PATH)
        return df

    output_folders = ["./data", "./data/raw", "./data/processed"]
    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)

    try:
        print("Attempting to authenticate...")
        kaggle.api.authenticate()

        print("Fetching Kaggle Data...")
        kaggle.api.dataset_download_file(DATA_SRC, file_name=FILENAME, path=RAW_DIR)
        print(f"Data download to: {RAW_DIR}")

        df = load_df(filepath=FILE_PATH)
        return df

    except OSError as e:
        print(f"\n❌ Authentication File Error: {e}")
        print(
            "Tip: If you're providing variables above, ensure they are exact strings without trailing spaces."
        )

        return None

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return None


def validate_raw(df: pd.DataFrame) -> None:
    """Validate dataset schema integrity, checking for missing required columns, 
    null values, and duplicate primary key records.
    """
    if df.empty:
        raise ValueError("Fetched data is empty!!!")

    required_columns = ["game_id", "date", "home_club_goals", "away_club_goals"]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise KeyError(f"❌ Missing critical columns from dataset: {missing_columns}")

    for col in required_columns:
        if df[col].isnull().any():
            total_nulls = df[col].isnull().sum()
            raise ValueError(f"❌ Column '{col}' contains {total_nulls} null/missing values!")

    if not df['game_id'].is_unique:
        raise ValueError("❌ Duplicate game_id rows detected!")

    print("No Issues Detected. Data is safe to use.")


if __name__ == "__main__":

    df = load_raw_data()
    validate_raw(df)