"""Football match data cleaning pipeline.

Filters domestic league matches, computes match outcomes (H/D/A), parses dates,
extracts round numbers, and exports the essential cleaned features.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def filter_domestic_league(df: pd.DataFrame) -> pd.DataFrame:
    """Filters the dataset to include only domestic league matches.

    Args:
        df (pd.DataFrame): Raw football match DataFrame containing a 
            'competition_type' column.

    Returns:
        pd.DataFrame: A filtered copy of the DataFrame containing domestic 
            league games.
    """
    print("\nFiltering in domestic data....")
    # Retain only domestic league records to ensure consistent analysis scope
    filtered_data = df[df["competition_type"] == "domestic_league"].copy()

    print("Data has been filtered")

    return filtered_data


def derive_result(df: pd.DataFrame) -> pd.DataFrame:
    """Map match goals to 'H' (Home win), 'D' (Draw), or 'A' (Away win).

    Drops rows where the outcome cannot be resolved or goals data is missing.

    Args:
        df (pd.DataFrame): DataFrame containing 'home_club_goals' and 
            'away_club_goals' columns.

    Returns:
        pd.DataFrame: DataFrame with an added categorical 'result' column.
    """
    # Evaluate goal differentials to establish outcome categories
    conditions = [
        df["home_club_goals"] > df["away_club_goals"],
        df["home_club_goals"] == df["away_club_goals"],
        df["home_club_goals"] < df["away_club_goals"],
    ]
    choices = ["H", "D", "A"]

    # Assign outcome label; fallback to UNKNOWN for unresolved/null entries
    df["result"] = np.select(choicelist=choices, condlist=conditions, default="UNKNOWN")
    
    # Filter out unresolvable outcomes
    df = df[df["result"] != "UNKNOWN"].copy()

    return df


def parse_date(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the 'date' string column to standard pandas datetime objects.

    Args:
        df (pd.DataFrame): DataFrame containing a string-formatted 'date' column.

    Returns:
        pd.DataFrame: DataFrame with the 'date' column cast to datetime64.
    """
    df["date"] = pd.to_datetime(df["date"])
    print("\nSucessfully converted 'date' column to datetime.")

    return df


def add_numeric_round(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric matchday digits from the 'round' string column.

    Parses strings (e.g., '14. Matchday') into integer values and drops rows 
    where numerical extraction fails.

    Args:
        df (pd.DataFrame): DataFrame containing the raw 'round' text column.

    Returns:
        pd.DataFrame: DataFrame with a new integer 'matchday' column.
    """
    print("\nAdding numeric rounds....")
    len_before = len(df)
    
    # Extract first sequence of digits from round descriptions
    df["matchday"] = df["round"].str.extract(r"(\d+)")

    invalid_outputs = df["matchday"].isna().sum()
    print(f"Invalid_outputs: {invalid_outputs}")

    # Remove records missing valid round numbers and cast to integer type
    df = df.dropna(subset=["matchday"])
    df["matchday"] = df["matchday"].astype(int)
    len_after = len(df)

    print(f"Invalid rounds dropped: {len_before - len_after}")
    print("Column 'matchday' has been successfully created!!!")

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selects target features required for modeling and drops incomplete rows.

    Args:
        df (pd.DataFrame): Transformed DataFrame containing all intermediate features.

    Returns:
        pd.DataFrame: Cleaned DataFrame containing only specified core features 
            without null values.
    """
    essential_columns = [
        "game_id",
        "competition_id",
        "season",
        "matchday",
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals",
        "home_club_position",
        "away_club_position",
        "aggregate",
        "home_club_name",
        "away_club_name",
        "result",
    ]

    # Subset required feature columns and purge incomplete records
    essential_only_df = df[essential_columns]
    clean_df = essential_only_df.dropna()
    print("\nColumns Remaining: ")
    print(essential_columns)

    return clean_df


def save_cleandf(df: pd.DataFrame, filepath: Path) -> None:
    """Save clean DataFrame to disk, prompting user confirmation if file exists.

    Args:
        df (pd.DataFrame): The processed DataFrame to export.
        filepath (Path): Destination file path for the CSV output.
    """
    print("\nSaving Data...")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Prompt user before overwriting pre-existing processed dataset
    if filepath.exists():
        print("Clean data already exists.")
        while True:
            decision = input("Overide the current file(Y/N): ").strip().lower()
            if decision == "y":
                df.to_csv(filepath, index=False)
                print("Data has been overridden")
                break
            elif decision == "n":
                print("Keeping existing clean data.")
                break
            else:
                print("Invalid Input!!!")
    else:
        df.to_csv(filepath, index=False)
        print(f"Clean Data has been saved to {filepath}.")


def main() -> None:
    """Runs the sequential data ingestion, cleaning, transformation, and export pipeline."""
    # Resolve project root relative path structure
    data_path = Path(__file__).resolve().parent.parent.parent / "data"
    raw_data_path = data_path / "raw" / "games.csv"
    clean_data_path = data_path / "processed" / "clean_data.csv"

    # Execution pipeline steps
    df = pd.read_csv(raw_data_path)
    df = filter_domestic_league(df)
    df = derive_result(df)
    df = parse_date(df)
    df = add_numeric_round(df)
    df = select_features(df)

    save_cleandf(df, clean_data_path)


if __name__ == "__main__":
    main()