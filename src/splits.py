"""Data Processing and Preparation Pipeline for Football Match Datasets.

This module loads featured match data, encodes categorical outcomes, splits the
dataset chronologically into training, validation, and testing sets, and saves
the processed data to disk.
"""

from pathlib import Path
import typing
import numpy as np
import pandas as pd


def map_result(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes categorical match outcomes into ordinal numeric values.

    Maps 'H' (Home Win) -> 2, 'D' (Draw) -> 1, and 'A' (Away Win) -> 0.

    Args:
        df (pd.DataFrame): Input DataFrame containing a string 'result' column.

    Returns:
        pd.DataFrame: DataFrame with the 'result' column converted to integers.
    """
    # Map categorical string match outcomes to ordinal target integers
    df["result"] = df["result"].map({"H": 2, "D": 1, "A": 0})
    results = df["result"].unique().tolist()
    print(f"\nResult has been mapped to: {results}")
    
    return df


def load_match_data(path: Path | str) -> pd.DataFrame:
    """Loads feature-engineered match data and removes identification metadata columns.

    Args:
        path (Path | str): Path to the input CSV dataset.

    Returns:
        pd.DataFrame: DataFrame containing parsed dates, clean features, 
            and mapped target values.
    """
    print(f"\nLoading data from {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    
    # Drop non-predictive metadata and target-adjacent identification features
    unnecessary_columns = [
        "game_id",
        "competition_id",
        "season",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals",
        "aggregate",
        "home_club_name",
        "away_club_name",
    ]
    df.drop(columns=unnecessary_columns, inplace=True)

    # Encode target labels
    df = map_result(df)
    return df


def time_based_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits match data chronologically based on unique dates to prevent temporal leakage.

    Splits unique dates sequentially into training, validation, and test partitions, 
    ensuring all matches on a given date fall within the same set and that strict 
    chronological boundaries are preserved.

    Args:
        df (pd.DataFrame): Processed match DataFrame containing a 'date' column.
        train_frac (float, optional): Proportion of unique dates for training. Defaults to 0.7.
        val_frac (float, optional): Proportion of unique dates for validation. Defaults to 0.15.

    Returns:
        typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing 
            (train_set, val_set, test_set).

    Raises:
        AssertionError: If date boundaries overlap between train, validation, or test sets.
    """
    print("\nSpliting data...")
    # Ensure date column is properly parsed as datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        print("Converting date column to datetime")
        df["date"] = pd.to_datetime(df["date"])

    # Enforce strict chronological order
    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
    unique_dates = np.sort(df["date"].unique())
    total_dates = len(unique_dates)

    # Calculate index cutoffs based on unique dates
    train_date_size = int(total_dates * train_frac)
    val_date_size = int(total_dates * val_frac)

    # Segment unique dates into non-overlapping chronological sets
    train_dates = set(unique_dates[:train_date_size])
    val_dates = set(unique_dates[train_date_size : train_date_size + val_date_size])
    test_dates = set(unique_dates[train_date_size + val_date_size :])

    # Partition dataset using date membership
    train_set = df[df["date"].isin(train_dates)].copy()
    val_set = df[df["date"].isin(val_dates)].copy()
    test_set = df[df["date"].isin(test_dates)].copy()

    print(f"Train matches: {len(train_set)}")
    print(f"Val matches:   {len(val_set)}")
    print(f"Test matches:  {len(test_set)}")

    # Verify absence of temporal leakage across boundary dates
    assert train_set["date"].max() < val_set["date"].min(), (
        f"Train/Val overlap leak! Train max date ({train_set['date'].max()}) "
        f">= Val min date ({val_set['date'].min()})"
    )

    assert val_set["date"].max() < test_set["date"].min(), (
        f"Val/Test overlap leak! Val max date ({val_set['date'].max()}) "
        f">= Test min date ({test_set['date'].min()})"
    )

    print("Data split complete.")

    return train_set, val_set, test_set


def prep_and_save_data(
    dfs: typing.Sequence[pd.DataFrame], output_dir: Path | str
) -> None:
    """Prepares dataset partitions by dropping non-feature columns and exports them to CSV.

    Drops the 'date' column prior to saving to prevent model overuse of time indices, 
    purges remaining null values, and writes sets to disk.

    Args:
        dfs (typing.Sequence[pd.DataFrame]): Sequence containing train, validation, and test DataFrames.
        output_dir (Path | str): Directory path where CSV files will be written.
    """
    total_len = 0
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    unnecessary_columns = ["date"]
    names = ["train_set", "val_set", "test_set"]

    for df, name in zip(dfs, names):
        # Remove timestamp column prior to saving final model input features
        cols_to_drop = [col for col in unnecessary_columns if col in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)

        total_len += len(df)
        df.dropna(inplace=True)

        # Export split partition to CSV
        file_path = out_path / f"{name}.csv"
        df.to_csv(file_path, index=False)
        print(f"\n{name} data saved to {file_path}\n")
        
    print(f"Total rows of Data: {total_len}")


def main() -> None:
    """Executes the data preparation, chronological splitting, and saving pipeline."""
    processed_data_path = Path(__file__).resolve().parent.parent / "data" / "processed"
    featured_data_path = processed_data_path / "featured.csv"

    # Execute dataset ingestion, split, and storage pipeline
    df = load_match_data(featured_data_path)
    dfs = time_based_split(df)
    prep_and_save_data(dfs, processed_data_path)
    
    print("Columns in all sets:")
    t_df = dfs[0]
    print(t_df.columns.tolist())

    print("\nData has been saved succesfully.")


if __name__ == "__main__":
    main()