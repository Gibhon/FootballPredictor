"""
Data Processing and Preparation Pipeline for Football Match Datasets.

This module loads featured match data, encodes categorical outcomes, splits the
dataset chronologically into training, validation, and testing sets, and saves
the processed data to disk.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def map_result(df):
    distinct_results_before = df["result"].unique().tolist()
    print(f"Distinct results before mapping: {distinct_results_before}")
    df["result"] = df["result"].map({"H": 2, "D": 1, "A": 0})
    distinct_results_after = df["result"].unique().tolist()
    print(f"Distinct results after mapping: {distinct_results_after}")
    print(f"Dtype of 'result' column after mapping: {df['result'].dtype}")
    return df


def load_match_data(path):
    print("-" * 150)
    print(f"Loading data from {path}...............")
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Data types:\n{df.dtypes}")
    unneccessary_columns = [
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
    df.drop(columns=unneccessary_columns, inplace=True)

    df = map_result(df)

    print("Columns after dropping unneccessary columns:")
    print(list(df.columns))
    return df


def time_based_split(df, train_frac=0.7, val_frac=0.15):
    """
    Splits match data chronologically based on unique dates to prevent temporal
    data leakage across training, validation, and test sets.
    """
    print("-" * 150)
    print(f"Date column dtype: {df['date'].dtype}")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        print("Converting date column to datetime")
        df["date"] = pd.to_datetime(df["date"])
        print(f"Date column dtype after conversion: {df['date'].dtype}")

    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)

    unique_dates = np.sort(df["date"].unique())
    total_dates = len(unique_dates)

    train_date_size = int(total_dates * train_frac)
    val_date_size = int(total_dates * val_frac)

    train_dates = set(unique_dates[:train_date_size])
    val_dates = set(unique_dates[train_date_size : train_date_size + val_date_size])
    test_dates = set(unique_dates[train_date_size + val_date_size :])

    train_set = df[df["date"].isin(train_dates)].copy()
    val_set = df[df["date"].isin(val_dates)].copy()
    test_set = df[df["date"].isin(test_dates)].copy()

    print(f"Total unique dates: {total_dates}")
    print(
        f"Train matches: {len(train_set)} across {len(train_dates)} dates ({train_set['date'].min().date()} to {train_set['date'].max().date()})"
    )
    print(
        f"Val matches:   {len(val_set)} across {len(val_dates)} dates ({val_set['date'].min().date()} to {val_set['date'].max().date()})"
    )
    print(
        f"Test matches:  {len(test_set)} across {len(test_dates)} dates ({test_set['date'].min().date()} to {test_set['date'].max().date()})"
    )

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


def prep_and_save_data(dfs, output_dir):
    print("-" * 150)
    total_len = 0
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    unnecessary_columns = ["date"]
    names = ["train_set", "val_set", "test_set"]

    for df, name in zip(dfs, names):
        print(f"Preparing {name} for saving...")

        cols_to_drop = [col for col in unnecessary_columns if col in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)

        print(f"Length before dropping NA: {len(df)}")
        total_len +=len(df)
        df.dropna(inplace=True)
        print(f"Length after dropping NA: {len(df)}")

        print(f"columns after dropping unnecessary columns: {list(df.columns)}")

        file_path = out_path / f"{name}.csv"
        print(f"Saving data to {file_path}...")
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}\n")
    print(f"Total rows of Data: {total_len}")


def main():
    processed_data_path = Path(__file__).parent.parent / "data" / "processed"
    featured_data_path = processed_data_path / "featured.csv"
    df = load_match_data(featured_data_path)
    dfs = [*time_based_split(df)]
    prep_and_save_data(dfs, processed_data_path)
    print("Done!")


if __name__ == "__main__":
    main()
