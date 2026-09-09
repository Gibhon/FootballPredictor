"""Football match data cleaning pipeline.

Filters domestic league matches, computes match outcomes (H/D/A), parses dates,
extracts round numbers, and exports the essential cleaned features.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def filter_domestic_league(df: pd.DataFrame) -> pd.DataFrame:
    filtered_data = df[df["competition_type"] == "domestic_league"].copy()

    print("Data has been filtered")
    print(f"New data Shape: {filtered_data.shape}")

    return filtered_data


def derive_result(df: pd.DataFrame) -> pd.DataFrame:
    """Map match goals to 'H' (Home win), 'D' (Draw), or 'A' (Away win),

    dropping rows where outcome cannot be resolved.
    """
    conditions = [
        df["home_club_goals"] > df["away_club_goals"],
        df["home_club_goals"] == df["away_club_goals"],
        df["home_club_goals"] < df["away_club_goals"],
    ]
    choices = ["H", "D", "A"]

    df["result"] = np.select(choicelist=choices, condlist=conditions, default="UNKNOWN")

    n_before = len(df)
    df = df[df["result"] != "UNKNOWN"].copy()
    print(f"Dropped {n_before - len(df)} rows with unresolvable result")

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
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

    essential_only_df = df[essential_columns]
    print(f"Shape before dropna: {essential_only_df.shape}")

    clean_df = essential_only_df.dropna()
    print(f"Shape after dropna: {clean_df.shape}")
    print(clean_df["result"].value_counts())
    print(f"These are the columns remaining: {essential_columns}")
    print(f"Data is now clean.")

    return clean_df


def parse_date(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    print(f'Datatype of "date": {df["date"].dtype}')

    return df


def add_numeric_round(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric matchday digits from the 'round' string column."""
    df["matchday"] = df["round"].str.extract(r"(\d+)")

    invalid_outputs = df["matchday"].isna().sum()
    print(f"Invalid_outputs: {invalid_outputs}")

    df = df.dropna(subset=["matchday"])
    df["matchday"] = df["matchday"].astype(int)

    print("Column 'matchday' has been successfully created!!!")

    return df


def save_cleandf(df: pd.DataFrame, filepath: Path) -> None:
    """Save clean DataFrame to disk, prompting user confirmation if file exists."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        print("Clean data already exists.")
        while True:
            decision = input("Overide the current file(Y/N):").strip().lower()
            if decision == "y":
                df.to_csv(filepath, index=False)
                print("Data has been overridden")
                break
            elif decision.lower() == "n":
                print("Keeping existing clean data.")
                break
            else:
                print("Invalid Input!!!")
    else:
        df.to_csv(filepath, index=False)
        print(f"Clean Data has been saved to {filepath}.")


def main():
    data_path = Path(__file__).resolve().parent.parent.parent / "data"
    raw_data_path = data_path / "raw" / "games.csv"
    clean_data_path = data_path / "processed" / "clean_data.csv"

    df = pd.read_csv(raw_data_path)
    df = filter_domestic_league(df)
    df = derive_result(df)
    df = parse_date(df)
    df = add_numeric_round(df)
    df = select_features(df)

    save_cleandf(df, clean_data_path)


if __name__ == "__main__":
    main()
