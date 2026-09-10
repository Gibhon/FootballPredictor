from pathlib import Path
import numpy as np
import pandas as pd


def csv_reader(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_team_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms match-level records into team-centric timeline logs.

    Duplicates each match into separate home and away team perspectives, sorts
    chronologically per club, and calculates core match outcomes (points, diff,
    results).
    """
    home_mapping = {
        "home_club_id": "club_id",
        "away_club_id": "opponent_id",
        "home_club_goals": "goals_for",
        "away_club_goals": "goals_against",
    }
    away_mapping = {
        "away_club_id": "club_id",
        "home_club_id": "opponent_id",
        "away_club_goals": "goals_for",
        "home_club_goals": "goals_against",
    }

    home_timeline = df[
        [
            "game_id",
            "date",
            "home_club_id",
            "away_club_id",
            "home_club_goals",
            "away_club_goals",
        ]
    ].rename(columns=home_mapping)
    home_timeline["is_home"] = 1

    away_timeline = df[
        [
            "game_id",
            "date",
            "home_club_id",
            "away_club_id",
            "home_club_goals",
            "away_club_goals",
        ]
    ].rename(columns=away_mapping)
    away_timeline["is_home"] = 0

    timeline = pd.concat([home_timeline, away_timeline], ignore_index=True)
    timeline = timeline.sort_values(by=["club_id", "date"]).reset_index(
        drop=True
    )

    conditions_result = [
        timeline["goals_for"] > timeline["goals_against"],
        timeline["goals_for"] == timeline["goals_against"],
        timeline["goals_for"] < timeline["goals_against"],
    ]
    choices_points = [3, 1, 0]

    timeline["points_earned"] = np.select(
        choicelist=choices_points, condlist=conditions_result
    )
    timeline["result"] = np.select(
        conditions_result, ["W", "D", "L"], default="UNKNOWN"
    )
    timeline["goals_diff"] = (
        timeline["goals_for"] - timeline["goals_against"]
    )

    print(f"The shape of timeline is {timeline.shape}")
    print(f"The datatype of date column is {timeline['date'].dtype}")
    print(
        f"The datatype of points_earned column is {timeline['points_earned'].dtype}"
    )

    return timeline


def add_rolling_form(team_log: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """Calculates historical rolling window averages per club without data leakage.

    Shifts metrics by 1 period prior to rolling mean computation to ensure match
    features reflect past form only.
    """
    if "date" in team_log.columns:
        team_log = team_log.sort_values(by=["club_id", "date"]).reset_index(
            drop=True
        )

    grouped = team_log.groupby("club_id")

    metrics = {
        "goals_for": f"avg_goals_for_l{window}",
        "goals_against": f"avg_goals_against_l{window}",
        "points_earned": f"avg_points_earned_l{window}",
    }

    for col, new_col in metrics.items():
        team_log[new_col] = grouped[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
    return team_log


def merge_rolling_features(
    df: pd.DataFrame, team_log: pd.DataFrame, window: int = 15
) -> pd.DataFrame:
    """Maps rolling team features back onto primary match records for home and away sides."""
    len_before = len(df)
    print(f"Number of rows in df before merge: {len_before}")
    # 1. Define source metrics from team_log and target feature names
    feature_cols = [
        f"avg_goals_for_l{window}",
        f"avg_goals_against_l{window}",
        f"avg_points_earned_l{window}",
    ]

    # 2. Prepare sub-DataFrames for home and away merges
    merge_cols = ["game_id", "club_id"] + feature_cols

    home_features = team_log[merge_cols].rename(
        columns={col: f"home_{col}" for col in feature_cols}
    )
    away_features = team_log[merge_cols].rename(
        columns={col: f"away_{col}" for col in feature_cols}
    )

    # 3. Merge Home Features
    df = df.merge(
        home_features,
        how="left",
        left_on=["game_id", "home_club_id"],
        right_on=["game_id", "club_id"],
    ).drop(columns=["club_id"])

    # 4. Merge Away Features
    df = df.merge(
        away_features,
        how="left",
        left_on=["game_id", "away_club_id"],
        right_on=["game_id", "club_id"],
    ).drop(columns=["club_id"])

    df.dropna(inplace=True)
    len_after = len(df)
    print(f"Number of rows in df after dropping NaNs: {len_after}")
    print(f"Number of rows dropped due to NaNs: {len_before - len_after}")
    print(f"Number of unique clubs in team_log: {team_log['club_id'].nunique()}")
    # print(df.head(50))
    # Prints every column name and its value vertically for row 0
    print(df.iloc[0].to_string())
    print(team_log[team_log.club_id == 11126].head(3))
    print(df.dtypes)

    return df


def main():
    processed_data_path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "processed"
    )

    clean_data_path = processed_data_path / "clean_data.csv"
    feature_engineered_data_path = processed_data_path / "featured.csv"

    clean_data = csv_reader(clean_data_path)
    timeline = build_team_match_log(clean_data)
    timeline_with_form = add_rolling_form(timeline, window=15)
    final_df = merge_rolling_features(clean_data, timeline_with_form, window=15)

    final_df.to_csv(feature_engineered_data_path, index=False)
    print("Data has been successfully processed and saved to featured.csv")


if __name__ == "__main__":
    main()