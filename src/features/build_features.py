from pathlib import Path
import numpy as np
import pandas as pd


def csv_reader(file_path: Path) -> pd.DataFrame:
    """Reads a CSV file into a pandas DataFrame and standardizes date formatting."""
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_team_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms match-level records into team-centric timeline logs.

    Duplicates each match into separate home and away team perspectives, sorts
    chronologically per club, and calculates core match outcomes (points, diff,
    results). Tracks venue via the 'is_home' boolean flag.
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
    timeline = timeline.sort_values(by=["club_id", "date"]).reset_index(drop=True)

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
    timeline["goals_diff"] = timeline["goals_for"] - timeline["goals_against"]
    timeline["is_draw"] = (timeline["result"] == "D").astype(int)

    print(f"The shape of timeline is {timeline.shape}")
    print(f"The datatype of date column is {timeline['date'].dtype}")
    print(f"The datatype of points_earned column is {timeline['points_earned'].dtype}")

    return timeline


def add_rolling_form(team_log: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """Calculates historical rolling window averages per club without data leakage.

    Computes both overall rolling averages and venue-specific (home/away) rolling averages 
    by leveraging the 'is_home' flag. Shifts metrics by 1 period prior to rolling mean 
    computation to ensure match features reflect past form only.
    """
    if "date" in team_log.columns:
        team_log = team_log.sort_values(by=["club_id", "date"]).reset_index(drop=True)

    grouped_overall = team_log.groupby("club_id")
    grouped_venue = team_log.groupby(["club_id", "is_home"])

    metrics = {
        "goals_for": f"avg_goals_for_l{window}",
        "goals_against": f"avg_goals_against_l{window}",
        "points_earned": f"avg_points_earned_l{window}",
        "goals_diff": f"avg_goals_diff_l{window}",
        "is_draw": f"avg_is_draw_l{window}",
    }

    for col, new_col in metrics.items():
        team_log[new_col] = grouped_overall[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        
        venue_col = f"venue_{new_col}"
        team_log[venue_col] = grouped_venue[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    return team_log


def merge_rolling_features(
    df: pd.DataFrame, team_log: pd.DataFrame, window: int = 15
) -> pd.DataFrame:
    """Maps general and venue-specific rolling team features back onto primary match records.
    
    Binds the generated historical form metrics onto the original match DataFrame for both the 
    home side and away side. Venue-specific columns dynamically represent home form for the 
    home team and away form for the away team based on the match split.
    """
    len_before = len(df)
    print(f"Number of rows in df before merge: {len_before}")
    
    feature_cols = [
        f"avg_goals_for_l{window}",
        f"avg_goals_against_l{window}",
        f"avg_points_earned_l{window}",
        f"avg_goals_diff_l{window}",
        f"avg_is_draw_l{window}",
        f"venue_avg_goals_for_l{window}",
        f"venue_avg_goals_against_l{window}",
        f"venue_avg_points_earned_l{window}",
        f"venue_avg_goals_diff_l{window}",
        f"venue_avg_is_draw_l{window}",
    ]

    merge_cols = ["game_id", "club_id"] + feature_cols

    home_features = team_log[merge_cols].rename(
        columns={col: f"home_{col}" for col in feature_cols}
    )
    away_features = team_log[merge_cols].rename(
        columns={col: f"away_{col}" for col in feature_cols}
    )

    df = df.merge(
        home_features,
        how="left",
        left_on=["game_id", "home_club_id"],
        right_on=["game_id", "club_id"],
    ).drop(columns=["club_id"])

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
    print(df.dtypes)

    return df


def fix_position_leak(df: pd.DataFrame) -> pd.DataFrame:
    """Shifts league positions backward by one game to prevent current-match data leakage.
    
    A match record normally contains the team's standing after the match resolves. This 
    aligns the position sequence so the model evaluates pre-match standings instead.
    """
    df = df.sort_values(["season", "competition_id", "date"]).reset_index(drop=True)

    home_df = df[
        [
            "game_id",
            "season",
            "competition_id",
            "date",
            "home_club_id",
            "home_club_position",
        ]
    ].copy()
    home_df.rename(
        columns={"home_club_id": "club_id", "home_club_position": "post_game_pos"},
        inplace=True,
    )

    away_df = df[
        [
            "game_id",
            "season",
            "competition_id",
            "date",
            "away_club_id",
            "away_club_position",
        ]
    ].copy()
    away_df.rename(
        columns={"away_club_id": "club_id", "away_club_position": "post_game_pos"},
        inplace=True,
    )

    timeline = pd.concat([home_df, away_df]).sort_values(["season", "club_id", "date"])

    timeline["pre_game_pos"] = timeline.groupby(["season", "club_id"])[
        "post_game_pos"
    ].shift(1)

    home_pre = timeline[["game_id", "club_id", "pre_game_pos"]].rename(
        columns={"club_id": "home_club_id", "pre_game_pos": "home_club_position_pre"}
    )
    away_pre = timeline[["game_id", "club_id", "pre_game_pos"]].rename(
        columns={"club_id": "away_club_id", "pre_game_pos": "away_club_position_pre"}
    )

    df = df.merge(home_pre, on=["game_id", "home_club_id"], how="left")
    df = df.merge(away_pre, on=["game_id", "away_club_id"], how="left")

    df = df.drop(columns=["home_club_position", "away_club_position"])

    return df


def main():
    """Executes the core data engineering pipeline end-to-end."""
    processed_data_path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "processed"
    )

    clean_data_path = processed_data_path / "clean_data.csv"
    feature_engineered_data_path = processed_data_path / "featured.csv"

    clean_data = csv_reader(clean_data_path)
    timeline = build_team_match_log(clean_data)
    timeline_with_form = add_rolling_form(timeline, window=15)
    merged_df = merge_rolling_features(clean_data, timeline_with_form, window=15)
    final_df = fix_position_leak(merged_df)

    final_df.to_csv(feature_engineered_data_path, index=False)
    print("Data has been successfully processed and saved to featured.csv")


if __name__ == "__main__":
    main()