"""Feature engineering module for football match data.

Computes pre-game ELO ratings, constructs team-centric timeline logs, 
calculates rolling form and venue-specific metrics, resolves data leakage 
in league standings, and exports a final feature dataset for modeling.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def csv_reader(file_path: Path | str) -> pd.DataFrame:
    """Reads a CSV file into a pandas DataFrame and standardizes date formatting.

    Args:
        file_path (Path | str): Path to the target CSV file.

    Returns:
        pd.DataFrame: DataFrame sorted chronologically by match date with reset indices.
    """
    print("\nReading CSV data...")
    df = pd.read_csv(file_path)
    
    # Parse dates and enforce strict chronological sequence for temporal feature calculation
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    
    return df


def add_elo(
    df: pd.DataFrame,
    base_elo: float = 1500.0,
    k: float = 20.0,
    min_rating: float = 0.0,
    home_adv: float = 50.0,
) -> pd.DataFrame:
    """Calculates pre-game ELO ratings for home and away teams iteratively.

    Accounts for home venue advantage in expected win probabilities and dynamically 
    updates team ratings after each match outcome.

    Args:
        df (pd.DataFrame): Input DataFrame containing match records sorted by date.
        base_elo (float, optional): Initial ELO rating assigned to all teams. Defaults to 1500.0.
        k (float, optional): Weighting factor controlling maximum rating adjustment per match. Defaults to 20.0.
        min_rating (float, optional): Floor value for lowest permissible ELO rating. Defaults to 0.0.
        home_adv (float, optional): ELO point bonus granted to home teams during expected outcome evaluation. Defaults to 50.0.

    Returns:
        pd.DataFrame: DataFrame with pre-match ELO columns ('home_club_elo_pre', 
            'away_club_elo_pre') and ELO differential ('elo_diff').
    """
    home_ids = df["home_club_id"].to_numpy()
    away_ids = df["away_club_id"].to_numpy()

    # Numeric score mappings for home outcome
    result_map = {"H": 1.0, "D": 0.5, "A": 0.0}
    scores_home = df["result"].map(result_map).to_numpy()

    n_rows = len(df)
    unique_ids = np.unique(np.concatenate((home_ids, away_ids)))
    id_to_idx = {cid: idx for idx, cid in enumerate(unique_ids)}

    # Map team IDs to static array indices for fast vectorized ELO tracking
    home_indices = np.array([id_to_idx[cid] for cid in home_ids])
    away_indices = np.array([id_to_idx[cid] for cid in away_ids])

    ratings = np.full(len(unique_ids), float(base_elo))
    home_elo_pre = np.empty(n_rows, dtype=np.float64)
    away_elo_pre = np.empty(n_rows, dtype=np.float64)

    # Iteratively update ratings over time
    for i in range(n_rows):
        h_idx = home_indices[i]
        a_idx = away_indices[i]

        r_h = ratings[h_idx]
        r_a = ratings[a_idx]

        # Record ratings prior to current match resolution
        home_elo_pre[i] = r_h
        away_elo_pre[i] = r_a

        # Compute home expected score incorporating home advantage adjustment
        exp_h = 1.0 / (1.0 + 10.0 ** (((r_a) - (r_h + home_adv)) / 400.0))
        shift = k * (scores_home[i] - exp_h)

        # Update running ratings with floor boundaries
        ratings[h_idx] = max(min_rating, r_h + shift)
        ratings[a_idx] = max(min_rating, r_a - shift)

    # Append engineered ELO features
    df["home_club_elo_pre"] = home_elo_pre
    df["away_club_elo_pre"] = away_elo_pre
    df["elo_diff"] = (df["home_club_elo_pre"] + home_adv) - df["away_club_elo_pre"]

    print("\nELO calculation completed successfully.")
    return df


def build_team_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms match-level records into team-centric timeline logs.

    Doubles records into a team-centric perspective where each row captures 
    a team's specific performance, goals, match points, and rest days.

    Args:
        df (pd.DataFrame): Match-level DataFrame containing match scores and team IDs.

    Returns:
        pd.DataFrame: Long-format team match log sorted by team and match date.
    """
    cols = [
        "game_id",
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals",
    ]

    # Structure home team view
    home_timeline = df[cols].rename(
        columns={
            "home_club_id": "club_id",
            "away_club_id": "opponent_id",
            "home_club_goals": "goals_for",
            "away_club_goals": "goals_against",
        }
    )
    home_timeline["is_home"] = 1

    # Structure away team view
    away_timeline = df[cols].rename(
        columns={
            "away_club_id": "club_id",
            "home_club_id": "opponent_id",
            "away_club_goals": "goals_for",
            "home_club_goals": "goals_against",
        }
    )
    away_timeline["is_home"] = 0

    # Combine both viewpoints and sort chronologically per team
    timeline = pd.concat([home_timeline, away_timeline], ignore_index=True)
    timeline = timeline.sort_values(by=["club_id", "date"]).reset_index(drop=True)

    # Compute team performance indicators
    conditions_result = [
        timeline["goals_for"] > timeline["goals_against"],
        timeline["goals_for"] == timeline["goals_against"],
        timeline["goals_for"] < timeline["goals_against"],
    ]

    timeline["points_earned"] = np.select(conditions_result, [3, 1, 0], default=0)
    timeline["result"] = np.select(
        conditions_result, ["W", "D", "L"], default="UNKNOWN"
    )
    timeline["goals_diff"] = timeline["goals_for"] - timeline["goals_against"]
    timeline["is_draw"] = (timeline["result"] == "D").astype(int)

    # Compute rest duration (days between consecutive matches) per team
    timeline["rest_days"] = timeline.groupby("club_id")["date"].diff().dt.days

    return timeline


def add_rolling_form(team_log: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Calculates historical rolling window averages per club without data leakage.

    Computes overall and venue-specific rolling averages for team performance 
    metrics. Utilizes `.shift(1)` to strictly omit the current match outcome from 
    rolling calculations.

    Args:
        team_log (pd.DataFrame): Team-centric match log returned by `build_team_match_log`.
        window (int, optional): Historical game lookback window size. Defaults to 10.

    Returns:
        pd.DataFrame: Team log augmented with overall and venue-specific rolling form features.
    """
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

    # Compute shifted rolling statistics across general and venue contexts
    for col, new_col in metrics.items():
        # Overall historical form
        team_log[new_col] = grouped_overall[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

        # Venue-specific (Home vs Away) historical form
        venue_col = f"venue_{new_col}"
        team_log[venue_col] = grouped_venue[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

    return team_log


def merge_rolling_features(
    df: pd.DataFrame, team_log: pd.DataFrame, window: int = 10
) -> pd.DataFrame:
    """Maps general and venue-specific rolling team features back onto primary match records.

    Args:
        df (pd.DataFrame): Primary match-level DataFrame.
        team_log (pd.DataFrame): Processed team match log containing engineered rolling features.
        window (int, optional): Rolling lookback window size used during feature creation. Defaults to 10.

    Returns:
        pd.DataFrame: Primary match DataFrame augmented with home and away rolling form features.
    """
    print("\nMerging rolling features...")
    print(f"Window: {window}")
    
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
        "rest_days",
    ]

    merge_cols = ["game_id", "club_id"] + feature_cols

    # Prepare home and away feature projection subsets
    home_features = team_log[merge_cols].rename(
        columns={col: f"home_{col}" for col in feature_cols}
    )
    away_features = team_log[merge_cols].rename(
        columns={col: f"away_{col}" for col in feature_cols}
    )

    # Join home team rolling form
    df = df.merge(
        home_features,
        how="left",
        left_on=["game_id", "home_club_id"],
        right_on=["game_id", "club_id"],
    ).drop(columns=["club_id"])

    # Join away team rolling form
    df = df.merge(
        away_features,
        how="left",
        left_on=["game_id", "away_club_id"],
        right_on=["game_id", "club_id"],
    ).drop(columns=["club_id"])

    return df


def fix_position_leak(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Shifts league positions backward by one game to prevent current-match data leakage.

    Replaces post-match league positions with pre-match standings and drops initial 
    warm-up rows where rolling form/standings metrics cannot be calculated.

    Args:
        df (pd.DataFrame): Match DataFrame containing post-match club positions.
        window (int, optional): Lookback window size to check for missing feature completeness. Defaults to 10.

    Returns:
        pd.DataFrame: Leak-free DataFrame containing pre-match position rankings 
            ('home_club_position_pre', 'away_club_position_pre').
    """
    df = df.sort_values(["season", "competition_id", "date"]).reset_index(drop=True)

    base_cols = ["game_id", "season", "competition_id", "date"]

    home_df = df[base_cols + ["home_club_id", "home_club_position"]].rename(
        columns={"home_club_id": "club_id", "home_club_position": "post_game_pos"}
    )
    away_df = df[base_cols + ["away_club_id", "away_club_position"]].rename(
        columns={"away_club_id": "club_id", "away_club_position": "post_game_pos"}
    )

    # Aggregate timeline to shift position rankings chronologically
    timeline = pd.concat([home_df, away_df]).sort_values(
        ["season", "competition_id", "club_id", "date"]
    )

    # Shift position backward by 1 matchday to resolve leakage
    timeline["pre_game_pos"] = timeline.groupby(
        ["season", "competition_id", "club_id"]
    )["post_game_pos"].shift(1)

    # Re-map pre-game positions back to home and away teams
    home_pre = (
        timeline[["game_id", "club_id", "pre_game_pos"]]
        .drop_duplicates(subset=["game_id", "club_id"])
        .rename(
            columns={
                "club_id": "home_club_id",
                "pre_game_pos": "home_club_position_pre",
            }
        )
    )
    away_pre = (
        timeline[["game_id", "club_id", "pre_game_pos"]]
        .drop_duplicates(subset=["game_id", "club_id"])
        .rename(
            columns={
                "club_id": "away_club_id",
                "pre_game_pos": "away_club_position_pre",
            }
        )
    )

    df = df.merge(home_pre, on=["game_id", "home_club_id"], how="left")
    df = df.merge(away_pre, on=["game_id", "away_club_id"], how="left")

    # Drop original post-match leak columns
    df = df.drop(columns=["home_club_position", "away_club_position"], errors="ignore")

    # Filter out initial warm-up records missing complete rolling histories
    feature_subset = [
        f"home_avg_points_earned_l{window}",
        f"away_avg_points_earned_l{window}",
        f"home_venue_avg_points_earned_l{window}",
        f"away_venue_avg_points_earned_l{window}",
        "home_club_position_pre",
        "away_club_position_pre",
    ]

    df = df.dropna(subset=feature_subset).reset_index(drop=True)
    return df


def main() -> None:
    """Executes the core data engineering pipeline end-to-end."""
    window_size = 7
    processed_data_path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "processed"
    )

    clean_data_path = processed_data_path / "clean_data.csv"
    feature_engineered_data_path = processed_data_path / "featured.csv"

    # Sequential execution of feature engineering transformations
    clean_df = csv_reader(clean_data_path)
    clean_elo_df = add_elo(clean_df)
    timeline = build_team_match_log(clean_elo_df)
    timeline_with_form = add_rolling_form(timeline, window=window_size)
    merged_df = merge_rolling_features(clean_df, timeline_with_form, window=window_size)
    position_leak_fixed_df = fix_position_leak(merged_df, window=window_size)

    position_leak_fixed_df.to_csv(feature_engineered_data_path, index=False)
    print("\nData processing complete. Saved to featured.csv")


if __name__ == "__main__":
    main()