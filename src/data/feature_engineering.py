import pandas as pd
import numpy as np

COMPETITON_WEIGHTS = {
    "E0": 30,
    "SP1": 30,
    "D1": 30,
    "F1": 30,
    "I1": 30,
    "FIFA World Cup": 75,
    "UEFA Euro": 60,
    "UEFA Nations League": 45,
    "Copa América": 50,
    "UEFA Euro qualification": 25,
    "Friendly": 15,
}


def k_calculator(competition, home_score, away_score):
    base_k = COMPETITON_WEIGHTS.get(competition, 15)
    margin = abs(int(away_score) - int(home_score))
    if margin == 1:
        multiplier = 1
    elif margin == 0:
        multiplier = 0.5
    elif margin == 2:
        multiplier = 2
    else:
        multiplier = 1.75 + (margin - 3) / 8

    return base_k * multiplier


def process_elo(df, initial_elo=1500.0, min_elo=-10000,home_adv=0):
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    current_elo = {}

    home_elos_pre = np.zeros(len(df))
    away_elos_pre = np.zeros(len(df))

    for i in range(len(df)):
        home = df.loc[i, "home_team"]
        away = df.loc[i, "away_team"]

        r_home = current_elo.get(home, initial_elo)
        r_away = current_elo.get(away, initial_elo)

        home_elos_pre[i] = r_home
        away_elos_pre[i] = r_away

        g_home = df.loc[i, "home_score"]
        g_away = df.loc[i, "away_score"]
        res = df.loc[i, "result"]

        comp = df.loc[i, "competition"] if "competition" in df.columns else "Friendly"

        if res == 0:
            w_home, w_away = 1.0, 0.0
        elif res == 1:
            w_home, w_away = 0.5, 0.5
        elif res == 2:
            w_home, w_away = 0.0, 1.0
        else:
            continue
        e_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        e_away = 1 - e_home

        k = k_calculator(competition=comp, home_score=g_home, away_score=g_away)

        new_r_home = r_home + k * (w_home - e_home)
        new_r_away = r_away + k * (w_away - e_away)

        current_elo[home] = max(min_elo, new_r_home)
        current_elo[away] = max(min_elo, new_r_away)

    df["home_elo"] = home_elos_pre
    df["away_elo"] = away_elos_pre
    df["elo_diff"] = home_elos_pre - away_elos_pre + home_adv

    return df


def create_draw_signals(df):
    pass


def timeline_features(df):
    df["match_date"] = pd.to_datetime(df["match_date"])

    home_rename = {
        "home_team": "team",
        "home_score": "goals_scored",
        "away_score": "goals_conceeded",
        "home_corner": "corners_taken",
        "away_corner": "consers_conceeded",
    }
    away_rename = {
        "away_team": "team",
        "home_score": "goals_conceeded",
        "away_score": "goals_scored",
        "away_corner": "corners_taken",
        "home_corner": "corners_conceeded",
    }

    home_matches = df[
        [
            "match_date",
            "home_team",
            "home_score",
            "away_score",
            "home_corner",
            "away_corner",
            "result",
        ]
    ].rename(columns=home_rename)
    home_matches["points"] = home_matches["result"].map({0: 3, 1: 1, 2: 0})

    away_matches = df[
        [
            "match_date",
            "away_team",
            "home_score",
            "away_score",
            "home_corner",
            "away_corner",
            "result",
        ]
    ].rename(columns=away_rename)
    away_matches["points"] = away_matches["result"].map({2: 3, 1: 1, 0: 0})

    timeline = (
        pd.concat([home_matches, away_matches])
        .sort_values(by=["team", "match_date"])
        .reset_index(drop=True)
    )

    timeline["match_date"] = pd.to_datetime(timeline["match_date"])

    timeline["goals_scored_l15"] = timeline.groupby("team")["goals_scored"].transform(
        lambda x: x.shift(1).rolling(window=15, min_periods=1).mean()
    )

    timeline["goals_conceeded_l15"] = timeline.groupby("team")[
        "goals_conceeded"
    ].transform(lambda x: x.shift(1).rolling(window=15, min_periods=1).mean())

    timeline["form_points"] = timeline.groupby("team")["points"].transform(
        lambda x: x.shift(1).rolling(window=15, min_periods=1).mean()
    )

    timeline["corners_taken_l15"] = timeline.groupby("team")["corners_taken"].transform(
        lambda x: x.shift(1).rolling(window=15, min_periods=1).mean()
    )

    timeline["corners_conceeded_l15"] = timeline.groupby("team")[
        "corners_conceeded"
    ].transform(lambda x: x.shift(1).rolling(window=15, min_periods=1).mean())

    timeline["rest_days"] = (
        timeline.groupby("team")["match_date"]
        .transform(lambda x: x.diff().dt.days)
        .fillna(7)
        .clip(upper=7)
    )

    df = (
        df.merge(
            timeline[
                [
                    "match_date",
                    "team",
                    "goals_scored_l15",
                    "goals_conceeded_l15",
                    "corners_taken_l15",
                    "corners_conceeded_l15",
                    "rest_days",
                    "form_points",
                ]
            ],
            left_on=["match_date", "home_team"],
            right_on=["match_date", "team"],
            how="left",
        )
        .rename(
            columns={
                "goals_scored_l15": "home_goals_scored_l15",
                "goals_conceeded_l15": "home_goals_conceeded_l15",
                "corners_taken_l15": "home_corners_taken_l15",
                "corners_conceeded_l15": "home_corners_conceeded_l15",
                "rest_days": "home_rest_days",
                "form_points": "home_form_points",
            }
        )
        .drop(columns=["team"])
    )

    df = (
        df.merge(
            timeline[
                [
                    "match_date",
                    "team",
                    "goals_scored_l15",
                    "goals_conceeded_l15",
                    "corners_taken_l15",
                    "corners_conceeded_l15",
                    "rest_days",
                    "form_points",
                ]
            ],
            left_on=["match_date", "away_team"],
            right_on=["match_date", "team"],
            how="left",
        )
        .rename(
            columns={
                "goals_scored_l15": "away_goals_scored_l15",
                "goals_conceeded_l15": "away_goals_conceeded_l15",
                "corners_taken_l15": "away_corners_taken_l15",
                "corners_conceeded_l15": "away_corners_conceeded_l15",
                "rest_days": "away_rest_days",
                "form_points": "away_form_points",
            }
        )
        .drop(columns=["team"])
    )
    df = df.dropna()

    return df


def save_df(df):
    df = df.drop(
        [
            "match_id",
            "match_date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "competition",
            "home_corner",
            "away_corner",
            "home_odds",
            "draw_odds",
            "away_odds",
        ],
        axis=1,
    )
    df.to_csv(
        r"C:\Users\Sponge\Documents\Code\python\projects\FootballPredictor\data\ready_domestic_D.csv",
        index=False,
    )
    print(df.info())
    print("We ready")


def main():
    df = pd.read_csv(
        r"C:\Users\Sponge\Documents\Code\python\projects\FootballPredictor\data\preprocessed_domestic_data.csv"
    )

    df = process_elo(df)
    # df = create_draw_signals(df)
    df = timeline_features(df)
    save_df(df)


if __name__ == "__main__":
    main()
