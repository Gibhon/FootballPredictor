import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce


def combine_domestic_data(data_path):

    domestic_path = data_path / "domestic"

    all_domestic_data_files = list(domestic_path.rglob("*.csv"))

    df_list = []
    for file in all_domestic_data_files:
        try:
            df = pd.read_csv(file)

            df_list.append(df)
        except Exception as e:
            print(f"Skipping corrupted or empty file {file.name}: {e}")

    common_columns = list(
        reduce(lambda x, y: x.intersection(y), [df.columns for df in df_list])
    )

    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        final_combined_df = combined_df[common_columns]
        print(f"Shape of combined domestic files is {combined_df.shape}")
        return final_combined_df
    else:
        print("No valid data files found.")


def get_FTR(df):
    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] == df["away_score"],
        df["home_score"] < df["away_score"],
    ]
    choices = ["H", "D", "A"]

    df["result"] = np.select(conditions, choices, default="U")


def main():
    data_path = Path(__file__).resolve().parent.parent.parent / "data"
    domestic = combine_domestic_data(data_path=data_path)

    # Drop Unnecessary Columns
    columns_drop_domestic = [
        "HTHG",
        "HTAG",
        "HTR",
        "HS",
        "AS",
        "HST",
        "AST",
        "HF",
        "AF",
        "HY",
        "AY",
        "HR",
        "AR",
        "B365H",
        "B365D",
        "B365A",
        "BWH",
        "BWD",
        "BWA",
        "PSH",
        "PSD",
        "PSA",
    ]
    domestic = domestic.drop(columns=columns_drop_domestic)

    # Rename To Concat
    rename_map_domestic = {
        "Div": "competition",
        "Date": "match_date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_score",
        "FTAG": "away_score",
        "HC": "home_corner",
        "AC": "away_corner",
        "FTR": "result",
        "PSCH": "home_odds",
        "PSCD": "draw_odds",
        "PSCA": "away_odds",
    }

    domestic = domestic.rename(columns=rename_map_domestic)

    # Concat
    new_order = [
        "match_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "competition",
        "home_odds",
        "away_odds",
        "draw_odds",
        "result",
    ]
    # domestic = domestic[new_order]
    domestic = domestic.dropna()
    print(domestic.columns)

    domestic.to_csv(
        str(
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "combined_domestic.csv"
        ),
        index=False,
    )
    print(f"The shape of COMBINED DATA is {domestic.shape}")
    print(domestic.info())


if __name__ == "__main__":
    main()
