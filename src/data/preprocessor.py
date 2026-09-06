import pandas as pd
from pathlib import Path

def result_encoding(df):
    mapping = {"H":0, "D":1, "A":2}
    df["result"] = df["result"].map(mapping)
    df["result"] = df["result"].astype("Int32")

def odds_to_probability(df):
    df["homeW_probability"] = 1/df["home_odds"]
    df["draw_probability"] = 1/df["draw_odds"]
    df["awayW_probability"] = 1/df["away_odds"]

    df.drop(["home_odds", "draw_odds", "away_odds"], axis=1)

def main():
    data_path = Path(__file__).resolve().parent.parent.parent / "data"
    data = pd.read_csv(data_path / "combined_domestic.csv")

    data["match_date"] = pd.to_datetime(data["match_date"])
    data.sort_values(by="match_date", inplace=True)
    data.reset_index(drop=True, inplace=True)
    result_encoding(data)
    odds_to_probability(data)

    data.index.name = "match_id"
    data.to_csv(data_path / "preprocessed_domestic_data.csv", index=True)
    print("Data has been Preprocessed")
    print(data.info())


if __name__ == "__main__":
    main()