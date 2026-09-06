import os
import requests
import pandas as pd

# Define modern era window: 2012-13 onwards
# This avoids massive tactical shifts, pre-modern rule sets, and structural era bias 
# while maintaining a rich multi-year sample size (~14 seasons).
START_YEAR = 12
CURRENT_YEAR = 26  # Up to 2025/26 season

def generate_seasons(start, current):
    seasons = []
    for y in range(start, current):
        next_y = (y + 1) % 100
        seasons.append(f"{y:02d}{next_y:02d}")
    return seasons

SEASONS = generate_seasons(START_YEAR, CURRENT_YEAR)
os.makedirs('data/domestic', exist_ok=True)

print("--- 1. Downloading Domestic Leagues (football-data.co.uk) ---")
LEAGUES = {
    'Premier_League': 'E0',
    'La_Liga': 'SP1',
    'Bundesliga': 'D1',
    'Serie_A': 'I1',
    'Ligue_1': 'F1'
}

domestic_frames = []
for league_name, code in LEAGUES.items():
    for season in SEASONS:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                path = f"data/domestic/domestic_{league_name}_{season}.csv"
                with open(path, 'wb') as f:
                    f.write(res.content)
                df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
                df['DataSource'] = 'Domestic_League'
                df['Competition'] = league_name
                df['Season'] = season
                domestic_frames.append(df)
        except Exception as e:
            pass

print("\n--- 2. Downloading International / National Team Data ---")
# Using the standard open public international football results repository (1872-Present)
INT_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
try:
    os.makedirs('data/international', exist_ok=True)
    int_df = pd.read_csv(INT_URL)
    # Filter strictly by modern era (2012-01-01 onwards)
    int_df['date'] = pd.to_datetime(int_df['date'])
    int_modern = int_df[int_df['date'].dt.year >= 2012].copy()
    int_modern['DataSource'] = 'International'
    int_modern.to_csv("data/international/combined_international_matches.csv", index=False)
    print(f"International matches compiled (2012+): {len(int_modern)} total matches.")
except Exception as e:
    print(f"Failed to fetch international data: {e}")

print("\n--- 3. Downloading UEFA Champions League Data ---")
# Looping through European Champions League data repository by season folders (footballcsv)
ucl_frames = []
os.makedirs('data/international', exist_ok=True)
for y in range(START_YEAR + 2000, 2026):
    season_str = f"{y}-{str(y+1)[-2:]}"
    ucl_url = f"https://raw.githubusercontent.com/footballcsv/europe-champions-league/main/{season_str}/champs.csv"
    try:
        res = requests.get(ucl_url, timeout=5)
        if res.status_code == 200:
            path = f"data/ucl_{season_str}.csv"
            with open(path, 'wb') as f:
                f.write(res.content)
            df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
            df['DataSource'] = 'UCL'
            df['Season'] = season_str
            ucl_frames.append(df)
    except Exception:
        pass

if ucl_frames:
    master_ucl = pd.concat(ucl_frames, ignore_index=True)
    master_ucl.to_csv("data/master_ucl_matches.csv", index=False)
    print(f"UEFA Champions League compiled: {len(master_ucl)} total matches.")
else:
    print("UCL collection skipped or failed structure check.")