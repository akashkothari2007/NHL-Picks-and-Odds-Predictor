import pandas as pd

def getGameLogFeatureSet(gameDF):
    """
    Creates features from game logs with head-to-head matchup history.
    Returns a dataset ready for modeling with no NaNs.
    """
    
    df = gameDF.copy()
    
    # Ensure proper data types
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["TEAM_ID"] = df["TEAM_ID"].astype(int)
    df["GAME_ID"] = df["GAME_ID"].astype(int)
    
    # Sort by team and date
    df.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"], inplace=True)
    
    # === BASIC WIN PERCENTAGES ===
    df["TOTAL_GAMES_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"]).cumcount()
    df["WINS_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"])["W"].apply(lambda x: x.shift(1).fillna(0).cumsum()).reset_index(level=[0,1], drop=True)
    
    df["TOTAL_WIN_PCTG"] = df["WINS_PRIOR"] / df["TOTAL_GAMES_PRIOR"].replace(0, 1)
    df["TOTAL_WIN_PCTG"] = df["TOTAL_WIN_PCTG"].fillna(0.5)
    
    # === FIX: Cumulative home/away win percentages ===
    df["HOME_WINS_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"])["W_HOME"].apply(lambda x: x.shift(1).fillna(0).cumsum()).reset_index(level=[0,1], drop=True)
    df["HOME_LOSSES_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"])["L_HOME"].apply(lambda x: x.shift(1).fillna(0).cumsum()).reset_index(level=[0,1], drop=True)
    df["HOME_WIN_PCTG"] = df["HOME_WINS_PRIOR"] / (df["HOME_WINS_PRIOR"] + df["HOME_LOSSES_PRIOR"]).replace(0, 1)
    df["HOME_WIN_PCTG"] = df["HOME_WIN_PCTG"].fillna(0.5)
    
    df["AWAY_WINS_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"])["W_ROAD"].apply(lambda x: x.shift(1).fillna(0).cumsum()).reset_index(level=[0,1], drop=True)
    df["AWAY_LOSSES_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"])["L_ROAD"].apply(lambda x: x.shift(1).fillna(0).cumsum()).reset_index(level=[0,1], drop=True)
    df["AWAY_WIN_PCTG"] = df["AWAY_WINS_PRIOR"] / (df["AWAY_WINS_PRIOR"] + df["AWAY_LOSSES_PRIOR"]).replace(0, 1)
    df["AWAY_WIN_PCTG"] = df["AWAY_WIN_PCTG"].fillna(0.5)
    
    df["LAST_GAME_HOME_WIN_PCTG"] = df.groupby(["TEAM_ID", "SEASON"])["HOME_WIN_PCTG"].shift(1).fillna(0.5)
    df["LAST_GAME_AWAY_WIN_PCTG"] = df.groupby(["TEAM_ID", "SEASON"])["AWAY_WIN_PCTG"].shift(1).fillna(0.5)
    df["LAST_GAME_TOTAL_WIN_PCTG"] = df.groupby(["TEAM_ID", "SEASON"])["TOTAL_WIN_PCTG"].shift(1).fillna(0.5)
    
    # === REST DAYS ===
    df["PREV_GAME_DATE"] = df.groupby(["TEAM_ID", "SEASON"])["GAME_DATE"].shift(1)
    df["NUM_REST_DAYS"] = (df["GAME_DATE"] - df["PREV_GAME_DATE"]).dt.days
    df["NUM_REST_DAYS"] = df["NUM_REST_DAYS"].fillna(7).clip(upper=30)
    
    df["IS_BACK_TO_BACK"] = (df["NUM_REST_DAYS"] == 1).astype(int)
    
    # === ROLLING STATS (LAST 5 GAMES) ===
    df["ROLLING_OE"] = df.groupby(["TEAM_ID", "SEASON"])["OFFENSIVE_EFFICIENCY"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["LAST_GAME_ROLLING_OE"] = df.groupby(["TEAM_ID", "SEASON"])["ROLLING_OE"].shift(1).fillna(0.5)
    
    df["ROLLING_SCORING_MARGIN"] = df.groupby(["TEAM_ID", "SEASON"])["SCORING_MARGIN"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["LAST_GAME_ROLLING_SCORING_MARGIN"] = df.groupby(["TEAM_ID", "SEASON"])["ROLLING_SCORING_MARGIN"].shift(1).fillna(0)
    
    df["ROLLING_FG_PCT"] = df.groupby(["TEAM_ID", "SEASON"])["FG_PCT"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["LAST_GAME_ROLLING_FG_PCT"] = df.groupby(["TEAM_ID", "SEASON"])["ROLLING_FG_PCT"].shift(1).fillna(0.45)
    
    # === RECENT FORM (LAST 3 WINS) ===
    df["LAST_3_WINS"] = df.groupby(["TEAM_ID", "SEASON"])["W"].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    df["LAST_GAME_LAST_3_WINS"] = df.groupby(["TEAM_ID", "SEASON"])["LAST_3_WINS"].shift(1).fillna(1)
    
    # === IDENTIFY OPPONENT FOR EACH GAME ===
    home_games = df[df["CITY"] != "OPPONENTS"].copy()
    away_games = df[df["CITY"] == "OPPONENTS"].copy()
    
    game_matchups = pd.merge(
        home_games[["GAME_ID", "TEAM_ID", "GAME_DATE", "SEASON"]],
        away_games[["GAME_ID", "TEAM_ID"]],
        on="GAME_ID",
        suffixes=("_home", "_away")
    )
    
    # === HEAD-TO-HEAD HISTORY ===
    print("Calculating head-to-head features...")
    h2h_features = []
    
    for _, matchup in game_matchups.iterrows():
        game_id = matchup["GAME_ID"]
        home_id = matchup["TEAM_ID_home"]
        away_id = matchup["TEAM_ID_away"]
        game_date = matchup["GAME_DATE"]
        season = matchup["SEASON"]
        
        prev_h2h = df[
            (df["GAME_DATE"] < game_date) &
            (
                ((df["TEAM_ID"] == home_id) & (df["CITY"] != "OPPONENTS")) |
                ((df["TEAM_ID"] == away_id) & (df["CITY"] == "OPPONENTS"))
            )
        ]
        
        prev_home_games = prev_h2h[(prev_h2h["TEAM_ID"] == home_id) & (prev_h2h["CITY"] != "OPPONENTS")]["GAME_ID"]
        prev_away_games = prev_h2h[(prev_h2h["TEAM_ID"] == away_id) & (prev_h2h["CITY"] == "OPPONENTS")]["GAME_ID"]
        common_games = set(prev_home_games).intersection(set(prev_away_games))
        
        if len(common_games) >= 2:
            h2h_games = prev_h2h[prev_h2h["GAME_ID"].isin(common_games)].tail(10)
            home_h2h = h2h_games[(h2h_games["TEAM_ID"] == home_id) & (h2h_games["CITY"] != "OPPONENTS")]
            h2h_home_wins = home_h2h["W"].sum()
            h2h_home_win_pct = h2h_home_wins / (len(home_h2h) + 0.001)
            h2h_home_avg_margin = home_h2h["SCORING_MARGIN"].mean()
        else:
            h2h_home_win_pct = 0.5
            h2h_home_avg_margin = 0.0
        
        h2h_features.append({
            "GAME_ID": game_id,
            "H2H_HOME_WIN_PCT": h2h_home_win_pct,
            "H2H_HOME_AVG_MARGIN": h2h_home_avg_margin
        })
    
    h2h_df = pd.DataFrame(h2h_features)
    
    # === SPLIT HOME AND AWAY, MERGE ===
    home = df[df["CITY"] != "OPPONENTS"].copy()
    away = df[df["CITY"] == "OPPONENTS"].copy()
    
    keep_cols_home = [
        "TEAM_ID", "GAME_ID", "SEASON", "W",
        "LAST_GAME_HOME_WIN_PCTG", "LAST_GAME_AWAY_WIN_PCTG", "LAST_GAME_TOTAL_WIN_PCTG",
        "NUM_REST_DAYS", "IS_BACK_TO_BACK",
        "LAST_GAME_ROLLING_OE", "LAST_GAME_ROLLING_SCORING_MARGIN",
        "LAST_GAME_ROLLING_FG_PCT", "LAST_GAME_LAST_3_WINS"
    ]
    
    keep_cols_away = [
        "TEAM_ID", "GAME_ID", "SEASON",
        "LAST_GAME_HOME_WIN_PCTG", "LAST_GAME_AWAY_WIN_PCTG", "LAST_GAME_TOTAL_WIN_PCTG",
        "NUM_REST_DAYS", "IS_BACK_TO_BACK",
        "LAST_GAME_ROLLING_OE", "LAST_GAME_ROLLING_SCORING_MARGIN",
        "LAST_GAME_ROLLING_FG_PCT", "LAST_GAME_LAST_3_WINS"
    ]
    
    home = home[keep_cols_home].rename(columns={c: "HOME_" + c for c in keep_cols_home if c not in ["GAME_ID", "SEASON"]})
    away = away[keep_cols_away].rename(columns={c: "AWAY_" + c for c in keep_cols_away if c not in ["GAME_ID", "SEASON"]})
    
    merged = pd.merge(home, away, on=["GAME_ID", "SEASON"], how="inner")
    merged = pd.merge(merged, h2h_df, on="GAME_ID", how="left")
    
    merged["H2H_HOME_WIN_PCT"] = merged["H2H_HOME_WIN_PCT"].fillna(0.5)
    merged["H2H_HOME_AVG_MARGIN"] = merged["H2H_HOME_AVG_MARGIN"].fillna(0.0)
    
    # === CREATE DIFFERENTIAL FEATURES ===
    merged["WIN_PCTG_DIFF"] = merged["HOME_LAST_GAME_TOTAL_WIN_PCTG"] - merged["AWAY_LAST_GAME_TOTAL_WIN_PCTG"]
    merged["OE_DIFF"] = merged["HOME_LAST_GAME_ROLLING_OE"] - merged["AWAY_LAST_GAME_ROLLING_OE"]
    merged["SCORING_MARGIN_DIFF"] = merged["HOME_LAST_GAME_ROLLING_SCORING_MARGIN"] - merged["AWAY_LAST_GAME_ROLLING_SCORING_MARGIN"]
    merged["REST_DIFF"] = merged["HOME_NUM_REST_DAYS"] - merged["AWAY_NUM_REST_DAYS"]
    merged["HOME_ADVANTAGE"] = merged["HOME_LAST_GAME_HOME_WIN_PCTG"] - merged["HOME_LAST_GAME_AWAY_WIN_PCTG"]
    merged["FG_PCT_DIFF"] = merged["HOME_LAST_GAME_ROLLING_FG_PCT"] - merged["AWAY_LAST_GAME_ROLLING_FG_PCT"]
    merged["FORM_DIFF"] = merged["HOME_LAST_GAME_LAST_3_WINS"] - merged["AWAY_LAST_GAME_LAST_3_WINS"]
    
    merged["HOME_W"] = merged["HOME_W"]
    
    print(f"\nDataset shape: {merged.shape}")
    print(f"NaN count per column:\n{merged.isna().sum()[merged.isna().sum() > 0]}")
    
    merged.fillna(0, inplace=True)
    
    merged.to_csv("finalModelTraining.csv", index=False)
    print("Saved finalModelTraining.csv")
    
    return merged


def getSingleGameFeatureSet(home_team_ids, away_team_ids, game_date=None, current_season="2025-26"):
    import pandas as pd
    from datetime import datetime
    import os

    if isinstance(home_team_ids, int):
        home_team_ids = [home_team_ids]
        away_team_ids = [away_team_ids]

    if game_date is None:
        game_date = datetime.now()
    else:
        game_date = pd.to_datetime(game_date)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_path = os.path.join(base_dir, "processedData", "gameLogs.csv")

    past = pd.read_csv(logs_path)
    past["GAME_DATE"] = pd.to_datetime(past["GAME_DATE"], format="mixed")
    past["TEAM_ID"] = past["TEAM_ID"].astype(int)

    past = past[past["GAME_DATE"] < game_date]

    all_rows = []
    for home, away in zip(home_team_ids, away_team_ids):
        row = calculate_single_game_features(home, away, game_date, past, current_season)
        all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # Ensure ordering EXACTLY matches training columns
    ordered_cols = [
        'HOME_LAST_GAME_HOME_WIN_PCTG', 'HOME_LAST_GAME_AWAY_WIN_PCTG', 
        'HOME_LAST_GAME_TOTAL_WIN_PCTG', 'HOME_NUM_REST_DAYS', 'HOME_IS_BACK_TO_BACK',
        'HOME_LAST_GAME_ROLLING_OE', 'HOME_LAST_GAME_ROLLING_SCORING_MARGIN',
        'HOME_LAST_GAME_ROLLING_FG_PCT', 'HOME_LAST_GAME_LAST_3_WINS',
        
        'AWAY_LAST_GAME_HOME_WIN_PCTG', 'AWAY_LAST_GAME_AWAY_WIN_PCTG',
        'AWAY_LAST_GAME_TOTAL_WIN_PCTG', 'AWAY_NUM_REST_DAYS', 'AWAY_IS_BACK_TO_BACK',
        'AWAY_LAST_GAME_ROLLING_OE', 'AWAY_LAST_GAME_ROLLING_SCORING_MARGIN',
        'AWAY_LAST_GAME_ROLLING_FG_PCT', 'AWAY_LAST_GAME_LAST_3_WINS',

        'H2H_HOME_WIN_PCT', 'H2H_HOME_AVG_MARGIN',
        'WIN_PCTG_DIFF', 'OE_DIFF', 'SCORING_MARGIN_DIFF', 'REST_DIFF',
        'HOME_ADVANTAGE', 'FG_PCT_DIFF', 'FORM_DIFF'
    ]

    df = df[ordered_cols]

    return df


def calculate_single_game_features(home_id, away_id, game_date, past_games, current_season):
    import pandas as pd

    # Filter only current season
    season_games = past_games[past_games["SEASON"] == current_season].copy()

    # ----------------------------
    # Helper to compute training-style rolling features
    # ----------------------------
    def compute_team_features(team_id):
        team_games = season_games[season_games["TEAM_ID"] == team_id].sort_values("GAME_DATE")

        # If not enough games → neutral
        if len(team_games) == 0:
            return None

        # LAST GAME VALUES
        last = team_games.iloc[-1]
        last_date = pd.to_datetime(last["GAME_DATE"])
        rest_days = (game_date - last_date).days
        is_b2b = 1 if rest_days == 1 else 0

        # TOTAL WIN PCT (TRAINING STYLE)
        total_wins = team_games["W"].sum()
        total_losses = team_games["L"].sum()
        total_win_pct = total_wins / max(total_wins + total_losses, 1)

        # HOME/AWAY WIN %
        home_wins = team_games["W_HOME"].sum()
        home_losses = team_games["L_HOME"].sum()
        home_win_pct = home_wins / max(home_wins + home_losses, 1)

        away_wins = team_games["W_ROAD"].sum()
        away_losses = team_games["L_ROAD"].sum()
        away_win_pct = away_wins / max(away_wins + away_losses, 1)

        # ROLLING (LAST 5 GAMES) – EXACT MATCH TO TRAINING
        recent = team_games.tail(5)
        rolling_oe = recent["OFFENSIVE_EFFICIENCY"].mean()
        rolling_margin = recent["SCORING_MARGIN"].mean()
        rolling_fg = recent["FG_PCT"].mean()
        last_3_wins = recent["W"].tail(3).sum()

        return {
            "LAST_GAME_HOME_WIN_PCTG": home_win_pct,
            "LAST_GAME_AWAY_WIN_PCTG": away_win_pct,
            "LAST_GAME_TOTAL_WIN_PCTG": total_win_pct,
            "NUM_REST_DAYS": rest_days,
            "IS_BACK_TO_BACK": is_b2b,
            "LAST_GAME_ROLLING_OE": rolling_oe,
            "LAST_GAME_ROLLING_SCORING_MARGIN": rolling_margin,
            "LAST_GAME_ROLLING_FG_PCT": rolling_fg,
            "LAST_GAME_LAST_3_WINS": last_3_wins
        }

    # Compute home + away features
    home = compute_team_features(home_id)
    away = compute_team_features(away_id)

    if home is None or away is None:
        return get_neutral_features()

    # ----------------------------
    # HEAD TO HEAD (TRAINING STYLE)
    # ----------------------------
    h2h = past_games[
        (past_games["TEAM_ID"] == home_id) |
        (past_games["TEAM_ID"] == away_id)
    ]

    home_ids = set(h2h[h2h["TEAM_ID"] == home_id]["GAME_ID"])
    away_ids = set(h2h[h2h["TEAM_ID"] == away_id]["GAME_ID"])
    common_games = home_ids.intersection(away_ids)

    if len(common_games) >= 2:
        last_h2h = h2h[h2h["GAME_ID"].isin(common_games)].sort_values("GAME_DATE").tail(10)
        home_h2h = last_h2h[last_h2h["TEAM_ID"] == home_id]

        h2h_win_pct = home_h2h["W"].sum() / len(home_h2h)
        h2h_margin = home_h2h["SCORING_MARGIN"].mean()
    else:
        h2h_win_pct = 0.5
        h2h_margin = 0.0

    # ----------------------------
    # BUILD FINAL MATCHING FEATURE SET
    # ----------------------------
    features = {
        "HOME_LAST_GAME_HOME_WIN_PCTG": home["LAST_GAME_HOME_WIN_PCTG"],
        "HOME_LAST_GAME_AWAY_WIN_PCTG": home["LAST_GAME_AWAY_WIN_PCTG"],
        "HOME_LAST_GAME_TOTAL_WIN_PCTG": home["LAST_GAME_TOTAL_WIN_PCTG"],
        "HOME_NUM_REST_DAYS": home["NUM_REST_DAYS"],
        "HOME_IS_BACK_TO_BACK": home["IS_BACK_TO_BACK"],
        "HOME_LAST_GAME_ROLLING_OE": home["LAST_GAME_ROLLING_OE"],
        "HOME_LAST_GAME_ROLLING_SCORING_MARGIN": home["LAST_GAME_ROLLING_SCORING_MARGIN"],
        "HOME_LAST_GAME_ROLLING_FG_PCT": home["LAST_GAME_ROLLING_FG_PCT"],
        "HOME_LAST_GAME_LAST_3_WINS": home["LAST_GAME_LAST_3_WINS"],

        "AWAY_LAST_GAME_HOME_WIN_PCTG": away["LAST_GAME_HOME_WIN_PCTG"],
        "AWAY_LAST_GAME_AWAY_WIN_PCTG": away["LAST_GAME_AWAY_WIN_PCTG"],
        "AWAY_LAST_GAME_TOTAL_WIN_PCTG": away["LAST_GAME_TOTAL_WIN_PCTG"],
        "AWAY_NUM_REST_DAYS": away["NUM_REST_DAYS"],
        "AWAY_IS_BACK_TO_BACK": away["IS_BACK_TO_BACK"],
        "AWAY_LAST_GAME_ROLLING_OE": away["LAST_GAME_ROLLING_OE"],
        "AWAY_LAST_GAME_ROLLING_SCORING_MARGIN": away["LAST_GAME_ROLLING_SCORING_MARGIN"],
        "AWAY_LAST_GAME_ROLLING_FG_PCT": away["LAST_GAME_ROLLING_FG_PCT"],
        "AWAY_LAST_GAME_LAST_3_WINS": away["LAST_GAME_LAST_3_WINS"],

        "H2H_HOME_WIN_PCT": h2h_win_pct,
        "H2H_HOME_AVG_MARGIN": h2h_margin,

        # DIFF FEATURES (match training naming)
        "WIN_PCTG_DIFF": home["LAST_GAME_TOTAL_WIN_PCTG"] - away["LAST_GAME_TOTAL_WIN_PCTG"],
        "OE_DIFF": home["LAST_GAME_ROLLING_OE"] - away["LAST_GAME_ROLLING_OE"],
        "SCORING_MARGIN_DIFF": home["LAST_GAME_ROLLING_SCORING_MARGIN"] - away["LAST_GAME_ROLLING_SCORING_MARGIN"],
        "REST_DIFF": home["NUM_REST_DAYS"] - away["NUM_REST_DAYS"],
        "HOME_ADVANTAGE": home["LAST_GAME_HOME_WIN_PCTG"] - home["LAST_GAME_AWAY_WIN_PCTG"],
        "FG_PCT_DIFF": home["LAST_GAME_ROLLING_FG_PCT"] - away["LAST_GAME_ROLLING_FG_PCT"],
        "FORM_DIFF": home["LAST_GAME_LAST_3_WINS"] - away["LAST_GAME_LAST_3_WINS"],
    }

    return features


def get_neutral_features():
    """Returns neutral/default features when not enough data"""
    return {
        'HOME_LAST_GAME_HOME_WIN_PCTG': 0.5,
        'HOME_LAST_GAME_AWAY_WIN_PCTG': 0.5,
        'HOME_LAST_GAME_TOTAL_WIN_PCTG': 0.5,
        'HOME_NUM_REST_DAYS': 2,
        'HOME_IS_BACK_TO_BACK': 0,
        'HOME_LAST_GAME_ROLLING_OE': 0.5,
        'HOME_LAST_GAME_ROLLING_SCORING_MARGIN': 0,
        'HOME_LAST_GAME_ROLLING_FG_PCT': 0.45,
        'HOME_LAST_GAME_LAST_3_WINS': 1.5,
        
        'AWAY_LAST_GAME_HOME_WIN_PCTG': 0.5,
        'AWAY_LAST_GAME_AWAY_WIN_PCTG': 0.5,
        'AWAY_LAST_GAME_TOTAL_WIN_PCTG': 0.5,
        'AWAY_NUM_REST_DAYS': 2,
        'AWAY_IS_BACK_TO_BACK': 0,
        'AWAY_LAST_GAME_ROLLING_OE': 0.5,
        'AWAY_LAST_GAME_ROLLING_SCORING_MARGIN': 0,
        'AWAY_LAST_GAME_ROLLING_FG_PCT': 0.45,
        'AWAY_LAST_GAME_LAST_3_WINS': 1.5,
        
        'H2H_HOME_WIN_PCT': 0.5,
        'H2H_HOME_AVG_MARGIN': 0,
        
        'WIN_PCTG_DIFF': 0,
        'OE_DIFF': 0,
        'SCORING_MARGIN_DIFF': 0,
        'REST_DIFF': 0,
        'HOME_ADVANTAGE': 0,
        'FG_PCT_DIFF': 0,
        'FORM_DIFF': 0,
    }