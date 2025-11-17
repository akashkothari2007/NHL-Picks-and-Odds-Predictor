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
    df["WINS_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"])["W"].shift(1).fillna(0).cumsum()
    
    df["TOTAL_WIN_PCTG"] = df["WINS_PRIOR"] / df["TOTAL_GAMES_PRIOR"].replace(0, 1)
    df["TOTAL_WIN_PCTG"] = df["TOTAL_WIN_PCTG"].fillna(0.5)
    
    # === FIX: Cumulative home/away win percentages ===
    df["HOME_WINS_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"])["W_HOME"].shift(1).fillna(0).cumsum()
    df["HOME_LOSSES_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"])["L_HOME"].shift(1).fillna(0).cumsum()
    df["HOME_WIN_PCTG"] = df["HOME_WINS_PRIOR"] / (df["HOME_WINS_PRIOR"] + df["HOME_LOSSES_PRIOR"]).replace(0, 1)
    df["HOME_WIN_PCTG"] = df["HOME_WIN_PCTG"].fillna(0.5)
    
    df["AWAY_WINS_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"])["W_ROAD"].shift(1).fillna(0).cumsum()
    df["AWAY_LOSSES_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"])["L_ROAD"].shift(1).fillna(0).cumsum()
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


def getSingleGameFeatureSet(home_team_ids, away_team_ids, game_date=None, current_season="2024-25"):
    """
    Calculate features for multiple upcoming games.
    
    Args:
        home_team_ids: List of home team IDs (or single int)
        away_team_ids: List of away team IDs (or single int)
        game_date: Date for prediction (defaults to today)
        current_season: Current NBA season (e.g., "2024-25")
    
    Returns:
        DataFrame with features for each game (ready for model.predict())
    """
    import pandas as pd
    import numpy as np
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
    gamelogs_path = os.path.join(base_dir, "processedData", "gameLogs.csv")
    
    print(f"Loading gameLogs from {gamelogs_path}...")
    gameLogs = pd.read_csv(gamelogs_path)
    gameLogs['GAME_DATE'] = pd.to_datetime(gameLogs['GAME_DATE'], format='mixed')
    gameLogs['TEAM_ID'] = gameLogs['TEAM_ID'].astype(int)
    
    past_games = gameLogs[gameLogs['GAME_DATE'] < game_date].copy()
    
    print(f"Calculating features for {len(home_team_ids)} games (Season: {current_season})...")
    
    all_features = []
    
    for home_id, away_id in zip(home_team_ids, away_team_ids):
        features = calculate_single_game_features(home_id, away_id, game_date, past_games, current_season)
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    feature_columns = [
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
    
    features_df = features_df[feature_columns]
    
    print(f"✓ Features calculated for {len(features_df)} games")
    
    return features_df


def calculate_single_game_features(home_id, away_id, game_date, past_games, current_season):
    import pandas as pd
    import numpy as np
    
    # === FIX: Filter to current season FIRST ===
    season_games = past_games[past_games['SEASON'] == current_season].copy()
    
    # === HOME TEAM FEATURES ===
    home_games = season_games[season_games['TEAM_ID'] == home_id].sort_values('GAME_DATE')
    
    if len(home_games) < 5:
        print(f"⚠️  Warning: Only {len(home_games)} games for home team {home_id} in {current_season}")
        return get_neutral_features()
    
    home_recent = home_games.tail(5)
    
    home_rolling_oe = home_recent['OFFENSIVE_EFFICIENCY'].mean()
    home_rolling_margin = home_recent['SCORING_MARGIN'].mean()
    home_rolling_fg = home_recent['FG_PCT'].mean()
    home_last_3_wins = home_recent['W'].tail(3).sum()
    
    # Sum all games in current season
    home_total_wins = home_games['W'].sum()
    home_total_losses = home_games['L'].sum()
    home_total_games = home_total_wins + home_total_losses
    home_total_win_pct = home_total_wins / max(home_total_games, 1)
    
    home_home_wins = home_games['W_HOME'].sum()
    home_home_losses = home_games['L_HOME'].sum()
    home_home_win_pct = home_home_wins / max(home_home_wins + home_home_losses, 1)
    
    home_away_wins = home_games['W_ROAD'].sum()
    home_away_losses = home_games['L_ROAD'].sum()
    home_away_win_pct = home_away_wins / max(home_away_wins + home_away_losses, 1)
    
    home_last = home_games.iloc[-1]
    home_last_game_date = pd.to_datetime(home_last['GAME_DATE'])
    home_rest_days = (game_date - home_last_game_date).days
    home_is_b2b = 1 if home_rest_days == 1 else 0
    


    # === AWAY TEAM FEATURES ===
    away_games = season_games[season_games['TEAM_ID'] == away_id].sort_values('GAME_DATE')
    
    if len(away_games) < 5:
        print(f"⚠️  Warning: Only {len(away_games)} games for away team {away_id} in {current_season}")
        return get_neutral_features()
    
    away_recent = away_games.tail(5)
    
    away_rolling_oe = away_recent['OFFENSIVE_EFFICIENCY'].mean()
    away_rolling_margin = away_recent['SCORING_MARGIN'].mean()
    away_rolling_fg = away_recent['FG_PCT'].mean()
    away_last_3_wins = away_recent['W'].tail(3).sum()
    
    away_total_wins = away_games['W'].sum()
    away_total_losses = away_games['L'].sum()
    away_total_games = away_total_wins + away_total_losses
    away_total_win_pct = away_total_wins / max(away_total_games, 1)
    
    away_home_wins = away_games['W_HOME'].sum()
    away_home_losses = away_games['L_HOME'].sum()
    away_home_win_pct = away_home_wins / max(away_home_wins + away_home_losses, 1)
    
    away_away_wins = away_games['W_ROAD'].sum()
    away_away_losses = away_games['L_ROAD'].sum()
    away_away_win_pct = away_away_wins / max(away_away_wins + away_away_losses, 1)
    
    away_last = away_games.iloc[-1]
    away_last_game_date = pd.to_datetime(away_last['GAME_DATE'])
    away_rest_days = (game_date - away_last_game_date).days
    away_is_b2b = 1 if away_rest_days == 1 else 0


    
    # === HEAD-TO-HEAD (uses all historical data, not just current season) ===
    h2h_games = past_games[
        (past_games['TEAM_ID'] == home_id) | (past_games['TEAM_ID'] == away_id)
    ]
    
    home_h2h_game_ids = set(h2h_games[h2h_games['TEAM_ID'] == home_id]['GAME_ID'])
    away_h2h_game_ids = set(h2h_games[h2h_games['TEAM_ID'] == away_id]['GAME_ID'])
    common_game_ids = home_h2h_game_ids.intersection(away_h2h_game_ids)
    
    if len(common_game_ids) >= 2:
        h2h = h2h_games[h2h_games['GAME_ID'].isin(common_game_ids)].sort_values('GAME_DATE').tail(10)
        home_h2h = h2h[h2h['TEAM_ID'] == home_id]
        
        if len(home_h2h) > 0:
            h2h_home_win_pct = home_h2h['W'].sum() / len(home_h2h)
            h2h_home_avg_margin = home_h2h['SCORING_MARGIN'].mean()
        else:
            h2h_home_win_pct = 0.5
            h2h_home_avg_margin = 0.0
    else:
        h2h_home_win_pct = 0.5
        h2h_home_avg_margin = 0.0
    
    # === BUILD FEATURES DICT ===
    features = {
        'HOME_LAST_GAME_HOME_WIN_PCTG': home_home_win_pct,
        'HOME_LAST_GAME_AWAY_WIN_PCTG': home_away_win_pct,
        'HOME_LAST_GAME_TOTAL_WIN_PCTG': home_total_win_pct,
        'HOME_NUM_REST_DAYS': home_rest_days,
        'HOME_IS_BACK_TO_BACK': home_is_b2b,
        'HOME_LAST_GAME_ROLLING_OE': home_rolling_oe,
        'HOME_LAST_GAME_ROLLING_SCORING_MARGIN': home_rolling_margin,
        'HOME_LAST_GAME_ROLLING_FG_PCT': home_rolling_fg,
        'HOME_LAST_GAME_LAST_3_WINS': home_last_3_wins,
        
        'AWAY_LAST_GAME_HOME_WIN_PCTG': away_home_win_pct,
        'AWAY_LAST_GAME_AWAY_WIN_PCTG': away_away_win_pct,
        'AWAY_LAST_GAME_TOTAL_WIN_PCTG': away_total_win_pct,
        'AWAY_NUM_REST_DAYS': away_rest_days,
        'AWAY_IS_BACK_TO_BACK': away_is_b2b,
        'AWAY_LAST_GAME_ROLLING_OE': away_rolling_oe,
        'AWAY_LAST_GAME_ROLLING_SCORING_MARGIN': away_rolling_margin,
        'AWAY_LAST_GAME_ROLLING_FG_PCT': away_rolling_fg,
        'AWAY_LAST_GAME_LAST_3_WINS': away_last_3_wins,
        
        'H2H_HOME_WIN_PCT': h2h_home_win_pct,
        'H2H_HOME_AVG_MARGIN': h2h_home_avg_margin,
        
        'WIN_PCTG_DIFF': home_total_win_pct - away_total_win_pct,
        'OE_DIFF': home_rolling_oe - away_rolling_oe,
        'SCORING_MARGIN_DIFF': home_rolling_margin - away_rolling_margin,
        'REST_DIFF': home_rest_days - away_rest_days,
        'HOME_ADVANTAGE': home_home_win_pct - home_away_win_pct,
        'FG_PCT_DIFF': home_rolling_fg - away_rolling_fg,
        'FORM_DIFF': home_last_3_wins - away_last_3_wins,
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