from nba_api.stats.static import teams
from nba_api.stats.endpoints import cumestatsteamgames, cumestatsteam
import pandas as pd, numpy as np, json, difflib, time, requests, os

#retry func (nba api keeps fucking failing for some reason randomly)
def retry(func, retries=3, delay=5):
    def wrapper(*args, **kwargs):
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                print(f"Network error: {e} (attempt {attempt+1}/{retries})")
                time.sleep(delay)
        print("Failed after retries, skipping.")
        return pd.DataFrame()
    return wrapper


#get season schedule (put full thing in full_schedule.csv)
def getSeasonScheduleFrame(seasons, seasonType):
    def getGameDate(m): return m.partition(' at')[0][:10]
    def getHomeTeam(m): return m.partition(' at')[2]
    def getAwayTeam(m): return m.partition(' at')[0][10:]

    def getTeamIDFromNickname(nickname):
        try:
            match = difflib.get_close_matches(nickname, teamLookup['nickname'], 1)[0]
            return teamLookup.loc[teamLookup['nickname'] == match].values[0][0]
        except Exception:
            return np.nan

    @retry
    def getRegularSeasonSchedule(season, teamID, seasonType):
        season_str = f"{season}-{str(season+1)[-2:]}"
        try:
            res = cumestatsteamgames.CumeStatsTeamGames(
                league_id='00', season=season_str,
                season_type_all_star=seasonType, team_id=teamID
            ).get_normalized_json()
            if not res:
                print(f"Empty response {season_str} team {teamID}")
                return pd.DataFrame()
            df = pd.DataFrame(json.loads(res)['CumeStatsTeamGames'])
            df['SEASON'] = season_str
            return df
        except (json.JSONDecodeError, KeyError) as e:
            print(f"JSON error for {season_str} team {teamID}: {e}")
            return pd.DataFrame()

    teamLookup = pd.DataFrame(teams.get_teams())
    scheduleFrame = pd.DataFrame()

    for season in seasons:
        out_path = f"schedule_{season}.csv"
        if os.path.exists(out_path):
            print(f"Found {out_path}, skipping scrape.")
            part = pd.read_csv(out_path)
        else:
            part = pd.DataFrame()
            for tid in teamLookup['id']:
                df = getRegularSeasonSchedule(season, tid, seasonType)
                if not df.empty:
                    part = pd.concat([part, df], ignore_index=True)
                time.sleep(1.5)
            part.to_csv(out_path, index=False)
            print(f"Saved partial {out_path}")
        scheduleFrame = pd.concat([scheduleFrame, part], ignore_index=True)

    scheduleFrame['GAME_DATE'] = pd.to_datetime(scheduleFrame['MATCHUP'].map(getGameDate))
    scheduleFrame['HOME_TEAM_NICKNAME'] = scheduleFrame['MATCHUP'].map(getHomeTeam)
    scheduleFrame['HOME_TEAM_ID'] = scheduleFrame['HOME_TEAM_NICKNAME'].map(getTeamIDFromNickname)
    scheduleFrame['AWAY_TEAM_NICKNAME'] = scheduleFrame['MATCHUP'].map(getAwayTeam)
    scheduleFrame['AWAY_TEAM_ID'] = scheduleFrame['AWAY_TEAM_NICKNAME'].map(getTeamIDFromNickname)
    scheduleFrame.drop_duplicates(inplace=True)
    scheduleFrame.reset_index(drop=True, inplace=True)
    return scheduleFrame


#after full season for each game get metrics put in gamelogs
def getSingleGameMetrics(gameID, homeTeamID, awayTeamID, awayTeamNickname, seasonYear, gameDate):
    @retry
    def getGameStats(teamID, gameID, seasonYear):
        res = cumestatsteam.CumeStatsTeam(
            game_ids=gameID, league_id="00",
            season=seasonYear, season_type_all_star="Regular Season",
            team_id=teamID
        ).get_normalized_json()
        if not res:
            print(f"Empty game {gameID} {seasonYear}")
            return pd.DataFrame()
        return pd.DataFrame(json.loads(res)['TotalTeamStats'])

    try:
        data = getGameStats(homeTeamID, gameID, seasonYear)
        if data.empty: return pd.DataFrame()
        data.at[1, 'NICKNAME'] = awayTeamNickname
        data.at[1, 'TEAM_ID'] = awayTeamID
        data.at[1, 'OFFENSIVE_EFFICIENCY'] = (data.at[1, 'FG'] + data.at[1, 'AST']) / \
            (data.at[1, 'FGA'] - data.at[1, 'OFF_REB'] + data.at[1, 'AST'] + data.at[1, 'TOTAL_TURNOVERS'])
        data.at[1, 'SCORING_MARGIN'] = data.at[1, 'PTS'] - data.at[0, 'PTS']

        data.at[0, 'OFFENSIVE_EFFICIENCY'] = (data.at[0, 'FG'] + data.at[0, 'AST']) / \
            (data.at[0, 'FGA'] - data.at[0, 'OFF_REB'] + data.at[0, 'AST'] + data.at[0, 'TOTAL_TURNOVERS'])
        data.at[0, 'SCORING_MARGIN'] = data.at[0, 'PTS'] - data.at[1, 'PTS']

        data['SEASON'] = seasonYear
        data['GAME_DATE'] = gameDate
        data['GAME_ID'] = gameID
        return data
    except Exception as e:
        print(f"Error in getSingleGameMetrics {gameID}: {e}")
        return pd.DataFrame()


#runs that for every game in scheduleframe
def getGameLogs(scheduleFrame):
    gameLogs = pd.DataFrame()
    out_path = "gameLogs_progress.csv"
    if os.path.exists(out_path):
        gameLogs = pd.read_csv(out_path)
        print(f"Resuming from {len(gameLogs)} rows.")

    start_time = time.time()
    for i in range(len(gameLogs)//2, len(scheduleFrame)):
        row = scheduleFrame.iloc[i]
        g = getSingleGameMetrics(row.GAME_ID, row.HOME_TEAM_ID, row.AWAY_TEAM_ID,
                                 row.AWAY_TEAM_NICKNAME, row.SEASON, row.GAME_DATE)
        if not g.empty:
            gameLogs = pd.concat([gameLogs, g], ignore_index=True)
        if i % 200 == 0:
            gameLogs.to_csv(out_path, index=False)
            elapsed = (time.time() - start_time)/60
            print(f"Saved progress at {i}/{len(scheduleFrame)} games ({elapsed:.1f} min)")
        time.sleep(2)
    gameLogs.to_csv("gameLogs.csv", index=False)
    print("Finished all games.")
    return gameLogs


#uses game logs to make feature set for model (nbaHomeWinLossModelDataset.csv)
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
    
    df["HOME_WIN_PCTG"] = df["W_HOME"] / (df["W_HOME"] + df["L_HOME"]).replace(0, 1)
    df["AWAY_WIN_PCTG"] = df["W_ROAD"] / (df["W_ROAD"] + df["L_ROAD"]).replace(0, 1)
    
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
    # Home teams have CITY != "OPPONENTS", away teams have CITY == "OPPONENTS"
    home_games = df[df["CITY"] != "OPPONENTS"].copy()
    away_games = df[df["CITY"] == "OPPONENTS"].copy()
    
    # Merge to get opponent IDs
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
        
        # Get all previous games between these two teams (any season)
        prev_h2h = df[
            (df["GAME_DATE"] < game_date) &
            (
                ((df["TEAM_ID"] == home_id) & (df["CITY"] != "OPPONENTS")) |
                ((df["TEAM_ID"] == away_id) & (df["CITY"] == "OPPONENTS"))
            )
        ]
        
        # Filter to games where they played each other
        prev_home_games = prev_h2h[(prev_h2h["TEAM_ID"] == home_id) & (prev_h2h["CITY"] != "OPPONENTS")]["GAME_ID"]
        prev_away_games = prev_h2h[(prev_h2h["TEAM_ID"] == away_id) & (prev_h2h["CITY"] == "OPPONENTS")]["GAME_ID"]
        common_games = set(prev_home_games).intersection(set(prev_away_games))
        
        if len(common_games) >= 2:
            h2h_games = prev_h2h[prev_h2h["GAME_ID"].isin(common_games)].tail(10)  # Last 5 matchups = 10 rows
            
            # Home team's performance in these matchups
            home_h2h = h2h_games[(h2h_games["TEAM_ID"] == home_id) & (h2h_games["CITY"] != "OPPONENTS")]
            h2h_home_wins = home_h2h["W"].sum()
            h2h_home_win_pct = h2h_home_wins / (len(home_h2h) + 0.001)
            h2h_home_avg_margin = home_h2h["SCORING_MARGIN"].mean()
        else:
            # Not enough history - use neutral values
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
    
    # Fill any remaining NaNs from h2h merge
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
    
    # === TARGET ===
    merged["HOME_W"] = merged["HOME_W"]
    
    # Final NaN check
    print(f"\nDataset shape: {merged.shape}")
    print(f"NaN count per column:\n{merged.isna().sum()[merged.isna().sum() > 0]}")
    
    merged.fillna(0, inplace=True)
    
    merged.to_csv("finalModelTraining.csv", index=False)
    print("Saved finalModelTraining.csv")
    
    return merged


#main func (edited constantly cuz apis fail and progress is saved so just re-running at random points)
if __name__ == "__main__":
    gameLogs = pd.read_csv("processedData/gameLogs.csv")
    featureSet = getGameLogFeatureSet(gameLogs)


    