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

    df = gameDF.copy()
    df.sort_values(["TEAM_ID", "SEASON", "GAME_DATE"], inplace=True)

    # win pct
    df["TOTAL_GAMES_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"]).cumcount()
    df["WINS_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"])["W"].shift(1).fillna(0).cumsum()
    df["LOSSES_PRIOR"] = df.groupby(["TEAM_ID", "SEASON"])["L"].shift(1).fillna(0).cumsum()

    df["TOTAL_WIN_PCTG"] = df["WINS_PRIOR"] / df["TOTAL_GAMES_PRIOR"].replace(0, np.nan)

    # Home/Away win pct
    df["HOME_WIN_PCTG"] = df["W_HOME"] / (df["W_HOME"] + df["L_HOME"]).replace(0, np.nan)
    df["AWAY_WIN_PCTG"] = df["W_ROAD"] / (df["W_ROAD"] + df["L_ROAD"]).replace(0, np.nan)

    df["LAST_GAME_HOME_WIN_PCTG"] = df.groupby(["TEAM_ID", "SEASON"])["HOME_WIN_PCTG"].shift(1)
    df["LAST_GAME_AWAY_WIN_PCTG"] = df.groupby(["TEAM_ID", "SEASON"])["AWAY_WIN_PCTG"].shift(1)
    df["LAST_GAME_TOTAL_WIN_PCTG"] = df.groupby(["TEAM_ID", "SEASON"])["TOTAL_WIN_PCTG"].shift(1)

    # rest days
    df["PREV_GAME_DATE"] = df.groupby(["TEAM_ID", "SEASON"])["GAME_DATE"].shift(1)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["PREV_GAME_DATE"] = pd.to_datetime(df["PREV_GAME_DATE"], errors="coerce")
    df["NUM_REST_DAYS"] = (df["GAME_DATE"] - df["PREV_GAME_DATE"]).dt.days
    df["NUM_REST_DAYS"] = df["NUM_REST_DAYS"].fillna(7) 

    # rolling features
    df["LAST_GAME_OE"] = df.groupby(["TEAM_ID", "SEASON"])["OFFENSIVE_EFFICIENCY"].shift(1)
    df["ROLLING_OE"] = df.groupby(["TEAM_ID", "SEASON"])["OFFENSIVE_EFFICIENCY"].transform(lambda x: x.rolling(3, 1).mean())
    df["LAST_GAME_ROLLING_OE"] = df.groupby(["TEAM_ID", "SEASON"])["ROLLING_OE"].shift(1)

    df["ROLLING_SCORING_MARGIN"] = df.groupby(["TEAM_ID", "SEASON"])["SCORING_MARGIN"].transform(lambda x: x.rolling(3, 1).mean())
    df["LAST_GAME_ROLLING_SCORING_MARGIN"] = df.groupby(["TEAM_ID", "SEASON"])["ROLLING_SCORING_MARGIN"].shift(1)

    # Extra non-leaking features
    df["LAST_GAME_FG_PCT"] = df.groupby(["TEAM_ID", "SEASON"])["FG_PCT"].shift(1)
    df["LAST_GAME_TOT_REB"] = df.groupby(["TEAM_ID", "SEASON"])["TOT_REB"].shift(1)
    df["LAST_GAME_TURNOVERS"] = df.groupby(["TEAM_ID", "SEASON"])["TOTAL_TURNOVERS"].shift(1)

    # split home and away and merge
    home = df[df["CITY"] != "OPPONENTS"]
    away = df[df["CITY"] == "OPPONENTS"]

    keep_cols_home = [
        "TEAM_ID","GAME_ID","SEASON","W",
        "LAST_GAME_OE","LAST_GAME_HOME_WIN_PCTG","NUM_REST_DAYS",
        "LAST_GAME_AWAY_WIN_PCTG","LAST_GAME_TOTAL_WIN_PCTG",
        "LAST_GAME_ROLLING_SCORING_MARGIN","LAST_GAME_ROLLING_OE",
        "LAST_GAME_FG_PCT","LAST_GAME_TOT_REB","LAST_GAME_TURNOVERS"
    ]

    keep_cols_away = [
        "TEAM_ID","GAME_ID","SEASON",
        "LAST_GAME_OE","LAST_GAME_HOME_WIN_PCTG","NUM_REST_DAYS",
        "LAST_GAME_AWAY_WIN_PCTG","LAST_GAME_TOTAL_WIN_PCTG",
        "LAST_GAME_ROLLING_SCORING_MARGIN","LAST_GAME_ROLLING_OE",
        "LAST_GAME_FG_PCT","LAST_GAME_TOT_REB","LAST_GAME_TURNOVERS"
    ]

    home = home[keep_cols_home].rename(columns={c: "HOME_"+c for c in keep_cols_home if c not in ["GAME_ID","SEASON"]})
    away = away[keep_cols_away].rename(columns={c: "AWAY_"+c for c in keep_cols_away if c not in ["GAME_ID","SEASON"]})

    merged = pd.merge(home, away, on=["GAME_ID","SEASON"])

    # target
    merged["HOME_W"] = merged["HOME_W"]

    merged.to_csv("finalModelTraining.csv", index=False)
    print("Saved finalModelTraining.csv")
    return merged


#main func (edited constantly cuz apis fail and progress is saved so just re-running at random points)
if __name__ == "__main__":
    gameLogs = pd.read_csv("gameLogs.csv")
    featureSet = getGameLogFeatureSet(gameLogs)


    