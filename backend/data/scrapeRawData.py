from nba_api.stats.static import teams
from nba_api.stats.endpoints import cumestatsteamgames, cumestatsteam
import pandas as pd, numpy as np, json, difflib, time, requests, os

# ---------- Retry decorator ----------
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


# ---------- Season schedule ----------
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


# ---------- Single-game metrics ----------
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


# ---------- Game log builder ----------
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


# ---------- Feature engineering ----------
def getGameLogFeatureSet(gameDF):
    gameDF.sort_values('GAME_DATE', inplace=True)
    gameDF['LAST_GAME_OE'] = gameDF.groupby(['TEAM_ID', 'SEASON'])['OFFENSIVE_EFFICIENCY'].shift(1)
    gameDF['LAST_GAME_ROLLING_OE'] = gameDF.groupby(['TEAM_ID', 'SEASON'])['OFFENSIVE_EFFICIENCY'].transform(lambda x: x.rolling(3, 1).mean())
    home = gameDF[gameDF['CITY'] != 'OPPONENTS']
    away = gameDF[gameDF['CITY'] == 'OPPONENTS']
    home.rename(lambda c: f"HOME_{c}" if c not in ['GAME_ID','SEASON'] else c, axis=1, inplace=True)
    away.rename(lambda c: f"AWAY_{c}" if c not in ['GAME_ID','SEASON'] else c, axis=1, inplace=True)
    merged = pd.merge(home, away, on=['GAME_ID','SEASON'])
    merged.drop_duplicates(subset=['GAME_ID','SEASON'], inplace=True)
    merged.to_csv('nbaHomeWinLossModelDataset.csv', index=False)
    print("Feature dataset saved.")
    return merged


# ---------- Main ----------
if __name__ == "__main__":
    # Load schedule CSV
    scheduleFrame = pd.read_csv("full_schedule.csv")

    # --- Fix data types & formatting ---
    # GAME_ID → string, re-add missing leading "00"
    scheduleFrame['GAME_ID'] = (
        scheduleFrame['GAME_ID']
        .astype(str)
        .str.replace(r'\.0$', '', regex=True)
        .apply(lambda x: x if x.startswith("00") else f"00{x}")
    )

    # TEAM IDs → int
    scheduleFrame['HOME_TEAM_ID'] = scheduleFrame['HOME_TEAM_ID'].astype(int)
    scheduleFrame['AWAY_TEAM_ID'] = scheduleFrame['AWAY_TEAM_ID'].astype(int)

    # SEASON → string
    scheduleFrame['SEASON'] = scheduleFrame['SEASON'].astype(str)

    # GAME_DATE → proper datetime
    scheduleFrame['GAME_DATE'] = pd.to_datetime(
        scheduleFrame['GAME_DATE'].astype(str).str.extract(r'(\d{4}-\d{2}-\d{2})')[0],
        errors='coerce'
    )


    gameLogs = getGameLogs(scheduleFrame)
    modelDataset = getGameLogFeatureSet(gameLogs)


    