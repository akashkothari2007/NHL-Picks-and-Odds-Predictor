import pandas as pd
import numpy as np
import json
import time
import requests
import difflib
import os

from nba_api.stats.static import teams
from nba_api.stats.endpoints import cumestatsteamgames, cumestatsteam


def retry(func, retries=3, delay=2):
    def wrap(*args, **kwargs):
        for _ in range(retries):
            try:
                return func(*args, **kwargs)
            except:
                time.sleep(delay)
        return None
    return wrap


# ----------------------------------------------------------
# SCRAPE SEASON SCHEDULE
# ----------------------------------------------------------
def getSeasonScheduleFrame(season):
    teamLookup = pd.DataFrame(teams.get_teams())

    @retry
    def fetchSchedule(teamID):
        season_str = f"{season}-{str(season+1)[-2:]}"
        res = cumestatsteamgames.CumeStatsTeamGames(
            league_id='00',
            season=season_str,
            season_type_all_star="Regular Season",
            team_id=teamID
        ).get_normalized_json()

        if not res:
            return pd.DataFrame()

        df = pd.DataFrame(json.loads(res)['CumeStatsTeamGames'])
        df['SEASON'] = season_str
        return df

    schedule = pd.DataFrame()
    for tid in teamLookup['id']:
        df = fetchSchedule(tid)
        if df is not None and not df.empty:
            schedule = pd.concat([schedule, df], ignore_index=True)
        time.sleep(1.0)

    def getGameDate(m): return m.partition(' at')[0][:10]
    def getHomeTeam(m): return m.partition(' at')[2]
    def getAwayTeam(m): return m.partition(' at')[0][10:]

    def getID(nick):
        matches = difflib.get_close_matches(nick, teamLookup['nickname'], 1)
        if not matches:
            return np.nan
        return teamLookup.loc[teamLookup['nickname'] == matches[0], 'id'].values[0]

    schedule['GAME_DATE'] = pd.to_datetime(schedule['MATCHUP'].map(getGameDate))
    schedule['HOME_TEAM_NICKNAME'] = schedule['MATCHUP'].map(getHomeTeam)
    schedule['HOME_TEAM_ID'] = schedule['HOME_TEAM_NICKNAME'].map(getID)
    schedule['AWAY_TEAM_NICKNAME'] = schedule['MATCHUP'].map(getAwayTeam)
    schedule['AWAY_TEAM_ID'] = schedule['AWAY_TEAM_NICKNAME'].map(getID)

    # Clean IDs
    schedule['GAME_ID'] = schedule['GAME_ID'].astype(str).str.zfill(10)
    schedule.drop_duplicates(subset=['GAME_ID'], inplace=True)

    return schedule


# ----------------------------------------------------------
# SCRAPE METRICS FOR A SINGLE GAME
# ----------------------------------------------------------
def getSingleGameMetrics(gameID, homeID, awayID, awayNick, seasonStr, gameDate):

    @retry
    def getStats(teamID):
        res = cumestatsteam.CumeStatsTeam(
            game_ids=gameID,
            league_id="00",
            season=seasonStr,
            season_type_all_star="Regular Season",
            team_id=teamID
        ).get_normalized_json()

        if not res:
            return pd.DataFrame()

        return pd.DataFrame(json.loads(res)['TotalTeamStats'])

    try:
        data = getStats(homeID)
        if data is None or data.empty:
            return pd.DataFrame()

        # Fix second row for away team
        data.at[1, 'NICKNAME'] = awayNick
        data.at[1, 'TEAM_ID'] = awayID

        # Off efficiency & scoring margin
        data.at[1, 'OFFENSIVE_EFFICIENCY'] = (data.at[1, 'FG'] + data.at[1, 'AST']) / \
            (data.at[1, 'FGA'] - data.at[1, 'OFF_REB'] + data.at[1, 'AST'] + data.at[1, 'TOTAL_TURNOVERS'])
        data.at[1, 'SCORING_MARGIN'] = data.at[1, 'PTS'] - data.at[0, 'PTS']

        data.at[0, 'OFFENSIVE_EFFICIENCY'] = (data.at[0, 'FG'] + data.at[0, 'AST']) / \
            (data.at[0, 'FGA'] - data.at[0, 'OFF_REB'] + data.at[0, 'AST'] + data.at[0, 'TOTAL_TURNOVERS'])
        data.at[0, 'SCORING_MARGIN'] = data.at[0, 'PTS'] - data.at[1, 'PTS']

        # Meta info
        data['SEASON'] = seasonStr
        data['GAME_DATE'] = gameDate
        data['GAME_ID'] = gameID

        return data

    except:
        return pd.DataFrame()


# ----------------------------------------------------------
# FEATURE BUILDER
# ----------------------------------------------------------


# ----------------------------------------------------------
# MAIN UPDATE FUNCTION (SAFE VERSION)
# ----------------------------------------------------------
def updateAll(current_season=2025):
    from .features import getGameLogFeatureSet

    base = os.path.dirname(__file__)
    proc = os.path.join(base, "processedData")

    full_schedule_path = os.path.join(proc, "full_schedule.csv")
    gamelogs_path = os.path.join(proc, "gameLogs.csv")
    finalmodel_path = os.path.join(proc, "finalModelTraining.csv")

    # LOAD + STANDARDIZE
    full_schedule = pd.read_csv(full_schedule_path, dtype={"GAME_ID": str})
    existing_logs = pd.read_csv(gamelogs_path, dtype={"GAME_ID": str, "TEAM_ID": int})

    full_schedule["GAME_ID"] = full_schedule["GAME_ID"].astype(str).str.zfill(10)
    existing_logs["GAME_ID"] = existing_logs["GAME_ID"].astype(str).str.zfill(10)

    print("Scraping updated schedule...")
    new_schedule = getSeasonScheduleFrame(current_season)
    new_schedule["GAME_ID"] = new_schedule["GAME_ID"].astype(str).str.zfill(10)

    # Remove duplicates inside scrape
    new_schedule.drop_duplicates(subset=["GAME_ID"], inplace=True)

    # True new games
    new_schedule = new_schedule[~new_schedule["GAME_ID"].isin(full_schedule["GAME_ID"])]

    print(f"New games found: {len(new_schedule)}")
    if new_schedule.empty:
        print("No new games.")
        return

    # SAVE UPDATED SCHEDULE
    full_schedule = pd.concat([full_schedule, new_schedule], ignore_index=True)
    full_schedule.drop_duplicates(subset=["GAME_ID"], inplace=True)
    full_schedule.to_csv(full_schedule_path, index=False)

    # SCRAPE LOGS
    print("Scraping new game logs...")
    new_logs = pd.DataFrame()

    for _, row in new_schedule.iterrows():
        d = getSingleGameMetrics(
            row.GAME_ID,
            row.HOME_TEAM_ID,
            row.AWAY_TEAM_ID,
            row.AWAY_TEAM_NICKNAME,
            row.SEASON,
            row.GAME_DATE
        )
        if not d.empty:
            d["GAME_ID"] = d["GAME_ID"].astype(str).str.zfill(10)
            d["TEAM_ID"] = d["TEAM_ID"].astype(int)
            new_logs = pd.concat([new_logs, d], ignore_index=True)

        time.sleep(1.5)

    print(f"New logs scraped: {len(new_logs)}")

    # MERGE LOGS SAFELY
    gamelogs = pd.concat([existing_logs, new_logs], ignore_index=True)
    gamelogs.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], inplace=True)
    gamelogs.to_csv(gamelogs_path, index=False)

    # REBUILD FEATURE SET
    print("Rebuilding features...")
    final = getGameLogFeatureSet(gamelogs)
    final.to_csv(finalmodel_path, index=False)

    print("✔ Update complete — no duplicates, IDs fixed, feature set rebuilt.")


if __name__ == "__main__":
    updateAll(current_season=2025)