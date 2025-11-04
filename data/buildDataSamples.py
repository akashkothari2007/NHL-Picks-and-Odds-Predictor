import pandas as pd

seasons = ["20202021", "20212022", "20222023", "20232024", "20242025"]
teams = ["ANA", "UTA", "ARI","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL","DET","EDM",
    "FLA","LAK","MIN","MTL","NSH","NJD","NYI","NYR","OTT","PHI","PIT","SEA",
    "SJS","STL","TBL","TOR","VAN","VGK","WSH","WPG"]
team_map = {team: i for i, team in enumerate(teams, start=1)}

COLUMNS = [
    "home_code", "away_code", "home_avg_goals_for5", "away_avg_goals_for5", "home_avg_goals_against5", "away_avg_goals_against5",

    "home_avg_shots_for5", "away_avg_shots_for5",

    "home_avg_shots_against5", "away_avg_shots_against5",

    "home_avg_goal_diff5", "away_avg_goal_diff5",

    "home_win_pct5", "away_win_pct5",

    "home_regulation_wins5", "away_regulation_wins5", "home_pim5", "away_pim5", "home_ppg_for5", "away_ppg_for5", 
    "home_ppg_against5", "away_ppg_against5",

    "home_season_win_pct", "away_season_win_pct",
    "home_season_avg_goals_for", "away_season_avg_goals_for",
    "home_season_avg_goals_against", "away_season_avg_goals_against",
    "home_season_goal_diff", "away_season_goal_diff",
    "home_season_avg_save_pct", "away_season_avg_save_pct",
    "home_season_avg_shots_for", "away_season_avg_shots_for",
    "home_season_avg_shots_against", "away_season_avg_shots_against",
    "days_rest_home", "days_rest_away", "home_team_streak", "away_team_streak",
    "home_last_game_result", "away_last_game_result",
    "travel_distance_estimate", 

    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_avg_goal_diff",
    "h2h_avg_total_goals",
    "h2h_avg_shots_diff",
    "h2h_avg_save_pct_home",
    "h2h_avg_save_pct_away",

    "season_code",
    "game_month",
    "venue_altitude",
    "timezone_offset",
    "result"








]



df = pd.DataFrame(columns=COLUMNS)
df.to_csv("samples.csv", index=False)
samples = []
numOfSamples = 0
for season in seasons:
    
    fileName = f"sortedData/Sorted{season}GameData.csv"
    df = pd.read_csv(fileName)
    for idx, row in df.iterrows():
        home = team_map[row['home_team']]
        away = team_map[row['away_team']]
        season = str(row['game_id'])[:4]
        month = str(row['date'])[5:7] #get digits 6 and 7 somehow


        #last 5 calculations
        homeGameCount = 0
        awayGameCount = 0
        index = numOfSamples - 1
        while (index >= 0):
            #home team first
            isHome = False
            isAway = False
            if (samples[index]["home_code"] == home or samples[index]["away_code"] == home):
                #game found tally shit if game less than 5
                if (homeGameCount < 5):
                    homeGameCount += 1
                    isHome = True
                #tally other shit like season record (so check if season is current)
            if (samples[index]["home_code"] == home or samples[index]["away_code"] == home):
                #game found tally shit
                if (awayGameCount < 5):
                    awayGameCount += 1
                    isAway = True
            if (isHome and isAway):
                #head to head shit here

        samples.append({
            #data per row in here
        })
        numOfSamples += 1

        
    

df = pd.DataFrame(samples)
df.to_csv("samples.csv", mode="a", header=False, index=False)
                    
        
        
    