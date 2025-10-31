import time
import requests
import pandas as pd

##Season and team name
seasons = ["20242025", "20232024", "20222023", "20212022", "20202021"]
TEAMS = ["ANA", "UTA", "ARI","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL","DET","EDM",
    "FLA","LAK","MIN","MTL","NSH","NJD","NYI","NYR","OTT","PHI","PIT","SEA",
    "SJS","STL","TBL","TOR","VAN","VGK","WSH","WPG"]

##set up csv with columns
COLUMNS = [
    "game_id", "date", "season", "venue", "venue_location",

    "home_team", "away_team",

    "home_score", "away_score",

    "home_sog", "away_sog",

    "periods", "winner_team",

    "home_goalie_even_shots", "home_goalie_even_goals", "home_goalie_short_shots", "home_goalie_short_goals", "home_goalie_power_shots", "home_goalie_power_goals", 
    "away_goalie_even_shots", "away_goalie_even_goals", "away_goalie_short_shots", "away_goalie_short_goals", "away_goalie_power_shots", "away_goalie_power_goals", 
    "home_save_pctg", "away_save_pctg",

    "home_total_pim", "away_total_pim",

    "start_time_UTC", "game_state"
]

for SEASON in seasons:
    try:
        FILENAME = "%sGameData"%SEASON
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(FILENAME, index=False)

        #seen ids so no duplicate games
        seen_ids = []
        #set up rows
        rows = []
        #one team at a time, access their games
        for team in TEAMS:
            try:
                url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{SEASON}"
                response = requests.get(url)
                if response.status_code != 200:
                    print("Failed:", url)
                    continue

                data = response.json()
                #one game at a time, get id to check boxscore which contains all the data
                for game in data['games']:
                    try:
                        game_id = game['id']
                        if game_id in seen_ids:
                            continue
                        seen_ids.append(game_id)
                        game_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
                        response = requests.get(game_url)
                        if response.status_code != 200:
                            print("Failed:", game_url)
                            continue
                        box_score = response.json()

                        #data calculations
                        away_pim = 0
                        home_pim = 0
                        home_even_strength_shots_against = 0
                        home_even_strength_goals_allowed = 0
                        home_power_play_shots_against = 0
                        home_power_play_goals_allowed = 0
                        home_short_handed_shots_against = 0
                        home_short_handed_goals_allowed = 0
                        away_even_strength_shots_against = 0
                        away_even_strength_goals_allowed = 0
                        away_power_play_shots_against = 0
                        away_power_play_goals_allowed = 0
                        away_short_handed_shots_against = 0
                        away_short_handed_goals_allowed = 0
                    
                        #sum pim for all players
                        for forward in box_score['playerByGameStats']['homeTeam']['forwards']:
                            home_pim += forward['pim']
                        for forward in box_score['playerByGameStats']['awayTeam']['forwards']:
                            away_pim += forward['pim']
                        for defense in box_score['playerByGameStats']['homeTeam']['defense']:
                            home_pim += defense['pim']
                        for defense in box_score['playerByGameStats']['awayTeam']['defense']:
                            away_pim += defense['pim']

                        # Home goalies
                        for goalie in box_score['playerByGameStats']['homeTeam']['goalies']:
                            evenShots = goalie['evenStrengthShotsAgainst'].split("/")
                            powerShots = goalie['powerPlayShotsAgainst'].split("/")
                            shortShots = goalie['shorthandedShotsAgainst'].split("/")

                            # Handle 0/0 case by checking total saves/shots
                            total_saves = 0
                            total_shots = 0
                            if goalie.get("saveShotsAgainst"):
                                try:
                                    made, total = [float(x) for x in goalie["saveShotsAgainst"].split("/")]
                                    total_saves = made
                                    total_shots = total
                                except:
                                    pass

                            e_total = float(evenShots[1])
                            p_total = float(powerShots[1])
                            s_total = float(shortShots[1])

                            # If all breakdowns are 0 but total is non-zero, use that
                            if (e_total + p_total + s_total) == 0 and total_shots > 0:
                                home_even_strength_shots_against += total_shots
                                home_even_strength_goals_allowed += total_shots - total_saves
                            else:
                                home_even_strength_shots_against += e_total
                                home_even_strength_goals_allowed += float(evenShots[1]) - float(evenShots[0])
                                home_power_play_shots_against += float(powerShots[1])
                                home_power_play_goals_allowed += float(powerShots[1]) - float(powerShots[0])
                                home_short_handed_shots_against += float(shortShots[1])
                                home_short_handed_goals_allowed += float(shortShots[1]) - float(shortShots[0])

                        # Away goalies
                        for goalie in box_score['playerByGameStats']['awayTeam']['goalies']:
                            evenShots = goalie['evenStrengthShotsAgainst'].split("/")
                            powerShots = goalie['powerPlayShotsAgainst'].split("/")
                            shortShots = goalie['shorthandedShotsAgainst'].split("/")

                            total_saves = 0
                            total_shots = 0
                            if goalie.get("saveShotsAgainst"):
                                try:
                                    made, total = [float(x) for x in goalie["saveShotsAgainst"].split("/")]
                                    total_saves = made
                                    total_shots = total
                                except:
                                    pass

                            e_total = float(evenShots[1])
                            p_total = float(powerShots[1])
                            s_total = float(shortShots[1])

                            if (e_total + p_total + s_total) == 0 and total_shots > 0:
                                away_even_strength_shots_against += total_shots
                                away_even_strength_goals_allowed += total_shots - total_saves
                            else:
                                away_even_strength_shots_against += e_total
                                away_even_strength_goals_allowed += float(evenShots[1]) - float(evenShots[0])
                                away_power_play_shots_against += float(powerShots[1])
                                away_power_play_goals_allowed += float(powerShots[1]) - float(powerShots[0])
                                away_short_handed_shots_against += float(shortShots[1])
                                away_short_handed_goals_allowed += float(shortShots[1]) - float(shortShots[0])

                        # Safe save pctg calculator
                        def safe_pctg(shots, goals):
                            return 0 if shots == 0 else round((shots - goals) / shots, 3)

                        home_total_shots = home_even_strength_shots_against + home_power_play_shots_against + home_short_handed_shots_against
                        home_total_goals = home_even_strength_goals_allowed + home_power_play_goals_allowed + home_short_handed_goals_allowed
                        away_total_shots = away_even_strength_shots_against + away_power_play_shots_against + away_short_handed_shots_against
                        away_total_goals = away_even_strength_goals_allowed + away_power_play_goals_allowed + away_short_handed_goals_allowed

                        home_save_pctg = safe_pctg(home_total_shots, home_total_goals)
                        away_save_pctg = safe_pctg(away_total_shots, away_total_goals)

                        winner = "home"
                        if (box_score['homeTeam']['score'] < box_score['awayTeam']['score']):
                            winner = "away"
                        ## add to rows
                        rows.append({
                            "game_id": game_id,
                            "date": box_score['gameDate'],
                            "season": box_score['season'],
                            "venue": box_score['venue']['default'],
                            "venue_location": box_score['venueLocation']['default'],
                            "home_team": box_score['homeTeam']['abbrev'],
                            "away_team": box_score['awayTeam']['abbrev'],
                            "home_score": box_score['homeTeam']['score'],
                            "away_score": box_score['awayTeam']['score'],
                            "home_sog": box_score['homeTeam']['sog'],
                            "away_sog": box_score['awayTeam']['sog'],
                            "periods": box_score['regPeriods'],
                            "winner_team": winner,
                            "home_goalie_even_shots": home_even_strength_shots_against,
                            "home_goalie_even_goals": home_even_strength_goals_allowed,
                            "home_goalie_short_shots": home_short_handed_shots_against,
                            "home_goalie_short_goals": home_short_handed_goals_allowed,
                            "home_goalie_power_shots": home_power_play_shots_against,
                            "home_goalie_power_goals": home_power_play_goals_allowed,
                            "away_goalie_even_shots": away_even_strength_shots_against,
                            "away_goalie_even_goals": away_even_strength_goals_allowed,
                            "away_goalie_short_shots": away_short_handed_shots_against,
                            "away_goalie_short_goals": away_short_handed_goals_allowed,
                            "away_goalie_power_shots": away_power_play_shots_against,
                            "away_goalie_power_goals": away_power_play_goals_allowed,
                            "home_save_pctg": home_save_pctg,
                            "away_save_pctg": away_save_pctg,
                            "home_total_pim": home_pim,
                            "away_total_pim": away_pim,
                            "start_time_UTC": box_score['startTimeUTC'],
                            "game_state": box_score['gameState']
                        })
                    except Exception as e:
                        print(f"Skipped game cuz {e}")
                ##after a full team is done, add to csv, clear rows
                if rows:  # only if we actually have data
                    df = pd.DataFrame(rows)
                    if "game_id" in df.columns and not df.empty:
                        df.sort_values(by="game_id", inplace=True)
                    df.to_csv(FILENAME, mode="a", header=False, index=False)
                    print(f"Added {len(rows)} games for {team}")
                    rows = []
                else:
                    print(f"(no valid rows for {team})")
                time.sleep(0.5)
            except Exception as e:
                print(f"Skipped team {team} cuz {e}")

    except Exception as e:
        print(f"something happened cuz {e}")  