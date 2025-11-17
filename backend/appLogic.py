from nba_api.live.nba.endpoints import scoreboard
from models.RandomForestClassifier import trainModel
def getGamesToday():
    games = scoreboard.ScoreBoard().get_dict()

    games_today = games['scoreboard']['games']
    home_teams_id = []
    away_teams_id=[]
    home_teams = []
    away_teams = []
    times = []
    home_teams_scores = []
    away_teams_scores = []
    for game in games_today:
        home_teams_id.append(game['homeTeam']['teamId'])
        away_teams_id.append(game['awayTeam']['teamId'])
        home_teams.append(game['homeTeam']['teamName'])
        away_teams.append(game['awayTeam']['teamName'])
        times.append(game['gameStatusText'])
        home_teams_scores.append(game['homeTeam']['score'])
        away_teams_scores.append(game['awayTeam']['score'])
    return home_teams_id, away_teams_id, home_teams, away_teams, times, home_teams_scores, away_teams_scores

def updateAllGames():
    from data.scrapeRawData import updateAll
    updateAll(current_season=2025)

def retrainModel():
    
    trainModel()

def buildSingleGameFeatures(homeTeams, awayTeams):
    from data.features import getSingleGameFeatureSet
    df = getSingleGameFeatureSet(homeTeams, awayTeams)
    return df
def predictSingleGame(featureDF):
    from models.RandomForestClassifier import predictSingleGame
    return predictSingleGame(featureDF)


if __name__ == "__main__":
    # Example usage
    retrainModel()
