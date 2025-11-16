from nba_api.live.nba.endpoints import scoreboard
from models.RandomForestClassifier import trainModel
def getGamesToday():
    games = scoreboard.ScoreBoard().get_dict()

    games_today = games['scoreboard']['games']
    for game in games_today:
        print(game['homeTeam']['teamId'], "vs", game['awayTeam']['teamId'])
    return games_today  

def updateAllGames():
    from data.scrapeRawData import updateAll
    updateAll(current_season=2025)

def retrainModel():
    
    trainModel()

def buildSingleGameFeatures():
    from data.features import getSingleGameFeatureSet
    df = getSingleGameFeatureSet()
    return df
def predictSingleGame(featureDF):
    from models.RandomForestClassifier import predictSingleGame
    return predictSingleGame(featureDF)

getGamesToday()
updateAllGames()
retrainModel()
